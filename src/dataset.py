import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from typing import Optional, Callable, Tuple

from src.utils import rle_decode, rle_decode_relative


class PneumothoraxDataset(Dataset):
    """PyTorch Dataset for SIIM-ACR Pneumothorax segmentation.

    Handles DICOM image loading, RLE mask decoding, grayscale-to-RGB
    conversion, and albumentations transforms.

    Args:
        csv_path: Path to CSV with columns [ImageId, EncodedPixels].
        image_dir: Directory containing .dcm files.
        image_size: Target resize dimensions.
        transform: Albumentations Compose object.
        rle_format: "absolute" (Kaggle standard) or "relative" (fin.csv legacy).
        apply_windowing: Whether to apply lung windowing to DICOM.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        image_size: int = 512,
        transform: Optional[Callable] = None,
        rle_format: str = "absolute",
        apply_windowing: bool = False,
    ):
        self.df = pd.read_csv(csv_path)

        # Handle leading space in column name from some CSV exports
        if " EncodedPixels" in self.df.columns:
            self.df = self.df.rename(columns={" EncodedPixels": "EncodedPixels"})

        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        self.rle_format = rle_format
        self.apply_windowing = apply_windowing

        # Get unique image IDs
        self.image_ids = self.df["ImageId"].unique().tolist()

        # Build lookup: image_id -> list of RLE strings
        # (some images have multiple RLE entries for separate pneumothorax regions)
        self._rle_lookup = {}
        for img_id in self.image_ids:
            rows = self.df[self.df["ImageId"] == img_id]
            self._rle_lookup[img_id] = rows["EncodedPixels"].tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]

        # Load image
        image = self._load_image(image_id)

        # Decode mask (merge multiple annotations if present)
        mask = self._decode_mask(image_id)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )

        # Grayscale to RGB (pretrained encoder expects 3 channels)
        image = np.stack([image] * 3, axis=-1)  # (H, W, 3)

        # Apply albumentations transforms (jointly to image and mask)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors: image (H,W,3) -> (3,H,W), mask (H,W) -> (1,H,W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

    def _load_image(self, image_id: str) -> np.ndarray:
        """Load a DICOM or PNG image and return as uint8 grayscale."""
        dcm_path = os.path.join(self.image_dir, f"{image_id}.dcm")
        png_path = os.path.join(self.image_dir, f"{image_id}.png")

        if os.path.exists(dcm_path):
            return self._load_dicom(dcm_path)
        elif os.path.exists(png_path):
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            return img if img is not None else np.zeros((1024, 1024), dtype=np.uint8)
        else:
            raise FileNotFoundError(
                f"Image not found: {dcm_path} or {png_path}"
            )

    def _load_dicom(self, dcm_path: str) -> np.ndarray:
        """Load DICOM file and return pixel array as uint8 grayscale."""
        dcm = dcmread(dcm_path)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Handle inverted DICOM images
        if hasattr(dcm, "PhotometricInterpretation"):
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = pixel_array.max() - pixel_array

        if self.apply_windowing and hasattr(dcm, "RescaleSlope"):
            slope = float(dcm.RescaleSlope)
            intercept = float(dcm.RescaleIntercept)
            pixel_array = pixel_array * slope + intercept
            # Lung window: center=-600, width=1500
            win_min, win_max = -1350, 150
            pixel_array = np.clip(pixel_array, win_min, win_max)
            pixel_array = (pixel_array - win_min) / (win_max - win_min) * 255.0
        else:
            pmin, pmax = pixel_array.min(), pixel_array.max()
            if pmax > pmin:
                pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
            else:
                pixel_array = np.zeros_like(pixel_array)

        return pixel_array.astype(np.uint8)

    def _decode_mask(self, image_id: str) -> np.ndarray:
        """Decode all RLE masks for an image and merge them (logical OR)."""
        rles = self._rle_lookup[image_id]
        mask = np.zeros((1024, 1024), dtype=np.uint8)

        decoder = rle_decode if self.rle_format == "absolute" else rle_decode_relative

        for rle in rles:
            rle_str = str(rle).strip()
            if rle_str in ("-1", "nan", " -1", "", "-1.0"):
                continue
            single_mask = decoder(rle_str)
            mask = np.maximum(mask, single_mask)

        return mask

    def get_has_mask(self, idx: int) -> bool:
        """Check if sample at idx has a positive mask (for stratified splitting)."""
        image_id = self.image_ids[idx]
        rles = self._rle_lookup[image_id]
        return any(
            str(r).strip() not in ("-1", "nan", " -1", "", "-1.0") for r in rles
        )
