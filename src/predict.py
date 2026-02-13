import io
import torch
import numpy as np
import cv2
from PIL import Image
from pydicom import dcmread
from typing import Optional

import segmentation_models_pytorch as smp
import albumentations as A


class Predictor:
    """Single-image inference for pneumothorax segmentation.

    Handles DICOM and standard image formats. Applies the same
    preprocessing as training (ImageNet normalization).

    Args:
        model_path: Path to saved model checkpoint (.pth).
        device: "cuda" or "cpu".
        threshold: Probability threshold for binarization.
    """

    def __init__(self, model_path: str, device: str = "cpu", threshold: float = 0.5):
        self.device = torch.device(device)
        self.threshold = threshold
        self.image_size = 512
        self.model = self._load_model(model_path)
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model_config = checkpoint.get("config", {})

        model = smp.Unet(
            encoder_name=model_config.get("encoder_name", "efficientnet-b4"),
            encoder_weights=None,
            in_channels=model_config.get("in_channels", 3),
            classes=model_config.get("classes", 1),
            activation=None,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image: np.ndarray, threshold: Optional[float] = None) -> dict:
        """Predict from a numpy array (grayscale uint8).

        Args:
            image: Grayscale image array (H, W), uint8.
            threshold: Override default threshold.

        Returns:
            Dict with mask, probability_map, has_pneumothorax, confidence, original_image.
        """
        if threshold is None:
            threshold = self.threshold

        original = image.copy()
        h_orig, w_orig = image.shape[:2]

        # Resize
        image_resized = cv2.resize(image, (self.image_size, self.image_size))

        # Grayscale to RGB
        image_rgb = np.stack([image_resized] * 3, axis=-1)

        # Normalize
        augmented = self.transform(image=image_rgb)
        image_norm = augmented["image"]

        # To tensor: (H, W, 3) -> (1, 3, H, W)
        tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Binary mask
        mask = (prob_map > threshold).astype(np.uint8)

        # Resize back to original size
        prob_map_full = cv2.resize(prob_map, (w_orig, h_orig))
        mask_full = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        has_pneumothorax = mask_full.sum() > 0
        max_prob = float(prob_map.max())
        confidence = max_prob if has_pneumothorax else 1.0 - max_prob

        return {
            "mask": mask_full,
            "probability_map": prob_map_full,
            "has_pneumothorax": has_pneumothorax,
            "confidence": confidence,
            "original_image": original,
        }

    def predict_dicom(self, dicom_path: str, threshold: Optional[float] = None) -> dict:
        """Predict from a DICOM file.

        Args:
            dicom_path: Path to .dcm file.
            threshold: Override default threshold.

        Returns:
            Same dict as predict_image.
        """
        dcm = dcmread(dicom_path)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Handle inverted DICOM
        if hasattr(dcm, "PhotometricInterpretation"):
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = pixel_array.max() - pixel_array

        # Normalize to 0-255
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
        else:
            pixel_array = np.zeros_like(pixel_array)

        return self.predict_image(pixel_array.astype(np.uint8), threshold=threshold)

    def predict_file(self, file_bytes: bytes, filename: str, threshold: Optional[float] = None) -> dict:
        """Predict from file bytes (for Gradio/web upload).

        Auto-detects DICOM vs standard image format.

        Args:
            file_bytes: Raw file bytes.
            filename: Original filename (used for format detection).
            threshold: Override default threshold.

        Returns:
            Same dict as predict_image.
        """
        if filename.lower().endswith(".dcm"):
            # Write to temp buffer and read as DICOM
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            result = self.predict_dicom(tmp_path, threshold=threshold)
            import os
            os.unlink(tmp_path)
            return result
        else:
            # Standard image format
            img = Image.open(io.BytesIO(file_bytes)).convert("L")
            return self.predict_image(np.array(img), threshold=threshold)
