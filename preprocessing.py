"""
Improved DICOM Preprocessing for Pneumothorax Detection
Removes edge artifacts, applies proper windowing, and prepares clean training data
"""

import numpy as np
from PIL import Image
from pydicom import dcmread
import cv2


def apply_lung_window(pixel_array, window_center=-600, window_width=1500):
    """
    Apply lung windowing to DICOM pixel array.
    Standard lung window: Center=-600 HU, Width=1500 HU
    This emphasizes lung tissue and suppresses other structures.

    Args:
        pixel_array: Raw DICOM pixel array
        window_center: Window center in Hounsfield Units
        window_width: Window width in Hounsfield Units

    Returns:
        Windowed pixel array normalized to 0-255
    """
    # Calculate window min and max
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2

    # Apply windowing
    windowed = np.clip(pixel_array, window_min, window_max)

    # Normalize to 0-255
    windowed = ((windowed - window_min) / (window_max - window_min) * 255.0).astype(np.uint8)

    return windowed


def remove_text_annotations(image, border_crop_percent=0.05):
    """
    Remove text annotations typically found at image borders (PORTABLE, AP, etc.)
    by cropping the outer border and resizing back.

    Args:
        image: PIL Image or numpy array
        border_crop_percent: Percentage of border to crop (0.05 = 5%)

    Returns:
        Cropped and resized image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    width, height = image.size

    # Calculate crop box (remove 5% from each edge)
    left = int(width * border_crop_percent)
    top = int(height * border_crop_percent)
    right = int(width * (1 - border_crop_percent))
    bottom = int(height * (1 - border_crop_percent))

    # Crop and resize back to original dimensions
    cropped = image.crop((left, top, right, bottom))
    resized = cropped.resize((width, height), Image.LANCZOS)

    return resized


def center_crop_thorax(image, crop_ratio=0.9):
    """
    Center crop to focus on thorax region, removing edge artifacts.

    Args:
        image: PIL Image or numpy array
        crop_ratio: Ratio of image to keep (0.9 = keep center 90%)

    Returns:
        Center-cropped and resized image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    width, height = image.size

    # Calculate center crop dimensions
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Crop and resize back
    cropped = image.crop((left, top, right, bottom))
    resized = cropped.resize((width, height), Image.LANCZOS)

    return resized


def preprocess_dicom_for_training(dicom_path, resize_to=512, apply_windowing=True,
                                  remove_borders=True, center_crop=True):
    """
    Complete preprocessing pipeline for DICOM chest X-rays.

    Args:
        dicom_path: Path to DICOM file
        resize_to: Target size for resizing
        apply_windowing: Whether to apply lung windowing
        remove_borders: Whether to remove border annotations
        center_crop: Whether to apply center cropping

    Returns:
        Preprocessed image as numpy array (resize_to, resize_to)
    """
    # Load DICOM
    dcm = dcmread(dicom_path)
    pixel_array = dcm.pixel_array

    # Apply lung windowing if enabled
    if apply_windowing and hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
        # Convert to Hounsfield Units
        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)
        pixel_array = pixel_array * slope + intercept
        pixel_array = apply_lung_window(pixel_array)
    else:
        # Simple normalization if no HU calibration available
        pixel_array = ((pixel_array - pixel_array.min()) /
                      (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

    # Convert to PIL for easier manipulation
    image = Image.fromarray(pixel_array)

    # Remove border annotations
    if remove_borders:
        image = remove_text_annotations(image, border_crop_percent=0.05)

    # Center crop to focus on thorax
    if center_crop:
        image = center_crop_thorax(image, crop_ratio=0.92)

    # Resize to target size
    if image.size != (resize_to, resize_to):
        image = image.resize((resize_to, resize_to), Image.LANCZOS)

    return np.array(image)


def preprocess_dicom_for_inference(dicom_path_or_file, resize_to=512):
    """
    Preprocess DICOM for inference (matches training preprocessing).

    Args:
        dicom_path_or_file: Path to DICOM file or file-like object
        resize_to: Target size

    Returns:
        RGB array ready for model input (1, resize_to, resize_to, 3)
    """
    try:
        # Load DICOM
        if isinstance(dicom_path_or_file, str):
            dcm = dcmread(dicom_path_or_file)
        else:
            dcm = dcmread(dicom_path_or_file)

        pixel_array = dcm.pixel_array

        # Apply same preprocessing as training
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            slope = float(dcm.RescaleSlope)
            intercept = float(dcm.RescaleIntercept)
            pixel_array = pixel_array * slope + intercept
            pixel_array = apply_lung_window(pixel_array)
        else:
            pixel_array = ((pixel_array - pixel_array.min()) /
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

        image = Image.fromarray(pixel_array)
        image = remove_text_annotations(image, border_crop_percent=0.05)
        image = center_crop_thorax(image, crop_ratio=0.92)
        image = image.resize((resize_to, resize_to), Image.LANCZOS)

        # Convert to RGB for model
        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)
        rgb_array = np.expand_dims(rgb_array, axis=0)
        rgb_array = rgb_array.astype(np.float32) / 255.0

        return rgb_array, image

    except Exception as e:
        raise ValueError(f"Error preprocessing DICOM: {str(e)}")


def create_bounding_box_from_mask(rle_mask, image_width=1024, image_height=1024):
    """
    Convert RLE mask to bounding box coordinates.

    Args:
        rle_mask: RLE encoded mask string
        image_width: Original image width
        image_height: Original image height

    Returns:
        Normalized bounding box [x_min, y_min, x_max, y_max] or None if no mask
    """
    if rle_mask == '-1' or not rle_mask or rle_mask == ' -1':
        return None

    # Decode RLE to binary mask
    mask_array = rle_to_mask(rle_mask, image_width, image_height)

    if mask_array.sum() == 0:
        return None

    # Find bounding box of mask
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Normalize to 0-1 range
    x_min_norm = x_min / image_width
    y_min_norm = y_min / image_height
    x_max_norm = x_max / image_width
    y_max_norm = y_max / image_height

    return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]


def rle_to_mask(rle_string, width, height):
    """
    Convert RLE encoding to binary mask.

    Args:
        rle_string: RLE encoded string
        width: Image width
        height: Image height

    Returns:
        Binary mask array (height, width)
    """
    mask = np.zeros(width * height, dtype=np.uint8)

    if rle_string == '-1' or not rle_string or rle_string == ' -1':
        return mask.reshape(height, width)

    array = np.array([int(x) for x in rle_string.split()])
    starts = array[0::2] - 1  # RLE is 1-indexed
    lengths = array[1::2]

    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1

    return mask.reshape(height, width, order='F')  # Fortran order (column-major)
