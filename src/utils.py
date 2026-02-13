import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def rle_decode(rle_string: str, shape: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Decode RLE mask using absolute positions (SIIM-ACR Kaggle format).

    RLE is 1-indexed, column-major (Fortran) order.

    Args:
        rle_string: Space-separated "start length start length ..."
        shape: (height, width) of the original image.

    Returns:
        Binary mask of shape (height, width), dtype uint8.
    """
    rle_str = str(rle_string).strip()
    if rle_str in ("-1", "", " -1", "nan", "-1.0"):
        return np.zeros(shape, dtype=np.uint8)

    s = rle_str.split()
    starts = np.array(s[0::2], dtype=int) - 1  # convert to 0-indexed
    lengths = np.array(s[1::2], dtype=int)

    mask_flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask_flat[start : start + length] = 1

    return mask_flat.reshape(shape, order="F")


def rle_decode_relative(rle_string: str, shape: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Decode RLE mask using relative offsets (fin.csv legacy format).

    Each start value is relative to the end of the previous run.

    Args:
        rle_string: Space-separated "offset length offset length ..."
        shape: (height, width) of the original image.

    Returns:
        Binary mask of shape (height, width), dtype uint8.
    """
    rle_str = str(rle_string).strip()
    if rle_str in ("-1", "", " -1", "nan", "-1.0"):
        return np.zeros(shape, dtype=np.uint8)

    s = rle_str.split()
    starts = np.array(s[0::2], dtype=int)
    lengths = np.array(s[1::2], dtype=int)

    mask_flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    current_pos = 0
    for start, length in zip(starts, lengths):
        current_pos += start
        mask_flat[current_pos : current_pos + length] = 1
        current_pos += length

    return mask_flat.reshape(shape).T


def rle_encode(mask: np.ndarray) -> str:
    """Encode binary mask to RLE string (absolute, 1-indexed, column-major).

    Args:
        mask: Binary mask of shape (height, width).

    Returns:
        RLE string or "-1" if mask is empty.
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    if len(runs) == 0:
        return "-1"
    runs[0::2] += 1  # convert to 1-indexed
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


def visualize_prediction(
    image: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    dice_score: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a 3-panel visualization: image, ground truth overlay, prediction overlay.

    Args:
        image: Grayscale or RGB image array (H, W) or (H, W, 3).
        mask_gt: Ground truth binary mask (H, W).
        mask_pred: Predicted binary mask (H, W).
        dice_score: Dice score for this prediction.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    if image.ndim == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Ground truth overlay
    if image.ndim == 3:
        overlay_gt = image.copy()
    else:
        overlay_gt = np.stack([image] * 3, axis=-1)
    overlay_gt[mask_gt > 0] = [255, 0, 0]
    axes[1].imshow(overlay_gt)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Prediction overlay
    if image.ndim == 3:
        overlay_pred = image.copy()
    else:
        overlay_pred = np.stack([image] * 3, axis=-1)
    overlay_pred[mask_pred > 0] = [0, 255, 0]
    axes[2].imshow(overlay_pred)
    axes[2].set_title(f"Prediction (Dice: {dice_score:.4f})")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_batch(
    images: list,
    masks_gt: list,
    masks_pred: list,
    dice_scores: list,
    n: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize a grid of n predictions.

    Args:
        images: List of image arrays.
        masks_gt: List of ground truth masks.
        masks_pred: List of predicted masks.
        dice_scores: List of dice scores.
        n: Number of samples to show.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes[:, np.newaxis]

    for i in range(n):
        img = images[i]
        gt = masks_gt[i]
        pred = masks_pred[i]

        # Row 0: original
        if img.ndim == 3:
            axes[0, i].imshow(img)
        else:
            axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis("off")

        # Row 1: ground truth
        axes[1, i].imshow(gt, cmap="Reds", vmin=0, vmax=1)
        axes[1, i].set_title(f"GT (px={gt.sum():.0f})")
        axes[1, i].axis("off")

        # Row 2: prediction
        axes[2, i].imshow(pred, cmap="Greens", vmin=0, vmax=1)
        axes[2, i].set_title(f"Pred (Dice={dice_scores[i]:.3f})")
        axes[2, i].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
