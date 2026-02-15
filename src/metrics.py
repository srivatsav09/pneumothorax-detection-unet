import torch
import numpy as np
from typing import Dict, List


class MetricsCalculator:
    """Tracks and computes segmentation metrics across batches.

    Metrics:
    - Dice Coefficient (per-image, mean + std)
    - IoU / Jaccard Index (per-image, mean + std)
    - Pixel-level Precision, Recall, F1
    - Per-image Detection Accuracy, Precision, Recall, F1
      (treats each image as binary: is pneumothorax present?)

    Args:
        threshold: Probability threshold for binarization.
        smooth: Small value to prevent division by zero.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()

    def reset(self):
        """Reset all accumulators for a new epoch."""
        self.dice_scores: List[float] = []
        self.iou_scores: List[float] = []
        self.pixel_tp: int = 0
        self.pixel_fp: int = 0
        self.pixel_fn: int = 0
        self.pixel_tn: int = 0
        self.det_tp: int = 0
        self.det_fp: int = 0
        self.det_fn: int = 0
        self.det_tn: int = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update metrics with a batch of predictions.

        Args:
            logits: Raw model output (B, 1, H, W).
            targets: Ground truth masks (B, 1, H, W).
        """
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        batch_size = logits.shape[0]

        for i in range(batch_size):
            pred_i = preds[i].cpu().numpy().flatten()
            target_i = targets[i].cpu().numpy().flatten()

            # Dice - handle true negatives (both empty) as perfect score
            gt_empty = target_i.sum() == 0
            pred_empty = pred_i.sum() == 0
            intersection = (pred_i * target_i).sum()

            if gt_empty and pred_empty:
                dice = 1.0  # Both empty = perfect
                iou = 1.0
            elif gt_empty or pred_empty:
                dice = 0.0  # One empty, one not = no overlap
                iou = 0.0
            else:
                dice = (2.0 * intersection + self.smooth) / (
                    pred_i.sum() + target_i.sum() + self.smooth
                )
                union = pred_i.sum() + target_i.sum() - intersection
                iou = (intersection + self.smooth) / (union + self.smooth)

            self.dice_scores.append(float(dice))
            self.iou_scores.append(float(iou))

            # Pixel-level confusion matrix
            pred_bool = pred_i.astype(bool)
            target_bool = target_i.astype(bool)
            self.pixel_tp += int((pred_bool & target_bool).sum())
            self.pixel_fp += int((pred_bool & ~target_bool).sum())
            self.pixel_fn += int((~pred_bool & target_bool).sum())
            self.pixel_tn += int((~pred_bool & ~target_bool).sum())

            # Per-image detection
            gt_has_mask = target_i.sum() > 0
            pred_has_mask = pred_i.sum() > 0

            if gt_has_mask and pred_has_mask:
                self.det_tp += 1
            elif not gt_has_mask and pred_has_mask:
                self.det_fp += 1
            elif gt_has_mask and not pred_has_mask:
                self.det_fn += 1
            else:
                self.det_tn += 1

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated values."""
        results = {}

        # Dice
        results["dice_mean"] = float(np.mean(self.dice_scores)) if self.dice_scores else 0.0
        results["dice_std"] = float(np.std(self.dice_scores)) if self.dice_scores else 0.0

        # IoU
        results["iou_mean"] = float(np.mean(self.iou_scores)) if self.iou_scores else 0.0
        results["iou_std"] = float(np.std(self.iou_scores)) if self.iou_scores else 0.0

        # Pixel precision/recall/F1
        results["pixel_precision"] = self.pixel_tp / max(self.pixel_tp + self.pixel_fp, 1)
        results["pixel_recall"] = self.pixel_tp / max(self.pixel_tp + self.pixel_fn, 1)
        pr_sum = results["pixel_precision"] + results["pixel_recall"]
        results["pixel_f1"] = 2 * results["pixel_precision"] * results["pixel_recall"] / max(pr_sum, self.smooth)

        # Detection metrics (per-image)
        total = self.det_tp + self.det_fp + self.det_fn + self.det_tn
        results["detection_accuracy"] = (self.det_tp + self.det_tn) / max(total, 1)
        results["detection_precision"] = self.det_tp / max(self.det_tp + self.det_fp, 1)
        results["detection_recall"] = self.det_tp / max(self.det_tp + self.det_fn, 1)
        det_pr = results["detection_precision"] + results["detection_recall"]
        results["detection_f1"] = 2 * results["detection_precision"] * results["detection_recall"] / max(det_pr, self.smooth)

        results["num_samples"] = len(self.dice_scores)
        return results
