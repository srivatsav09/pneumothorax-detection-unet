import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import Dict, Optional

from src.metrics import MetricsCalculator


class Evaluator:
    """Comprehensive evaluation on validation/test set.

    Generates:
    - Full metrics report (JSON)
    - Dice score distribution histogram
    - Threshold sensitivity analysis
    - Sample prediction visualizations
    - Positive vs negative image breakdown

    Args:
        model: Trained PyTorch model.
        dataset: Evaluation dataset.
        device: "cuda" or "cpu".
        threshold: Default binarization threshold.
    """

    def __init__(self, model, dataset, device="cuda", threshold=0.5):
        self.model = model.to(device).eval()
        self.dataset = dataset
        self.device = device
        self.threshold = threshold
        self.loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    def run_full_evaluation(self, save_dir: str = "outputs/figures") -> Dict:
        """Run complete evaluation suite.

        Args:
            save_dir: Directory to save figures and reports.

        Returns:
            Dict with all metrics and analysis results.
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. Compute per-image metrics at default threshold
        per_image_dice, per_image_iou, gt_has_mask_flags = self._compute_per_image_metrics()

        # 2. Aggregate metrics at default threshold
        metrics_calc = MetricsCalculator(threshold=self.threshold)
        self._run_through_loader(metrics_calc)
        default_metrics = metrics_calc.compute()

        # 3. Threshold sensitivity analysis
        threshold_results = self._threshold_analysis()

        # 4. Find optimal threshold
        best_thresh_data = max(threshold_results, key=lambda x: x["dice_mean"])
        optimal_threshold = best_thresh_data["threshold"]

        # 5. Positive vs negative breakdown
        pos_dices = [d for d, has in zip(per_image_dice, gt_has_mask_flags) if has]
        neg_dices = [d for d, has in zip(per_image_dice, gt_has_mask_flags) if not has]

        # Build full results
        results = {
            "model": "UNet-EfficientNetB4",
            "dataset_size": len(self.dataset),
            "default_threshold": self.threshold,
            "optimal_threshold": round(optimal_threshold, 2),
            "metrics_at_default_threshold": default_metrics,
            "metrics_at_optimal_threshold": best_thresh_data,
            "positive_images": {
                "count": len(pos_dices),
                "dice_mean": float(np.mean(pos_dices)) if pos_dices else 0.0,
                "dice_std": float(np.std(pos_dices)) if pos_dices else 0.0,
            },
            "negative_images": {
                "count": len(neg_dices),
                "dice_mean": float(np.mean(neg_dices)) if neg_dices else 0.0,
                "dice_std": float(np.std(neg_dices)) if neg_dices else 0.0,
            },
        }

        # 6. Generate plots
        self._plot_dice_distribution(per_image_dice, gt_has_mask_flags, save_dir)
        self._plot_threshold_analysis(threshold_results, save_dir)
        self._plot_sample_predictions(save_dir, n=8)

        # 7. Save report
        report_path = os.path.join(save_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Report saved to {report_path}")

        return results

    def _run_through_loader(self, metrics_calc: MetricsCalculator):
        """Run model through dataloader and update metrics calculator."""
        with torch.no_grad():
            for images, masks in self.loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                with autocast(enabled=self.device != "cpu"):
                    logits = self.model(images)
                metrics_calc.update(logits, masks)

    def _compute_per_image_metrics(self):
        """Compute dice and iou for each image individually."""
        per_image_dice = []
        per_image_iou = []
        gt_has_mask_flags = []
        smooth = 1e-6

        with torch.no_grad():
            for images, masks in self.loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                with autocast(enabled=self.device != "cpu"):
                    logits = self.model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > self.threshold).float()

                for i in range(images.shape[0]):
                    pred_flat = preds[i].cpu().numpy().flatten()
                    gt_flat = masks[i].cpu().numpy().flatten()

                    intersection = (pred_flat * gt_flat).sum()
                    dice = (2 * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
                    union = pred_flat.sum() + gt_flat.sum() - intersection
                    iou = (intersection + smooth) / (union + smooth)

                    per_image_dice.append(float(dice))
                    per_image_iou.append(float(iou))
                    gt_has_mask_flags.append(gt_flat.sum() > 0)

        return per_image_dice, per_image_iou, gt_has_mask_flags

    def _threshold_analysis(self):
        """Evaluate metrics at multiple thresholds."""
        thresholds = np.arange(0.1, 0.95, 0.05)
        results = []

        for t in thresholds:
            calc = MetricsCalculator(threshold=float(t))
            self._run_through_loader(calc)
            metrics = calc.compute()
            metrics["threshold"] = round(float(t), 2)
            results.append(metrics)

        return results

    def _plot_dice_distribution(self, dices, has_mask_flags, save_dir):
        """Histogram of per-image Dice scores split by positive/negative."""
        fig, ax = plt.subplots(figsize=(10, 6))

        pos_dices = [d for d, has in zip(dices, has_mask_flags) if has]
        neg_dices = [d for d, has in zip(dices, has_mask_flags) if not has]

        if pos_dices:
            ax.hist(pos_dices, bins=30, alpha=0.7, label=f"Positive (n={len(pos_dices)})", color="coral")
        if neg_dices:
            ax.hist(neg_dices, bins=30, alpha=0.7, label=f"Negative (n={len(neg_dices)})", color="steelblue")

        ax.set_xlabel("Dice Score")
        ax.set_ylabel("Count")
        ax.set_title("Per-Image Dice Score Distribution")
        ax.legend()
        ax.set_xlim(0, 1)

        fig.savefig(os.path.join(save_dir, "dice_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_threshold_analysis(self, results, save_dir):
        """Plot Dice/IoU/Detection F1 vs threshold."""
        thresholds = [r["threshold"] for r in results]
        dices = [r["dice_mean"] for r in results]
        ious = [r["iou_mean"] for r in results]
        det_f1s = [r["detection_f1"] for r in results]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, dices, "o-", label="Dice", linewidth=2)
        ax.plot(thresholds, ious, "s-", label="IoU", linewidth=2)
        ax.plot(thresholds, det_f1s, "^-", label="Detection F1", linewidth=2)

        best_idx = np.argmax(dices)
        ax.axvline(thresholds[best_idx], color="red", linestyle="--", alpha=0.5,
                    label=f"Optimal threshold={thresholds[best_idx]:.2f}")

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Threshold Sensitivity Analysis")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        fig.savefig(os.path.join(save_dir, "threshold_analysis.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_sample_predictions(self, save_dir, n=8):
        """Grid showing best and worst predictions."""
        per_image_dice, _, gt_has_mask = self._compute_per_image_metrics()

        # Get positive images only (segmentation quality matters most here)
        pos_indices = [i for i, has in enumerate(gt_has_mask) if has]
        if len(pos_indices) == 0:
            return

        pos_dices = [(per_image_dice[i], i) for i in pos_indices]
        pos_dices.sort(key=lambda x: x[0])

        # Pick worst n//2 and best n//2
        n_half = min(n // 2, len(pos_dices))
        selected = pos_dices[:n_half] + pos_dices[-n_half:]

        fig, axes = plt.subplots(3, len(selected), figsize=(4 * len(selected), 12))
        if len(selected) == 1:
            axes = axes[:, np.newaxis]

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        with torch.no_grad():
            for col, (dice_val, idx) in enumerate(selected):
                img_tensor, mask_tensor = self.dataset[idx]
                img_input = img_tensor.unsqueeze(0).to(self.device)
                with autocast(enabled=self.device != "cpu"):
                    logit = self.model(img_input)
                pred = (torch.sigmoid(logit) > self.threshold).float()

                # Denormalize image for display
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = img_np * imagenet_std + imagenet_mean
                img_np = np.clip(img_np, 0, 1)

                mask_np = mask_tensor.squeeze().numpy()
                pred_np = pred.squeeze().cpu().numpy()

                axes[0, col].imshow(img_np)
                axes[0, col].set_title(f"Image (idx={idx})")
                axes[0, col].axis("off")

                axes[1, col].imshow(mask_np, cmap="Reds", vmin=0, vmax=1)
                axes[1, col].set_title("Ground Truth")
                axes[1, col].axis("off")

                axes[2, col].imshow(pred_np, cmap="Greens", vmin=0, vmax=1)
                axes[2, col].set_title(f"Pred (Dice={dice_val:.3f})")
                axes[2, col].axis("off")

        plt.suptitle("Worst (left) vs Best (right) Predictions on Positive Images", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "sample_predictions.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_training_curves(history: dict, save_path: str):
    """Plot training curves from training history dict.

    Args:
        history: Dict with keys train_loss, val_loss, train_dice, val_dice, lr, etc.
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(history["train_dice"], label="Train")
    axes[1].plot(history["val_dice"], label="Validation")
    axes[1].set_title("Dice Coefficient")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(history["lr"])
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
