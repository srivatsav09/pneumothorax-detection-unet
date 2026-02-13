import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import train_test_split

from src.losses import DiceBCELoss
from src.metrics import MetricsCalculator

import sys
sys.path.insert(0, "..")
from config import TrainConfig


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_stratified_split(dataset, train_ratio: float = 0.8, seed: int = 42):
    """Create stratified train/val split preserving positive/negative ratio.

    Args:
        dataset: PneumothoraxDataset instance.
        train_ratio: Proportion of data for training.
        seed: Random seed.

    Returns:
        (train_indices, val_indices) as lists.
    """
    labels = [1 if dataset.get_has_mask(i) else 0 for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=1.0 - train_ratio, stratify=labels, random_state=seed
    )
    return train_idx, val_idx


class Trainer:
    """Complete training pipeline with mixed precision, early stopping,
    cosine annealing with warmup, and comprehensive logging.

    Args:
        model: PyTorch segmentation model.
        train_dataset: Training dataset (or Subset).
        val_dataset: Validation dataset (or Subset).
        config: TrainConfig dataclass.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config: TrainConfig = None,
    ):
        if config is None:
            config = TrainConfig()
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Loss
        self.criterion = DiceBCELoss(
            dice_weight=config.dice_weight,
            bce_weight=config.bce_weight,
            pos_weight=config.pos_weight,
        ).to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler: linear warmup + cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs - config.warmup_epochs,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs],
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=config.use_amp)
        self.use_amp = config.use_amp

        # DataLoaders
        num_workers = 2 if torch.cuda.is_available() else 0
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=config.use_amp and torch.cuda.is_available(),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.use_amp and torch.cuda.is_available(),
        )

        # Metrics
        self.train_metrics = MetricsCalculator()
        self.val_metrics = MetricsCalculator()

        # Tracking
        self.best_dice = 0.0
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
            "train_iou": [],
            "val_iou": [],
            "lr": [],
        }

    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.grad_accum_steps
            num_batches += 1

            with torch.no_grad():
                self.train_metrics.update(logits.detach(), masks)

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate_epoch(self) -> float:
        """Validate for one epoch. Returns average loss."""
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            total_loss += loss.item()
            num_batches += 1
            self.val_metrics.update(logits, masks)

        return total_loss / max(num_batches, 1)

    def train(self):
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.batch_size}, Epochs: {self.config.num_epochs}")
        print("-" * 80)

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch()
            train_results = self.train_metrics.compute()

            val_loss = self.validate_epoch()
            val_results = self.val_metrics.compute()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_dice"].append(train_results["dice_mean"])
            self.history["val_dice"].append(val_results["dice_mean"])
            self.history["train_iou"].append(train_results["iou_mean"])
            self.history["val_iou"].append(val_results["iou_mean"])
            self.history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start

            print(
                f"Epoch {epoch+1:3d}/{self.config.num_epochs} "
                f"[{elapsed:.1f}s] "
                f"loss: {train_loss:.4f}/{val_loss:.4f} "
                f"dice: {train_results['dice_mean']:.4f}/{val_results['dice_mean']:.4f} "
                f"iou: {train_results['iou_mean']:.4f}/{val_results['iou_mean']:.4f} "
                f"det_acc: {val_results['detection_accuracy']:.4f} "
                f"lr: {current_lr:.2e}"
            )

            # Checkpoint best model
            if val_results["dice_mean"] > self.best_dice:
                self.best_dice = val_results["dice_mean"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_results)
                print(f"  -> New best dice: {self.best_dice:.4f}, saved checkpoint")
            else:
                self.patience_counter += 1

            # Save history after every epoch (in case of disconnection)
            self._save_history()

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\nTraining complete. Best val dice: {self.best_dice:.4f}")

    def _save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dice": self.best_dice,
            "metrics": metrics,
            "config": {
                "encoder_name": "efficientnet-b4",
                "image_size": 512,
                "in_channels": 3,
                "classes": 1,
            },
        }
        path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, path)

    def _save_history(self):
        """Save training history as JSON."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        path = os.path.join(self.config.log_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
