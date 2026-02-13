import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    Uses smooth factor to prevent division by zero when both prediction
    and ground truth are all-zeros (common with negative samples).

    Args:
        smooth: Smoothing factor (default 1.0).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output, shape (B, 1, H, W).
            targets: Binary masks, shape (B, 1, H, W).

        Returns:
            Scalar loss value.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy Loss.

    BCE provides pixel-level gradient signal (good for edges).
    Dice provides region-level signal (good for class imbalance).
    The BCE uses pos_weight to upweight positive pixels.

    Args:
        dice_weight: Weight for Dice loss component.
        bce_weight: Weight for BCE loss component.
        pos_weight: Weight for positive class in BCE.
        smooth: Smoothing factor for Dice.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        pos_weight: float = 10.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight_tensor.to(logits.device)
        )
        return self.dice_weight * dice + self.bce_weight * bce
