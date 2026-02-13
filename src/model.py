import torch.nn as nn
import segmentation_models_pytorch as smp

import sys
sys.path.insert(0, "..")
from config import ModelConfig


def create_model(config: ModelConfig = None) -> nn.Module:
    """Create a segmentation model using segmentation_models_pytorch.

    Default: UNet with EfficientNet-B4 encoder pretrained on ImageNet.

    Args:
        config: ModelConfig dataclass with architecture settings.

    Returns:
        PyTorch nn.Module (UNet model).
    """
    if config is None:
        config = ModelConfig()

    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        classes=config.classes,
        activation=config.activation,
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters.

    Returns:
        Dict with total, trainable, frozen counts and millions.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": round(total / 1e6, 2),
        "trainable_millions": round(trainable / 1e6, 2),
    }
