from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    csv_path: str = "data/train-rle.csv"
    image_dir: str = "data/dicom_images/"
    image_size: int = 512
    original_size: int = 1024
    train_split: float = 0.8
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class ModelConfig:
    architecture: str = "Unet"
    encoder_name: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 1
    activation: Optional[str] = None


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    pos_weight: float = 10.0
    grad_accum_steps: int = 1
    use_amp: bool = True
    early_stopping_patience: int = 10
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    seed: int = 42


@dataclass
class AugConfig:
    horizontal_flip_p: float = 0.5
    shift_limit: float = 0.0625
    scale_limit: float = 0.1
    rotate_limit: int = 15
    elastic_alpha: float = 120.0
    elastic_sigma: float = 6.0
    elastic_p: float = 0.3
    clahe_clip_limit: float = 2.0
    clahe_p: float = 0.3
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1
    gauss_noise_var_limit: tuple = (5.0, 30.0)
    gauss_noise_p: float = 0.2


@dataclass
class InferenceConfig:
    model_path: str = "outputs/checkpoints/best_model.pth"
    threshold: float = 0.5
    tta: bool = False
    device: str = "cuda"
