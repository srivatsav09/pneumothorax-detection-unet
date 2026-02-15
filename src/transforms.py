import cv2
import albumentations as A

import sys
sys.path.insert(0, "..")
from config import AugConfig


def get_training_transforms(image_size: int = 512, config: AugConfig = None) -> A.Compose:
    """Training augmentation pipeline applied jointly to image and mask.

    Key medical imaging considerations:
    - HorizontalFlip is safe (pneumothorax can occur on either side)
    - VerticalFlip is NOT used (chest X-rays have fixed orientation)
    - ElasticTransform simulates anatomical variation
    - CLAHE improves contrast in lung regions
    """
    if config is None:
        config = AugConfig()

    return A.Compose([
        A.HorizontalFlip(p=config.horizontal_flip_p),
        A.Affine(
            translate_percent={"x": (-config.shift_limit, config.shift_limit),
                              "y": (-config.shift_limit, config.shift_limit)},
            scale=(1 - config.scale_limit, 1 + config.scale_limit),
            rotate=(-config.rotate_limit, config.rotate_limit),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=config.elastic_alpha,
            sigma=config.elastic_sigma,
            p=config.elastic_p,
        ),
        A.OneOf([
            A.CLAHE(clip_limit=config.clahe_clip_limit, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                p=1.0,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        A.GaussNoise(
            std_range=(0.02, 0.1),
            p=config.gauss_noise_p,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_validation_transforms(image_size: int = 512) -> A.Compose:
    """Validation/test transforms - only normalization, no augmentation."""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
