"""Local training script for pneumothorax segmentation."""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import DataConfig, ModelConfig, TrainConfig, AugConfig
from src.dataset import PneumothoraxDataset
from src.transforms import get_training_transforms, get_validation_transforms
from src.model import create_model, count_parameters
from src.train import Trainer, create_stratified_split, seed_everything
from src.evaluate import Evaluator, plot_training_curves


def main():
    # --- Config ---
    SUBSET_DIR = os.path.join(os.path.dirname(__file__), "pneumothorax_subset_tmp")
    CSV_PATH = os.path.join(SUBSET_DIR, "train-rle.csv")
    IMAGE_DIR = os.path.join(SUBSET_DIR, "images")

    # Check data exists (from prepare_subset.py output before zipping)
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found")
        print("Run: python prepare_subset.py --data_dir . first")
        sys.exit(1)

    # Use 256x256 for CPU (4x fewer pixels than 512x512 = much faster)
    IMAGE_SIZE = 256

    data_cfg = DataConfig(csv_path=CSV_PATH, image_dir=IMAGE_DIR, image_size=IMAGE_SIZE)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        batch_size=4,          # smaller for CPU
        num_epochs=1,          # just 1 to test
        learning_rate=1e-4,
        use_amp=False,         # no AMP on CPU
        checkpoint_dir="outputs/checkpoints",
        log_dir="outputs/logs",
    )
    aug_cfg = AugConfig()

    # --- Seed ---
    seed_everything(train_cfg.seed)

    # --- Dataset ---
    print("Loading dataset...")
    full_dataset = PneumothoraxDataset(
        csv_path=data_cfg.csv_path,
        image_dir=data_cfg.image_dir,
        image_size=data_cfg.image_size,
        rle_format="absolute",
    )
    print(f"Total images: {len(full_dataset)}")

    train_idx, val_idx = create_stratified_split(
        full_dataset, train_ratio=data_cfg.train_split, seed=train_cfg.seed
    )

    # Use a small subset for quick local test (200 train, 50 val)
    MAX_TRAIN = 200
    MAX_VAL = 50
    if len(train_idx) > MAX_TRAIN:
        train_idx = train_idx[:MAX_TRAIN]
    if len(val_idx) > MAX_VAL:
        val_idx = val_idx[:MAX_VAL]

    train_dataset = PneumothoraxDataset(
        csv_path=data_cfg.csv_path,
        image_dir=data_cfg.image_dir,
        image_size=data_cfg.image_size,
        transform=get_training_transforms(data_cfg.image_size, aug_cfg),
        rle_format="absolute",
    )
    val_dataset = PneumothoraxDataset(
        csv_path=data_cfg.csv_path,
        image_dir=data_cfg.image_dir,
        image_size=data_cfg.image_size,
        transform=get_validation_transforms(data_cfg.image_size),
        rle_format="absolute",
    )

    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(val_dataset, val_idx)

    train_pos = sum(1 for i in train_idx if full_dataset.get_has_mask(i))
    val_pos = sum(1 for i in val_idx if full_dataset.get_has_mask(i))
    print(f"Train: {len(train_dataset)} ({train_pos} pos)")
    print(f"Val:   {len(val_dataset)} ({val_pos} pos)")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    # --- Model ---
    print("\nCreating model...")
    model = create_model(model_cfg)
    params = count_parameters(model)
    print(f"UNet + {model_cfg.encoder_name}: {params['total_millions']}M params")

    # --- Train ---
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_cfg,
    )
    trainer.train()

    # --- Training curves ---
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    plot_training_curves(trainer.history, os.path.join(train_cfg.log_dir, "training_curves.png"))
    print(f"\nCurves saved to {train_cfg.log_dir}/training_curves.png")


if __name__ == "__main__":
    main()