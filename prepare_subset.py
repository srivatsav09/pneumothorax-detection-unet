"""
Prepare a balanced ~2500 image subset from the SIIM-ACR Pneumothorax dataset.

Usage:
    1. Download the dataset from:
       https://www.kaggle.com/datasets/jesperdramsch/siim-acr-pneumothorax-segmentation-data
    2. Extract it somewhere (e.g., C:/Users/sriva/Downloads/siim-acr-data/)
    3. Run this script:
       python prepare_subset.py --data_dir "C:/Users/sriva/Downloads/siim-acr-data/"
    4. Upload the output zip to Google Drive -> open in Colab

Output: pneumothorax_subset.zip containing:
    - train-rle.csv  (filtered to subset images only)
    - images/         (~2500 .dcm or .png files)
"""

import os
import sys
import shutil
import argparse
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path


def find_csv(data_dir):
    """Find train-rle.csv in the data directory."""
    candidates = list(Path(data_dir).rglob("train-rle.csv"))
    if not candidates:
        # Also check for other common names
        candidates = list(Path(data_dir).rglob("*rle*.csv"))
    if candidates:
        return str(candidates[0])
    return None


def find_images(data_dir):
    """Build a lookup of ImageId -> file path for DICOM and PNG files."""
    lookup = {}
    for ext in ("*.dcm", "*.png"):
        for f in Path(data_dir).rglob(ext):
            lookup[f.stem] = str(f)
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced pneumothorax subset")
    parser.add_argument("--data_dir", required=True, help="Path to extracted dataset folder")
    parser.add_argument("--output", default="pneumothorax_subset.zip", help="Output zip filename")
    parser.add_argument("--num_images", default=2500, type=int, help="Target total images (pos + neg)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Find CSV
    csv_path = find_csv(args.data_dir)
    if not csv_path:
        print(f"ERROR: Could not find train-rle.csv in {args.data_dir}")
        print("Make sure you extracted the Kaggle dataset correctly.")
        sys.exit(1)
    print(f"CSV: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    if " EncodedPixels" in df.columns:
        df = df.rename(columns={" EncodedPixels": "EncodedPixels"})
    print(f"Total CSV rows: {len(df)}, Unique images: {df['ImageId'].nunique()}")

    # Find image files
    image_lookup = find_images(args.data_dir)
    print(f"Found {len(image_lookup)} image files on disk")

    # Classify positive vs negative
    positive_ids = []
    negative_ids = []
    for img_id in df["ImageId"].unique():
        rles = df[df["ImageId"] == img_id]["EncodedPixels"].tolist()
        has_mask = any(str(r).strip() not in ("-1", "nan", " -1", "", "-1.0") for r in rles)
        if has_mask:
            positive_ids.append(img_id)
        else:
            negative_ids.append(img_id)

    print(f"\nPositive (with pneumothorax): {len(positive_ids)}")
    print(f"Negative (no pneumothorax):   {len(negative_ids)}")

    # Balance: take all positive + equal number of negative
    num_pos = len(positive_ids)
    num_neg = min(num_pos, len(negative_ids))

    # If user wants fewer total images, reduce both proportionally
    if args.num_images < num_pos + num_neg:
        half = args.num_images // 2
        num_pos = min(half, len(positive_ids))
        num_neg = min(args.num_images - num_pos, len(negative_ids))
        positive_ids = list(np.random.choice(positive_ids, size=num_pos, replace=False))

    neg_sampled = list(np.random.choice(negative_ids, size=num_neg, replace=False))
    selected_ids = set(positive_ids + neg_sampled)

    print(f"\nSelected subset: {len(selected_ids)} images ({len(positive_ids)} pos + {len(neg_sampled)} neg)")

    # Check which selected images exist on disk
    available = {img_id for img_id in selected_ids if img_id in image_lookup}
    missing = selected_ids - available
    if missing:
        print(f"WARNING: {len(missing)} images not found on disk, skipping them")
    selected_ids = available

    print(f"Final subset: {len(selected_ids)} images")

    # Create output directory
    output_dir = "pneumothorax_subset_tmp"
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Filter and save CSV
    df_subset = df[df["ImageId"].isin(selected_ids)]
    csv_out = os.path.join(output_dir, "train-rle.csv")
    df_subset.to_csv(csv_out, index=False)
    print(f"\nCSV saved: {csv_out} ({len(df_subset)} rows)")

    # Copy image files
    print("Copying images...")
    copied = 0
    for img_id in selected_ids:
        src = image_lookup[img_id]
        ext = Path(src).suffix
        dst = os.path.join(images_dir, f"{img_id}{ext}")
        shutil.copy2(src, dst)
        copied += 1
        if copied % 500 == 0:
            print(f"  {copied}/{len(selected_ids)}")

    print(f"Copied {copied} images")

    # Create zip
    print(f"\nCreating {args.output}...")
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, fnames in os.walk(output_dir):
            for fname in fnames:
                filepath = os.path.join(root, fname)
                arcname = os.path.relpath(filepath, output_dir)
                zf.write(filepath, arcname)

    zip_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done! {args.output} ({zip_size:.0f} MB)")
    print(f"\nUpload this zip to Google Drive, then in Colab:")
    print(f"  !unzip -q /content/drive/MyDrive/{args.output} -d /content/pneumothorax_subset/")

    # Keep temp dir for local training
    print(f"\nSubset folder kept at: {output_dir}/")
    print("Run local training with: python train_local.py")


if __name__ == "__main__":
    main()