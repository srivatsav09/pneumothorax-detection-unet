# Pneumothorax Segmentation with UNet + EfficientNet-B4

Automated pneumothorax detection and segmentation from chest X-rays using a UNet architecture with an EfficientNet-B4 encoder, trained on the [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/) dataset.

## Architecture

```
Chest X-Ray (DICOM/PNG) -> Preprocessing -> UNet (EfficientNet-B4 encoder, ImageNet pretrained)
    -> 512x512 segmentation mask -> Post-processing -> Overlay + Metrics
```

- **Encoder:** EfficientNet-B4 pretrained on ImageNet (~19M params)
- **Decoder:** UNet with skip connections
- **Loss:** Combined Dice + Binary Cross-Entropy (with pos_weight for class imbalance)
- **Training:** AdamW + Cosine Annealing LR with warmup, mixed precision (AMP)
- **Library:** [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)

## Project Structure

```
├── config.py                  # Dataclass-based configuration
├── requirements.txt           # Dependencies
├── setup_env.sh / .bat        # Virtual environment setup
├── src/
│   ├── dataset.py             # PyTorch Dataset + DICOM loading + RLE decoding
│   ├── transforms.py          # Albumentations augmentation pipelines
│   ├── model.py               # UNet model factory
│   ├── losses.py              # Dice + BCE combined loss
│   ├── metrics.py             # Dice, IoU, pixel precision/recall, detection accuracy
│   ├── train.py               # Trainer with AMP, early stopping, checkpointing
│   ├── evaluate.py            # Full evaluation suite + visualizations
│   ├── predict.py             # Single-image inference
│   └── utils.py               # RLE encode/decode, visualization helpers
├── notebooks/
│   └── train_colab.ipynb      # Colab Pro training notebook
└── app/
    ├── app.py                 # Gradio app for HuggingFace Spaces
    └── requirements.txt       # App dependencies
```

## Quick Start

### Local Setup

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/pneumothorax-detection-unet.git
cd pneumothorax-detection-unet
git checkout pytorch-segmentation

# Create virtual environment
# Linux/Mac:
bash setup_env.sh
# Windows:
setup_env.bat
```

### Training on Colab Pro

1. Upload the SIIM-ACR dataset to Google Drive
2. Open `notebooks/train_colab.ipynb` in Google Colab
3. Configure the data paths in Cell 2
4. Run all cells

### Running the Gradio App

```bash
# Place your trained model at app/best_model.pth
cd app
python app.py
```

## Dataset

Uses ~2500 balanced images from the [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/) challenge:

- **Format:** DICOM chest X-rays with RLE-encoded segmentation masks
- **Original size:** 1024x1024, resized to 512x512 for training
- **Split:** 80/20 stratified train/val (preserving positive/negative ratio)
- **Class balance:** ~50% positive (pneumothorax present), ~50% negative

## Metrics

| Metric | Description |
|--------|-------------|
| **Dice Coefficient** | Primary metric - overlap between predicted and ground truth masks |
| **IoU (Jaccard)** | Intersection over union |
| **Pixel Precision** | TP / (TP + FP) at pixel level |
| **Pixel Recall** | TP / (TP + FN) at pixel level |
| **Detection Accuracy** | Per-image: correctly identifies if pneumothorax is present |
| **Detection F1** | Per-image detection precision/recall harmonic mean |

The evaluation suite also generates:
- Dice score distribution histograms
- Threshold sensitivity analysis
- Best/worst prediction visualizations
- Separate metrics for positive vs negative images

## Augmentations

Training augmentations (via albumentations):
- Horizontal flip
- Shift/Scale/Rotate
- Elastic transform
- CLAHE / Random brightness-contrast / Random gamma
- Gaussian noise
- ImageNet normalization

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [SIIM-ACR Pneumothorax Segmentation Challenge](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/)
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
