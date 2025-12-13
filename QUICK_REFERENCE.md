# 📌 Quick Reference Card - Version 2

## 🚀 One-Page Cheat Sheet

### Installation
```bash
pip install -r requirements.txt
```

### Train Models (Jupyter - Easiest)
```bash
jupyter notebook train_models_v2.ipynb
# Edit Section 2 paths → Run all cells
```

### Train Models (Python Scripts)
```bash
# Edit paths in files first!
python train_classifier.py      # ~30-60 min
python train_seg.py            # ~4-8 hours
```

### Run Web App
```bash
streamlit run app.py
# Open http://localhost:8501
```

### Make Predictions (Python)
```python
from predict import PneumothoraxPredictor

pp = PneumothoraxPredictor(
    fpath='path/to/images/',
    csv_path='labels.csv',
    classifier_path='saved_models/classifier/',
    seg_path='saved_models/segmentation/',
    classifier_threshold=0.3
)

pp.predict(use_tta=True)  # With TTA (recommended)
pp.predict(use_tta=False) # Faster
```

---

## 📊 Key Parameters

### Config.ini
```ini
batch_size = 1                          # GPU memory limited? Keep at 1
resize_to = 512                         # Don't change
train_prop = 0.8                        # 80% train, 20% val

classifier_threshold = 0.3              # ⬇️ = higher recall
classifier_backbone = EfficientNetB3    # Best choice
seg_architecture = attention_unet_plus_plus  # V2 architecture
loss_type = combined_dice_focal         # Best for pneumothorax
use_tta = True                          # Better accuracy
```

### Model Options

**Classifiers:**
- `EfficientNetB3` ✅ (recommended)
- `EfficientNetB0` (faster, less accurate)
- `EfficientNetB4` (slower, more accurate)
- `DenseNet169` (V1 default)

**Segmentation:**
- `attention_unet_plus_plus` ✅ (recommended)
- `unet_plus_plus` (V1 default)
- `unet` (baseline)

**Loss Functions:**
- `combined_dice_focal` ✅ (recommended)
- `combined_dice_wpce` (V1 default)
- `focal_tversky` (for tiny pneumothorax)

---

## 🎯 Threshold Guide

| Threshold | Sensitivity | Use Case |
|-----------|-------------|----------|
| 0.2 | Very High | Emergency screening |
| **0.3** | **High** | **Safety-critical (recommended)** ✅ |
| 0.5 | Medium | General purpose |
| 0.7 | Low | Reduce false positives |

---

## 📁 File Structure

```
├── models.py              # ✏️ Modified: Attention + EfficientNet
├── losses.py              # ✏️ Modified: Focal losses
├── predict.py             # ✏️ Modified: TTA support
├── config.ini             # ✏️ Modified: V2 params
├── train_models_v2.ipynb  # ➕ NEW: Jupyter notebook
├── app.py                 # ➕ NEW: Streamlit web app
├── requirements.txt       # ➕ NEW: Dependencies
├── README_V2.md           # ➕ NEW: Full docs
└── TRAINING_GUIDE.md      # ➕ NEW: How to train
```

---

## 🐛 Quick Troubleshooting

**Out of Memory?**
```ini
batch_size = 1
resize_to = 384
```

**Training Slow?**
```bash
pip install tensorflow-gpu  # If you have NVIDIA GPU
```

**Poor Dice Score (<0.5)?**
- Check CSV labels
- Train longer (100+ epochs)
- Try `focal_tversky` loss

**Models Not Saving?**
- Check disk space
- Use absolute paths
- Create folders manually

---

## 📊 Expected Results

### Training Time
- Classifier: 30-60 min (GPU) or 2-4 hours (CPU)
- Segmentation: 4-8 hours (GPU) or 12-24 hours (CPU)

### Performance
- Mean Dice: ~0.80-0.85
- Classifier Accuracy: ~0.94
- Recall @ threshold=0.3: ~0.92
- With TTA: +5-10% improvement

---

## 📝 CSV Format

**Classifier:**
```csv
ImageId,Class
image_001,0
image_002,1
```

**Segmentation:**
```csv
ImageId,EncodedPixels
image_001,-1
image_002,1 10 5 15 20 25
```

---

## 🎨 Web App Features

- ✅ Upload DICOM/PNG/JPG
- ✅ Adjustable threshold
- ✅ Toggle TTA
- ✅ Side-by-side visualization
- ✅ Confidence scores
- ✅ Affected area %
- ✅ Beautiful UI

---

## 🔥 Pro Tips

1. **Always use TTA for final predictions** (`use_tta=True`)
2. **Use threshold=0.3 for safety-critical applications**
3. **Train for at least 80 epochs** for segmentation
4. **Monitor val_dice_coefficient**, not training dice
5. **Use combined_dice_focal loss** for best results
6. **GPU highly recommended** (10x faster)

---

## 📞 Common Commands

### Check GPU
```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Install Jupyter
```bash
pip install jupyter notebook
```

### View Training History
```python
# In notebook after training
plt.plot(history.history['val_dice_coefficient'])
```

### Load Saved Model
```python
from tensorflow.keras import models
model = models.load_model('path/to/model/')
```

---

## ✅ Pre-Training Checklist

- [ ] CSV format correct
- [ ] All images exist
- [ ] Paths are absolute
- [ ] Dependencies installed
- [ ] Config.ini updated
- [ ] Disk space available (20GB+)
- [ ] GPU detected (optional)

---

## 🎯 Performance Benchmarks

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Val Accuracy | >0.90 | >0.93 | >0.95 |
| Val Dice | >0.70 | >0.78 | >0.85 |
| Recall | >0.85 | >0.90 | >0.95 |

---

## 🆘 Help

1. Read [README_V2.md](README_V2.md)
2. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
3. Review Jupyter notebook comments
4. Ask specific questions

---

**Save this file for quick reference! 📌**
