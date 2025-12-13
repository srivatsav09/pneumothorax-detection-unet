# 🚀 Quick Training Guide - Version 2

## Before You Train - Checklist

✅ **Required:**
1. DICOM images in a folder (or PNG/JPG)
2. CSV file with proper labels (see format below)
3. At least 8GB RAM (16GB+ recommended)
4. GPU with CUDA support (highly recommended for faster training)
5. ~20GB free disk space for models and checkpoints

---

## 📝 Data Preparation

### Step 1: Prepare Your CSV Files

**For Classifier (`fin.csv` or similar):**
```csv
ImageId,Class
1.2.276.0.7230010.3.1.4.8323329.1000.1517875162.100001,0
1.2.276.0.7230010.3.1.4.8323329.1000.1517875162.100002,1
1.2.276.0.7230010.3.1.4.8323329.1000.1517875162.100003,0
```
- Column 1 (`ImageId`): Image filename **without** .dcm extension
- Column 2 (`Class`): 0 = No pneumothorax, 1 = Pneumothorax

**For Segmentation (same CSV or separate):**
```csv
ImageId,EncodedPixels
1.2.276.0.7230010.3.1.4.8323329.1000.1517875162.100001,-1
1.2.276.0.7230010.3.1.4.8323329.1000.1517875162.100002,1 10 5 15 20 25 30 35
1.2.276.0.7230010.3.1.4.8323329.1000.1517875162.100003,-1
```
- Column 1 (`ImageId`): Image filename **without** .dcm extension
- Column 2 (`EncodedPixels`):
  - `-1` if no pneumothorax
  - RLE-encoded mask if pneumothorax present (format: start1 length1 start2 length2 ...)

### Step 2: Organize Your Files
```
your_project/
├── images/
│   ├── image_001.dcm
│   ├── image_002.dcm
│   └── ...
├── labels.csv
└── pneumothorax-detection-unet/  (this repo)
```

---

## 🎯 Training Methods

### **METHOD 1: Jupyter Notebook (Easiest - RECOMMENDED)**

#### Step 1: Install Jupyter
```bash
pip install jupyter notebook ipykernel
```

#### Step 2: Open the Notebook
```bash
cd pneumothorax-detection-unet
jupyter notebook train_models_v2.ipynb
```

#### Step 3: Follow the Notebook
The notebook has **9 sections** with step-by-step instructions:
1. ✅ Setup and imports
2. ✅ Configure file paths (**EDIT THIS SECTION**)
3. ✅ Set hyperparameters
4. ✅ Train classifier (click "Run" - will take 30-60 min)
5. ✅ Visualize classifier results
6. ✅ Train segmentation model (click "Run" - will take 4-8 hours)
7. ✅ Visualize segmentation results
8. ✅ View training summary
9. ✅ Test predictions

**Just run each cell in order!**

---

### **METHOD 2: Python Scripts (Advanced)**

#### Step 1: Edit Paths in Scripts

Open `train_classifier.py` and update these lines:
```python
save_path = 'path/to/save/classifier/'
image_path = 'path/to/your/images/'
csv_path = 'path/to/labels.csv'
```

Open `train_seg.py` and update:
```python
save_path = 'path/to/save/segmentation/'
image_path = 'path/to/your/images/'
csv_path = 'path/to/labels.csv'
```

#### Step 2: Train Classifier
```bash
python train_classifier.py
```
Wait ~30-60 minutes (or 2-4 hours without GPU)

#### Step 3: Train Segmentation
```bash
python train_seg.py
```
Wait ~4-8 hours (or 12-24 hours without GPU)

---

## ⏱️ Training Time Estimates

| Model | GPU (NVIDIA) | CPU Only | Epochs |
|-------|--------------|----------|--------|
| Classifier | 30-60 min | 2-4 hours | 10 |
| Segmentation | 4-8 hours | 12-24 hours | 80 |

**GPU Recommendation:** NVIDIA RTX 3060 or better with CUDA 11.2+

---

## 🎛️ Key Settings to Adjust

### In `config.ini`:

```ini
[model params]
batch_size = 1          # Increase to 2-4 if you have 16GB+ GPU RAM
resize_to = 512         # Don't change unless you know what you're doing
train_prop = 0.8        # 80% training, 20% validation (standard)

[version 2 params]
classifier_threshold = 0.3    # Lower = higher recall (0.2-0.4 recommended)
classifier_epochs = 10        # Can go up to 20 for better accuracy
seg_epochs = 80              # Don't reduce below 60

loss_type = combined_dice_focal  # Best for pneumothorax (keep this)
use_tta = True                   # Use Test-Time Augmentation (recommended)
```

### Critical Parameters:

**For High Recall (Safety-First):**
- `classifier_threshold = 0.3` or lower
- `loss_type = combined_dice_focal`
- `use_tta = True`

**For Faster Training (Development):**
- `classifier_epochs = 5`
- `seg_epochs = 40`
- `batch_size = 2` (if GPU allows)

**For Maximum Accuracy (Production):**
- `classifier_epochs = 15`
- `seg_epochs = 100`
- `use_tta = True`
- Train multiple models and ensemble

---

## 📊 Monitoring Training

### What to Look For:

**Classifier Training:**
- ✅ Validation accuracy should increase (target: >0.90)
- ✅ Training loss should decrease steadily
- ⚠️ If val accuracy plateaus early, increase learning rate
- ⚠️ If overfitting (train >> val), add more dropout

**Segmentation Training:**
- ✅ Validation Dice should increase (target: >0.75)
- ✅ Both train and val dice improving = good
- ⚠️ Dice stuck below 0.5? Check your data labels
- ⚠️ High dice but poor visual results? Try focal_tversky loss

### Training Progress Indicators:

```
Epoch 5/10
1024/1024 [==============================] - 45s 44ms/step - loss: 0.1234 - accuracy: 0.9245 - val_loss: 0.1456 - val_accuracy: 0.9123
```
- `loss`: Lower is better (training set)
- `accuracy`: Higher is better (training set)
- `val_loss`: Lower is better (validation set) ← **Watch this!**
- `val_accuracy`: Higher is better (validation set) ← **Watch this!**

For segmentation:
```
Epoch 40/80
1024/1024 [==============================] - 120s 117ms/step - loss: 0.2341 - dice_coefficient: 0.7823 - val_loss: 0.2567 - val_dice_coefficient: 0.7645
```
- `dice_coefficient`: Higher is better (0-1 range)
- `val_dice_coefficient`: Higher is better ← **Watch this!**

---

## 🎓 When Training is Done

### Your models will be saved to:
- Classifier: `saved_models/classifier_efficientnet_v2/`
- Segmentation: `saved_models/segmentation_attention_unet_v2/`

### Next Steps:

**1. Test Your Models:**
```bash
# Using Python
python predict.py

# Or using Streamlit web app
streamlit run app.py
```

**2. Evaluate Performance:**
```bash
python performance.py
```
This will show:
- Precision-recall curves
- Mean Dice score
- Optimal threshold recommendations

**3. Adjust Threshold:**
Based on performance.py results, adjust `classifier_threshold` in config.ini:
- **More false negatives?** Lower threshold (0.2-0.25)
- **Too many false positives?** Raise threshold (0.4-0.5)
- **Balanced?** Keep at 0.3

**4. Deploy Web App:**
```bash
streamlit run app.py
```
Open browser to `http://localhost:8501`

---

## 🐛 Common Problems & Solutions

### Problem: "Out of Memory" Error
**Solution:**
```ini
batch_size = 1          # In config.ini
resize_to = 384         # Reduce if still OOM
```

### Problem: Training Very Slow
**Solutions:**
- Install GPU version: `pip install tensorflow-gpu`
- Reduce image size: `resize_to = 384`
- Reduce model depth: `seg_depth = 2`

### Problem: Poor Dice Score (<0.5)
**Solutions:**
- Check CSV labels are correct
- Increase epochs: `seg_epochs = 100`
- Try different loss: `loss_type = focal_tversky`
- Check beta value matches your data

### Problem: Models Not Saving
**Solutions:**
- Check disk space (need ~5GB per model)
- Check folder permissions
- Use absolute paths, not relative
- Create save directories manually

### Problem: Classifier High Accuracy but Poor Segmentation
**Solution:**
- This is normal! They're separate models
- Train segmentation longer (100+ epochs)
- Use `combined_dice_focal` loss
- Enable attention: `use_attention = True`

---

## ✅ Training Checklist

Before you start training, verify:

- [ ] CSV file has correct format (checked with Excel/pandas)
- [ ] All ImageIds in CSV exist as .dcm files
- [ ] Save paths are writable and have space
- [ ] Config.ini paths are absolute, not relative
- [ ] GPU is detected: `nvidia-smi` (if using GPU)
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Python version 3.8-3.10 (not 3.11+)
- [ ] TensorFlow version 2.10-2.15

**THEN:**

1. Open `train_models_v2.ipynb`
2. Update file paths in Section 2
3. Run each cell in order
4. Wait for training to complete
5. Check saved models exist
6. Test with `streamlit run app.py`

---

## 📞 Need Help?

1. Check [README_V2.md](README_V2.md) for detailed documentation
2. Review Jupyter notebook comments
3. Check TensorFlow/GPU setup: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
4. Verify data format matches examples exactly

---

**Good luck with training! 🚀**

Remember: **You only need to train once.** After training, you can use the models forever for predictions!
