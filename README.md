# Pneumothorax Detection - AI-Powered Chest X-Ray Analysis

An intelligent pneumothorax detection system using EfficientNet-B3 with GradCAM explainability.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_improved.py
```

The app will be available at `http://localhost:8501`

## What This Does

Upload a chest X-ray (DICOM, PNG, or JPG) and get:
- **Binary Classification**: Pneumothorax detected or not
- **Confidence Score**: Probability of pneumothorax presence
- **GradCAM Visualization**: See where the AI is looking
- **Lung-Focused Attention**: Masks attention to lung regions only

## Features

- ✅ **EfficientNet-B3** classifier with ImageNet pretraining
- ✅ **GradCAM explainability** with lung-focused masking
- ✅ **Test-Time Augmentation** for improved accuracy
- ✅ **Border artifact removal** to mitigate edge bias
- ✅ **Professional Streamlit interface**
- ✅ Supports DICOM, PNG, JPG formats

## Model Versions

### V2 Classifier (Currently Active)
- **Path**: `saved_models/classifier_efficientnet_v2/`
- **Architecture**: EfficientNet-B3
- **Accuracy**: ~79%
- **Input**: 512x512 RGB images
- **Known Issue**: Has edge artifact bias (see warning below)

### V3 Multitask (Recommended, Not Yet Integrated)
- **Path**: `saved_models/multitask_clean/`
- **Architecture**: EfficientNet-B3 + Bounding Box Head
- **Target Accuracy**: 85%+
- **Advantages**:
  - No edge artifact bias
  - Provides localization
  - Better preprocessing

## ⚠️ Important: V2 Model Limitations

The current V2 model has a **known bias** toward image edge artifacts (DICOM text labels). The model learned to make decisions based on text like "PORTABLE", "AP" at image borders instead of actual lung pathology.

**Mitigations Applied:**
- Border removal preprocessing (removes outer 5% of image)
- Improved lung segmentation for GradCAM
- Adjustable classification threshold

**For better accuracy**, consider upgrading to the V3 multitask model in `saved_models/multitask_clean/`.

## Configuration

### Detection Settings (in sidebar)

**Classification Threshold:**
- `0.2-0.3`: High sensitivity (catch all cases, more false positives)
- `0.4-0.6`: Balanced (default: 0.5)
- `0.7-0.9`: High specificity (fewer false positives, may miss subtle cases)

**Test-Time Augmentation (TTA):**
- Disabled: Fast (~200-500ms per image)
- Enabled: Slower (~1-2s), but +5-10% accuracy improvement
- Averages predictions across 4 augmentations

**Lung-Focused GradCAM:**
- Enabled (recommended): Shows attention only within lung regions
- Disabled: Shows full image attention

## Technical Details

### Preprocessing Pipeline

```python
1. Load DICOM/image → PIL Image (no normalization)
2. Remove outer 5% border (eliminates DICOM labels)
3. Resize to 512x512
4. Convert grayscale to RGB (3 channels)
5. Normalize to 0-1 (divide by 255)
```

### Model Architecture

```
Input (512x512x3 RGB)
    ↓
EfficientNet-B3 (pretrained ImageNet)
    ↓
GlobalAveragePooling
    ↓
Dropout(0.3)
    ↓
Dense(44, relu) + Dropout(0.2)
    ↓
Dense(1, sigmoid)
    ↓
Output: Probability (0-1)
```

### GradCAM Implementation

1. Extract last convolutional layer activations
2. Compute gradients of prediction w.r.t. activations
3. Weight activations by gradients
4. Generate heatmap
5. Optionally mask to lung regions only
6. Overlay on original image

## File Structure

```
pneumothorax-detection-unet/
├── app_improved.py              # Main Streamlit application
├── models.py                     # Model architecture definitions
├── generators.py                 # Data generators
├── losses.py                     # Loss functions
├── multitask_generator.py        # Multitask data loading
├── multitask_losses.py           # Multitask loss functions
├── preprocessing.py              # V3 preprocessing utilities
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── saved_models/                 # Trained models
    ├── classifier_efficientnet_v2/       (current)
    ├── classifier_efficientnet_improved/
    ├── multitask_clean/                  (recommended)
    └── multitask_v3/
```

## Requirements

- Python 3.8+
- TensorFlow 2.15.1
- Streamlit 1.25+
- OpenCV
- PyDICOM
- See `requirements.txt` for full list

## Usage Examples

### Basic Usage

1. Launch the app: `streamlit run app_improved.py`
2. Upload a chest X-ray image
3. Adjust settings in sidebar (optional)
4. Click "Analyze X-Ray"
5. View results and GradCAM visualization

### Adjusting Sensitivity

For **screening** (catch all cases):
- Set threshold to 0.3
- Enable TTA
- Accept more false positives

For **confirmation** (reduce false alarms):
- Set threshold to 0.7
- May miss subtle cases
- Fewer false positives

## Known Issues & Solutions

### Issue: Incorrect predictions on training images
**Cause**: V2 model edge artifact bias
**Solution**: Border removal is active, but model is fundamentally limited. Use V3 for better accuracy.

### Issue: GradCAM shows attention on edges/tubes
**Cause**: Model learned to focus on non-pathological features
**Solution**: Enable "Lung-Focused GradCAM" to mask attention to lung regions only.

### Issue: Slow first load
**Cause**: Loading 127MB model file
**Solution**: Normal (~17 seconds). Subsequent loads are instant due to caching.

## Performance Tips

1. **First run**: Takes ~17 seconds to load model, then instant
2. **Disable TTA**: For faster inference (200-500ms vs 1-2s)
3. **Enable caching**: Streamlit automatically caches loaded model
4. **Batch processing**: Not currently supported, process one at a time

## Development

### Model Components

- **models.py**: Contains `create_classification_model()` for building EfficientNet classifier
- **generators.py**: `ClassifierGenerator` for loading DICOM images during training
- **losses.py**: Custom loss functions (dice, focal, combined)
- **preprocessing.py**: V3 preprocessing functions (border removal, lung windowing)

### Key Functions

```python
# Load model
from tensorflow import keras
model = keras.models.load_model('saved_models/classifier_efficientnet_v2/')

# Preprocess image
from preprocessing import remove_text_annotations
image = remove_text_annotations(pil_image, border_crop_percent=0.05)

# Generate GradCAM
heatmap = generate_gradcam(model, preprocessed_image, use_lung_mask=True)
```

## Changelog

### Latest (December 2024)
- ✅ Added border removal preprocessing to mitigate V2 edge bias
- ✅ Improved lung segmentation for GradCAM (CLAHE + connected components)
- ✅ Fixed Keras compatibility (using SavedModel format)
- ✅ Removed excessive emojis from UI
- ✅ Fixed interpretation guide visibility
- ✅ Cleaned up project (removed training code, notebooks, temp files)

## References

- **EfficientNet**: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **GradCAM**: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
- **Dataset**: [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/)

## License

See original repository license.

## Support

For issues or questions about:
- **Model performance**: Review the V2 model warning above
- **Technical errors**: Check console output for error messages
- **Feature requests**: Consider the V3 multitask model

---

**Note**: This is an AI-assisted diagnostic tool. Always consult qualified medical professionals for actual clinical decisions.