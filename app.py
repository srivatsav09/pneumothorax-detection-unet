"""
Pneumothorax Detection Web Application
A clean, user-friendly interface for detecting pneumothorax in chest X-ray images.

Usage: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow.keras.models as models
from pydicom import dcmread
import io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tempfile
import os

from losses import combined_dice_focal_loss, combined_dice_wpce_loss
from metrics import dice_coefficient_wrapper


# Page configuration
st.set_page_config(
    page_title="Pneumothorax Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .positive {
        background-color: #ffcccc;
        border: 3px solid #ff0000;
    }
    .negative {
        background-color: #ccffcc;
        border: 3px solid #00ff00;
    }
    .confidence-text {
        font-size: 2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(classifier_path, seg_path, loss_type='combined_dice_focal'):
    """Load the trained models (cached for performance)"""
    with st.spinner('Loading AI models... This may take a minute...'):
        # Load classifier
        classifier = models.load_model(classifier_path)

        # Load segmentation model with appropriate loss
        if loss_type == 'combined_dice_focal':
            seg_loss = combined_dice_focal_loss()
        else:
            seg_loss = combined_dice_wpce_loss(0.010753784, 1)

        dice_coefficient = dice_coefficient_wrapper()

        seg = models.load_model(
            seg_path,
            custom_objects={'compute_loss': seg_loss, 'dice_coefficient': dice_coefficient}
        )

    return classifier, seg


def preprocess_image(uploaded_file, resize_to=512):
    """
    Preprocess uploaded image (DICOM or PNG/JPG) for model input.
    Returns both RGB (for classifier) and grayscale (for segmentation) versions.
    """
    # Check file type
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == 'dcm':
        # Handle DICOM files
        dcm_data = dcmread(io.BytesIO(uploaded_file.read()))
        pixel_data = dcm_data.pixel_array
        pil_image = Image.fromarray(pixel_data)
    else:
        # Handle standard image formats (PNG, JPG, etc.)
        pil_image = Image.open(uploaded_file)
        # Convert to grayscale if not already
        pil_image = pil_image.convert('L')

    # Resize
    if pil_image.size != (resize_to, resize_to):
        pil_image = pil_image.resize((resize_to, resize_to))

    # Create RGB version for classifier
    rgb_image = pil_image.convert('RGB')
    rgb_array = np.array(rgb_image)
    rgb_array = np.expand_dims(rgb_array, axis=0)
    rgb_array = rgb_array.astype(np.float32) / 255.0

    # Create grayscale version for segmentation
    gray_image = pil_image.convert('L')
    gray_array = np.array(gray_image)
    gray_array = np.expand_dims(gray_array, axis=0)
    gray_array = np.expand_dims(gray_array, axis=-1)
    gray_array = gray_array.astype(np.float32) / 255.0

    return rgb_array, gray_array, gray_image


def apply_tta_augmentations(image):
    """Generate test-time augmentation variants"""
    augmented_images = []
    inverse_transforms = []

    # Original
    augmented_images.append(image)
    inverse_transforms.append(lambda x: x)

    # Horizontal flip
    flipped_h = np.flip(image, axis=2)
    augmented_images.append(flipped_h)
    inverse_transforms.append(lambda x: np.flip(x, axis=2))

    # Vertical flip
    flipped_v = np.flip(image, axis=1)
    augmented_images.append(flipped_v)
    inverse_transforms.append(lambda x: np.flip(x, axis=1))

    # Both flips
    flipped_hv = np.flip(np.flip(image, axis=2), axis=1)
    augmented_images.append(flipped_hv)
    inverse_transforms.append(lambda x: np.flip(np.flip(x, axis=2), axis=1))

    return augmented_images, inverse_transforms


def predict_with_tta(classifier, seg_model, rgb_image, gray_image, threshold, use_tta):
    """Run prediction with optional Test-Time Augmentation"""

    if use_tta:
        # TTA for classification
        rgb_augs, _ = apply_tta_augmentations(rgb_image)
        classifier_preds = []
        for aug_img in rgb_augs:
            pred = classifier.predict(aug_img, verbose=0)
            classifier_preds.append(pred)
        classifier_pred = np.mean(classifier_preds)
    else:
        classifier_pred = classifier.predict(rgb_image, verbose=0)[0][0]

    # Check threshold
    if classifier_pred < threshold:
        # No pneumothorax detected
        confidence = 1 - classifier_pred
        seg_mask = None
        detected = False
    else:
        # Pneumothorax detected - run segmentation
        detected = True
        confidence = classifier_pred

        if use_tta:
            # TTA for segmentation
            gray_augs, inverse_transforms = apply_tta_augmentations(gray_image)
            seg_preds = []
            for aug_img, inverse_fn in zip(gray_augs, inverse_transforms):
                pred = seg_model.predict(aug_img, verbose=0)
                pred_original = inverse_fn(pred)
                seg_preds.append(pred_original)
            seg_mask = np.mean(seg_preds, axis=0)
        else:
            seg_mask = seg_model.predict(gray_image, verbose=0)

    return detected, confidence, seg_mask


def create_visualization(original_image, seg_mask, detected, confidence):
    """Create visualization with segmentation overlay"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Prediction with overlay
    axes[1].imshow(original_image, cmap='gray')
    if detected and seg_mask is not None:
        # Create colormap for segmentation
        cmap = plt.cm.get_cmap('viridis')
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 0.7, cmap.N)
        my_cmap = ListedColormap(my_cmap)

        axes[1].imshow(seg_mask[0, :, :, 0], cmap=my_cmap, alpha=None)
        axes[1].set_title(
            f'Pneumothorax Detected\nConfidence: {confidence*100:.1f}%',
            fontsize=14,
            fontweight='bold',
            color='red'
        )
    else:
        axes[1].set_title(
            f'No Pneumothorax Detected\nConfidence: {confidence*100:.1f}%',
            fontsize=14,
            fontweight='bold',
            color='green'
        )
    axes[1].axis('off')

    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Pneumothorax Detection System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem;'>
            AI-powered detection of pneumothorax in chest X-ray images using
            <strong>EfficientNet-B3</strong> classification and <strong>Attention U-Net++</strong> segmentation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Model paths
    st.sidebar.subheader("Model Paths")
    classifier_path = st.sidebar.text_input(
        "Classifier Model Path",
        value="C:/Users/sriva/gaip-model/pneumothorax-detection-unet/saved_models/classifier_efficientnet_v2/"
    )
    seg_path = st.sidebar.text_input(
        "Segmentation Model Path",
        value="C:/Users/sriva/gaip-model/pneumothorax-detection-unet/saved_models/segmentation_attention_unet_v2/"
    )

    # Detection settings
    st.sidebar.subheader("Detection Settings")
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Lower threshold = higher sensitivity (more likely to detect). Recommended: 0.3 for safety-critical applications."
    )

    use_tta = st.sidebar.checkbox(
        "Use Test-Time Augmentation (TTA)",
        value=True,
        help="TTA improves accuracy by averaging predictions across multiple image transformations. Recommended for better results."
    )

    loss_type = st.sidebar.selectbox(
        "Loss Function Type",
        options=['combined_dice_focal', 'combined_dice_wpce'],
        index=0,
        help="Select the loss function used during training."
    )

    # Info section
    with st.sidebar.expander("ℹ️ About This System"):
        st.markdown("""
        **Version 2 Improvements:**
        - ✅ Attention U-Net++ segmentation
        - ✅ EfficientNet-B3 classifier
        - ✅ Focal Loss for better imbalance handling
        - ✅ Test-Time Augmentation
        - ✅ Optimized for high recall (safety-first)

        **Supported Formats:**
        - DICOM (.dcm)
        - PNG (.png)
        - JPEG (.jpg, .jpeg)

        **Threshold Guide:**
        - 0.2-0.3: High sensitivity (safety-critical)
        - 0.4-0.5: Balanced
        - 0.6-0.8: High specificity (reduce false positives)
        """)

    # Load models
    try:
        classifier, seg_model = load_models(classifier_path, seg_path, loss_type)
        st.sidebar.success("✅ Models loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")
        st.error("Please check the model paths in the sidebar and ensure the models are trained and saved.")
        return

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image (DICOM, PNG, or JPG)",
            type=['dcm', 'png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image for pneumothorax detection"
        )

        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # Show file info
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.info(f"📊 File size: {file_size:.2f} KB")

    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Analysis")

            # Process button
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    # Preprocess
                    try:
                        rgb_image, gray_image, display_image = preprocess_image(uploaded_file)
                        st.success("✅ Image preprocessed successfully")
                    except Exception as e:
                        st.error(f"❌ Error preprocessing image: {str(e)}")
                        return

                    # Predict
                    with st.spinner("Running AI analysis..." + (" (with TTA)" if use_tta else "")):
                        detected, confidence, seg_mask = predict_with_tta(
                            classifier, seg_model, rgb_image, gray_image, threshold, use_tta
                        )

                    # Display results
                    st.markdown("---")

                    if detected:
                        st.markdown(f"""
                        <div class='result-box positive'>
                            <h2>⚠️ PNEUMOTHORAX DETECTED</h2>
                            <p class='confidence-text'>Confidence: {confidence*100:.1f}%</p>
                            <p><em>Please consult a medical professional immediately.</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-box negative'>
                            <h2>✅ NO PNEUMOTHORAX DETECTED</h2>
                            <p class='confidence-text'>Confidence: {confidence*100:.1f}%</p>
                            <p><em>This screening suggests no pneumothorax present.</em></p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Visualization
                    st.markdown("### 📊 Detailed Visualization")
                    fig = create_visualization(display_image, seg_mask, detected, confidence)
                    st.pyplot(fig)

                    # Additional info
                    with st.expander("📋 Technical Details"):
                        st.write(f"**Classification Score:** {confidence:.4f}")
                        st.write(f"**Threshold Used:** {threshold}")
                        st.write(f"**TTA Enabled:** {use_tta}")
                        st.write(f"**Image Size:** 512x512 pixels")
                        if detected and seg_mask is not None:
                            affected_pixels = np.sum(seg_mask > 0.5)
                            total_pixels = 512 * 512
                            percentage = (affected_pixels / total_pixels) * 100
                            st.write(f"**Affected Area:** {percentage:.2f}% of lung field")

                    # Download report button
                    st.markdown("---")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📥 Download Report", use_container_width=True):
                            st.info("Report download feature coming soon!")
                    with col_b:
                        if st.button("🔄 Analyze Another Image", use_container_width=True):
                            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Disclaimer:</strong> This system is for screening purposes only and should not replace professional medical diagnosis.
        Always consult with qualified healthcare professionals for medical decisions.</p>
        <p style='font-size: 0.9rem;'>Powered by TensorFlow • EfficientNet-B3 • Attention U-Net++ • Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
