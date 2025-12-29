"""
Enhanced Pneumothorax Detection Web Application
Professional medical AI interface with improved UX
Uses EfficientNet classifier with GradCAM visualization

Usage: streamlit run app_improved.py
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
from pydicom import dcmread
import io
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from datetime import datetime
from preprocessing import remove_text_annotations, center_crop_thorax

# Page configuration
st.set_page_config(
    page_title="Pneumothorax AI Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f5f7fa;
    }

    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 900;
        letter-spacing: -1px;
    }

    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* Result boxes */
    .result-box {
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }

    .result-box:hover {
        transform: translateY(-5px);
    }

    .positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: 3px solid #e03e52;
        animation: pulse 2s infinite;
    }

    .negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: 3px solid #0891b2;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 10px 30px rgba(240, 85, 108, 0.3); }
        50% { box-shadow: 0 15px 40px rgba(240, 85, 108, 0.5); }
    }

    .confidence-text {
        font-size: 4.5rem;
        font-weight: 900;
        margin: 1.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .result-title {
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .result-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 600;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }

    .metric-container {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Upload section */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* GradCAM section */
    .gradcam-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-top: 2rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        margin: 0.5rem;
    }

    .badge-success {
        background: #d1fae5;
        color: #065f46;
    }

    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }

    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        border-top: 2px solid #e2e8f0;
        margin-top: 4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier(model_path):
    """Load the trained classifier model (cached for performance)"""
    try:
        with st.spinner('🔄 Loading AI model...'):
            classifier = keras.models.load_model(model_path)
        return classifier
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def preprocess_image(uploaded_file, resize_to=512, remove_borders=True):
    """
    Preprocess uploaded image (DICOM or PNG/JPG) for model input.

    V2 Model Issue: The classifier was trained on images with edge artifacts
    and learned to focus on DICOM text labels instead of pathology.
    Setting remove_borders=True helps mitigate this bias.

    Args:
        uploaded_file: Streamlit uploaded file object
        resize_to: Target size for model input (default 512)
        remove_borders: Remove outer 5% to eliminate DICOM text artifacts

    Returns:
        tuple: (rgb_array for model, display_image for showing user)
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'dcm':
            dcm_data = dcmread(io.BytesIO(uploaded_file.read()))
            pixel_data = dcm_data.pixel_array
            pil_image = Image.fromarray(pixel_data)
        else:
            pil_image = Image.open(uploaded_file)
            pil_image = pil_image.convert('L')

        # Remove edge artifacts (DICOM labels like "PORTABLE", "AP")
        # This prevents the V2 model from exploiting spurious correlations
        if remove_borders:
            pil_image = remove_text_annotations(pil_image, border_crop_percent=0.05)

        # Store for display
        display_image = pil_image.copy()

        # Resize
        if pil_image.size != (resize_to, resize_to):
            pil_image = pil_image.resize((resize_to, resize_to))

        # Convert to RGB (matches training)
        rgb_image = pil_image.convert('RGB')
        rgb_array = np.array(rgb_image)
        rgb_array = np.expand_dims(rgb_array, axis=0)
        rgb_array = rgb_array.astype(np.float32) / 255.0

        return rgb_array, display_image

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None


def apply_tta(classifier, image, use_tta=False):
    """Apply Test-Time Augmentation if enabled"""
    if not use_tta:
        return classifier.predict(image, verbose=0)[0][0]

    # Generate augmented versions
    predictions = []

    # Original
    predictions.append(classifier.predict(image, verbose=0)[0][0])

    # Horizontal flip
    flipped_h = np.flip(image, axis=2)
    predictions.append(classifier.predict(flipped_h, verbose=0)[0][0])

    # Vertical flip
    flipped_v = np.flip(image, axis=1)
    predictions.append(classifier.predict(flipped_v, verbose=0)[0][0])

    # Both flips
    flipped_hv = np.flip(np.flip(image, axis=2), axis=1)
    predictions.append(classifier.predict(flipped_hv, verbose=0)[0][0])

    # Average predictions
    return np.mean(predictions)


def segment_lungs_simple(image_array):
    """
    Improved lung segmentation from chest X-ray.
    Lungs appear dark (low intensity) in X-rays due to air.
    This creates a mask focusing on the central lung regions.
    """
    # Convert to uint8 grayscale
    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        img = (image_array * 255).astype(np.uint8)
    else:
        img = image_array.copy()

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Bilateral filter to reduce noise while preserving edges
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Otsu thresholding - lungs are darker
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert so lungs are white
    binary = cv2.bitwise_not(binary)

    # Morphological operations to clean up
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Close gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # Remove small components and keep only large regions (lungs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # Create output mask
    mask = np.zeros_like(closed)

    # Keep only the 2-3 largest components (typically the two lungs)
    # Skip label 0 (background)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1]

        # Keep up to top 3 components that are reasonably large
        min_area = img.shape[0] * img.shape[1] * 0.05  # At least 5% of image
        for idx in sorted_indices[:3]:
            label_idx = idx + 1
            if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == label_idx] = 255

    # Dilate slightly to ensure lung edges are included
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    mask = cv2.dilate(mask, kernel_dilate, iterations=2)

    # Smooth the mask for better visualization
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # Normalize to 0-1
    return mask.astype(np.float32) / 255.0


def generate_gradcam(model, image, last_conv_layer_name=None, use_lung_mask=True):
    """
    Generate GradCAM heatmap showing where the model is focusing its attention.
    If use_lung_mask=True, masks the heatmap to show only lung regions.
    """
    # Auto-detect last conv layer for EfficientNet-B3
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and hasattr(layer, 'output_shape'):
                if len(layer.output_shape) == 4:
                    last_conv_layer_name = layer.name
                    break

    # Create a model that outputs both the conv layer output and final prediction
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient of the predicted class with respect to the conv layer output
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, 0]

    # Get the gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv layer output by the gradients
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()

    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Average across all channels to get the heatmap
    heatmap = np.mean(conv_outputs, axis=-1)

    # Normalize heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Resize heatmap to match input image size
    heatmap = cv2.resize(heatmap, (512, 512))

    # Apply lung mask if requested
    if use_lung_mask:
        # Get grayscale version of input
        gray = cv2.cvtColor((image[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lung_mask = segment_lungs_simple(gray)

        # Apply mask to heatmap
        heatmap = heatmap * lung_mask

        # Renormalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

    return heatmap


def create_gradcam_overlay(original_image, heatmap, alpha=0.5):
    """Create a GradCAM overlay on the original image."""
    # Convert PIL image to numpy if needed
    if isinstance(original_image, Image.Image):
        img_array = np.array(original_image.convert('RGB').resize((512, 512)))
    else:
        img_array = original_image

    # Convert heatmap to RGB colormap (jet colormap)
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Superimpose the heatmap on original image
    overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)

    return Image.fromarray(overlayed.astype(np.uint8))


def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Pneumothorax AI Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered chest X-ray analysis using deep learning • Instant results with explainable AI</p>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Configuration")

        # Model path
        model_path = st.text_input(
            "Model Path",
            value="saved_models/classifier_efficientnet_v2",
            help="Path to your trained classifier model (SavedModel format)"
        )

        st.markdown("---")

        # Detection settings
        st.markdown("### Detection Settings")

        threshold = st.slider(
            "Classification Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Lower = higher sensitivity (more detections)"
        )

        # Visual threshold indicator
        if threshold < 0.3:
            st.markdown('<span class="status-badge badge-danger">High Sensitivity</span>', unsafe_allow_html=True)
        elif threshold < 0.6:
            st.markdown('<span class="status-badge badge-warning">Balanced</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-success">High Specificity</span>', unsafe_allow_html=True)

        use_tta = st.checkbox(
            "Test-Time Augmentation",
            value=False,
            help="Improves accuracy (+5-10%) but slower (4x)"
        )

        use_lung_mask = st.checkbox(
            "Lung-Focused GradCAM",
            value=True,
            help="Masks GradCAM to show only attention within lung regions (recommended)"
        )

        use_gradcam = st.checkbox(
            "GradCAM Visualization",
            value=True,
            help="Show where AI is looking"
        )

        st.markdown("---")

        # Info
        with st.expander("About This System"):
            st.markdown("""
            **Model Architecture:**
            - EfficientNet-B3 backbone
            - Transfer learning from ImageNet
            - Fine-tuned on pneumothorax data

            **Supported Formats:**
            - DICOM (.dcm)
            - PNG (.png)
            - JPEG (.jpg, .jpeg)

            **Threshold Guide:**
            - 0.2-0.3: High sensitivity
            - 0.4-0.6: Balanced
            - 0.7-0.9: High specificity
            """)

    # Load model
    classifier = load_classifier(model_path)

    if classifier is None:
        st.error("Could not load model. Please check the path.")
        st.stop()

    st.sidebar.success("Model loaded successfully")

    # Main content
    st.markdown("---")

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown("### Upload Chest X-Ray")

        uploaded_file = st.file_uploader(
            "Drag and drop or browse",
            type=['dcm', 'png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image"
        )

        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / 1024
            file_type = uploaded_file.name.split('.')[-1].upper()

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Filename", uploaded_file.name[:20] + "...")
            with col_b:
                st.metric("Size", f"{file_size:.1f} KB")
            with col_c:
                st.metric("Type", file_type)

    with col2:
        if uploaded_file is not None:
            st.markdown("### Preview")

            # Process image
            with st.spinner("Processing..."):
                rgb_array, display_image = preprocess_image(uploaded_file)

            if rgb_array is not None and display_image is not None:
                st.image(display_image, use_column_width=True)

    # Analysis section
    if uploaded_file is not None and rgb_array is not None:
        st.markdown("---")

        # Center the analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("Analyze X-Ray", type="primary", use_container_width=True)

        if analyze_button:
            # Progress indicators
            progress_text = "🤖 AI is analyzing..." + (" (using TTA)" if use_tta else "")
            progress_bar = st.progress(0, text=progress_text)

            # Make prediction
            progress_bar.progress(30, text="🧠 Running neural network...")
            probability = apply_tta(classifier, rgb_array, use_tta)
            progress_bar.progress(70, text="📊 Calculating confidence...")

            # Generate GradCAM if enabled
            gradcam_overlay = None
            if use_gradcam:
                progress_bar.progress(85, text="🔥 Generating attention map...")
                try:
                    heatmap = generate_gradcam(classifier, rgb_array, use_lung_mask=use_lung_mask)
                    gradcam_overlay = create_gradcam_overlay(display_image, heatmap, alpha=0.5)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate GradCAM: {str(e)}")

            progress_bar.progress(100, text="✅ Analysis complete!")
            progress_bar.empty()

            # Determine result
            detected = probability >= threshold
            confidence = probability if detected else (1 - probability)

            # Display results
            st.markdown("---")

            if detected:
                st.markdown(f"""
                <div class='result-box positive'>
                    <div class='result-title'>⚠️ PNEUMOTHORAX DETECTED</div>
                    <div class='confidence-text'>{probability*100:.1f}%</div>
                    <div class='result-subtitle'>Detection Probability</div>
                    <p style='margin-top: 2rem; font-size: 1.2rem; line-height: 1.6;'>
                        This X-ray shows signs consistent with pneumothorax.<br>
                        <strong>⚕️ Please consult a medical professional immediately.</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box negative'>
                    <div class='result-title'>✅ NO PNEUMOTHORAX</div>
                    <div class='confidence-text'>{confidence*100:.1f}%</div>
                    <div class='result-subtitle'>Confidence Level</div>
                    <p style='margin-top: 2rem; font-size: 1.2rem; line-height: 1.6;'>
                        This screening suggests no pneumothorax present.<br>
                        <em>⚕️ This is not a definitive medical diagnosis.</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Metrics row
            st.markdown("### Analysis Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{probability:.3f}</div>
                    <div class='metric-label'>Raw Score</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{threshold:.2f}</div>
                    <div class='metric-label'>Threshold</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{'4x' if use_tta else '1x'}</div>
                    <div class='metric-label'>Augmentation</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{confidence*100:.0f}%</div>
                    <div class='metric-label'>Confidence</div>
                </div>
                """, unsafe_allow_html=True)

            # Probability visualization
            st.markdown("### Probability Distribution")
            fig, ax = plt.subplots(figsize=(12, 3))

            categories = ['Normal', 'Pneumothorax']
            probabilities = [1 - probability, probability]
            colors = ['#4facfe', '#f5576c']

            bars = ax.barh(categories, probabilities, color=colors, alpha=0.8, edgecolor='white', linewidth=3)

            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                ax.text(prob/2, i, f'{prob*100:.1f}%',
                       ha='center', va='center', fontsize=16, fontweight='bold', color='white')

            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability', fontsize=14, fontweight='bold')
            ax.axvline(x=threshold, color='#ff0000', linestyle='--', linewidth=3,
                      label=f'Threshold ({threshold:.2f})')
            ax.legend(fontsize=12, loc='upper right')
            ax.grid(axis='x', alpha=0.2, linestyle='--')
            ax.set_facecolor('#f5f7fa')
            fig.patch.set_facecolor('#f5f7fa')

            st.pyplot(fig)

            # GradCAM Visualization
            if gradcam_overlay is not None:
                st.markdown("---")
                st.markdown("### 🔥 AI Attention Heatmap")

                st.markdown("""
                <p style='text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>
                This visualization shows where the AI model focused when making its decision.<br>
                <strong style='color: #f5576c;'>🔴 Red/Yellow = High attention</strong> •
                <strong style='color: #4facfe;'>🔵 Blue/Green = Low attention</strong>
                </p>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original X-Ray**")
                    st.image(display_image, use_column_width=True)

                with col2:
                    st.markdown("**Attention Heatmap**")
                    st.image(gradcam_overlay, use_column_width=True)

                st.markdown("""
                <div style='background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #2196f3; margin-top: 1rem;'>
                    <h4 style='color: #1976d2; margin-top: 0;'>Interpretation Guide:</h4>
                    <ul style='color: #333; line-height: 1.8;'>
                        <li><strong style='color: #f44336;'>Red/Yellow areas</strong> = High model attention</li>
                        <li><strong style='color: #2196f3;'>Blue/Green areas</strong> = Low model attention</li>
                        <li>The heatmap shows regions the AI considered most important</li>
                        <li>For pneumothorax, expect attention on lung periphery and pleural space</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Technical details expander
            with st.expander("Technical Details"):
                st.markdown(f"""
                **Analysis Summary:**
                - Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
                - Model: EfficientNet-B3
                - Input: 512x512 RGB
                - Raw probability: `{probability:.6f}`
                - Threshold: `{threshold}`
                - Decision: `{'POSITIVE' if detected else 'NEGATIVE'}`
                - TTA: `{'Enabled (4 predictions averaged)' if use_tta else 'Disabled'}`
                - Confidence: `{confidence*100:.2f}%`

                **Classification Logic:**
                ```
                if probability > {threshold}:
                    result = "Pneumothorax Detected"
                else:
                    result = "No Pneumothorax"
                ```
                """)

            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🔄 Analyze Another", use_container_width=True):
                    st.rerun()

            with col2:
                st.button("📥 Download Report", use_container_width=True, disabled=True)
                st.caption("Coming soon")

            with col3:
                st.button("📧 Email Results", use_container_width=True, disabled=True)
                st.caption("Coming soon")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        <p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;'>
            ⚠️ Medical Disclaimer
        </p>
        <p style='font-size: 0.95rem; line-height: 1.6; max-width: 800px; margin: 0 auto;'>
            This AI system is for screening and educational purposes only. It should not replace
            professional medical diagnosis or treatment. Always consult qualified healthcare
            professionals for medical decisions.
        </p>
        <p style='font-size: 0.85rem; margin-top: 2rem; color: #94a3b8;'>
            Powered by TensorFlow & EfficientNet-B3 • Version 3.0 •
            <a href='https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation'
               style='color: #667eea; text-decoration: none;'>SIIM-ACR Dataset</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
