import os
import io
import tempfile
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pydicom import dcmread
import segmentation_models_pytorch as smp
import albumentations as A
import gradio as gr


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
DEVICE = torch.device("cpu")
IMAGE_SIZE = 512

_model = None
_optimal_threshold = 0.5


def get_model():
    """Lazy-load the model on first request."""
    global _model, _optimal_threshold

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please place best_model_deploy.pth in the app/ directory."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint.get("config", {})

    _model = smp.Unet(
        encoder_name=cfg.get("encoder_name", "efficientnet-b4"),
        encoder_weights=None,
        in_channels=cfg.get("in_channels", 3),
        classes=cfg.get("classes", 1),
        activation=None,
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model = _model.to(DEVICE)
    _model.eval()

    _optimal_threshold = checkpoint.get("optimal_threshold", 0.5)
    return _model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(pil_image=None, file_obj=None):
    """Load image from PIL Image or DICOM file upload. Returns grayscale uint8."""
    if file_obj is not None:
        file_bytes = file_obj if isinstance(file_obj, bytes) else open(file_obj, "rb").read()
        name = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", "upload.dcm")
        if name.lower().endswith(".dcm"):
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(file_bytes if isinstance(file_bytes, bytes) else open(name, "rb").read())
                tmp_path = tmp.name
            dcm = dcmread(tmp_path)
            os.unlink(tmp_path)
            pixel_array = dcm.pixel_array.astype(np.float32)
            if hasattr(dcm, "PhotometricInterpretation"):
                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    pixel_array = pixel_array.max() - pixel_array
            pmin, pmax = pixel_array.min(), pixel_array.max()
            if pmax > pmin:
                pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
            return pixel_array.astype(np.uint8)

    if pil_image is not None:
        return np.array(pil_image.convert("L"))

    raise ValueError("No image provided")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(input_image, threshold, file_upload=None):
    """Main prediction function for Gradio.

    Returns: (overlay_image, heatmap_image, results_markdown)
    """
    model = get_model()

    # Load image
    try:
        if file_upload is not None:
            gray = load_image(file_obj=file_upload)
        elif input_image is not None:
            gray = load_image(pil_image=Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image)
        else:
            return None, None, "Please upload an image."
    except Exception as e:
        return None, None, f"Error loading image: {e}"

    h_orig, w_orig = gray.shape[:2]

    # Preprocess
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = np.stack([resized] * 3, axis=-1)
    augmented = _transform(image=rgb)
    img_norm = augmented["image"]

    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Binary mask at selected threshold
    mask = (prob_map > threshold).astype(np.uint8)

    # Resize to original size
    prob_full = cv2.resize(prob_map, (w_orig, h_orig))
    mask_full = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Create overlay
    overlay = np.stack([gray] * 3, axis=-1).copy()
    overlay[mask_full == 1, 0] = np.minimum(overlay[mask_full == 1, 0].astype(int) + 120, 255).astype(np.uint8)
    overlay[mask_full == 1, 1] = (overlay[mask_full == 1, 1] * 0.4).astype(np.uint8)
    overlay[mask_full == 1, 2] = (overlay[mask_full == 1, 2] * 0.4).astype(np.uint8)

    # Create heatmap
    cmap = plt.cm.jet
    heatmap_rgba = cmap(prob_full)[:, :, :3]
    heatmap_img = (heatmap_rgba * 255).astype(np.uint8)

    # Results
    has_ptx = mask_full.sum() > 0
    max_prob = float(prob_map.max())
    confidence = max_prob if has_ptx else 1.0 - max_prob
    mask_area_pct = mask_full.sum() / mask_full.size * 100

    status = "DETECTED" if has_ptx else "NOT DETECTED"

    results_md = f"""## Pneumothorax: {status}

| Metric | Value |
|--------|-------|
| Confidence | {confidence * 100:.1f}% |
| Max Probability | {max_prob * 100:.1f}% |
| Mask Area | {mask_area_pct:.2f}% of image |
| Threshold | {threshold:.2f} |

---
*This is an AI-based screening tool and is NOT a medical diagnosis.
Always consult a qualified healthcare professional for clinical decisions.*
"""

    return overlay, heatmap_img, results_md


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

def create_demo():
    with gr.Blocks(
        title="Pneumothorax Segmentation",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
# Pneumothorax Segmentation
### UNet + EfficientNet-B4 | SIIM-ACR Dataset

Upload a chest X-ray (PNG/JPG or DICOM) to detect and segment pneumothorax regions.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Chest X-Ray (PNG/JPG)")
                file_input = gr.File(label="Or Upload DICOM (.dcm)", file_types=[".dcm"])
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Detection Threshold",
                )
                submit_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Row():
                    overlay_output = gr.Image(label="Segmentation Overlay")
                    heatmap_output = gr.Image(label="Probability Heatmap")
                results_output = gr.Markdown(label="Results")

        submit_btn.click(
            fn=predict,
            inputs=[image_input, threshold_slider, file_input],
            outputs=[overlay_output, heatmap_output, results_output],
        )

        gr.Markdown("""
---
**Model:** UNet with EfficientNet-B4 encoder | **Library:** segmentation_models_pytorch |
**Dataset:** SIIM-ACR Pneumothorax Segmentation | **Framework:** PyTorch
        """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
