"""
Grad-CAM Service for OncoVision FastAPI Backend.

Generates activation maps, overlays heatmap onto original scan,
and encodes the resulting figure into a frontend-compatible Base64 PNG data URI string.
"""

import sys
import io
import base64
from pathlib import Path
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")  # Non-GUI headless backend for server thread safety
import matplotlib.pyplot as plt
from PIL import Image


# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam.explain import GradCAM, MEDICAL_DISCLAIMER_TEXT


def generate_gradcam_base64(model, orig_pil: Image.Image, img_tensor, alpha=0.4):
    """
    Computes Grad-CAM overlay and encodes side-by-side visualization into Base64 PNG string.

    Args:
        model (nn.Module): Trained ResNet50 model.
        orig_pil (PIL.Image): Original uploaded image.
        img_tensor (torch.Tensor): Preprocessed input tensor [1, 3, 224, 224].
        alpha (float): Heatmap opacity weight.

    Returns:
        tuple: (base64_png_uri, pred_class_idx, confidence, probabilities_list)
    """
    orig_np = np.array(orig_pil)
    orig_h, orig_w = orig_np.shape[:2]

    # Compute Grad-CAM Map
    gradcam_engine = GradCAM(model)
    cam_map_2d, pred_idx, confidence, probs = gradcam_engine.generate_map(img_tensor)

    # Resize Grad-CAM Map to Original Image Dimensions
    cam_resized = cv2.resize(cam_map_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Color Heatmap using JET Colormap
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha Blended Overlay
    overlay_rgb = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_rgb, alpha, 0)

    # Plot Side-by-Side Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        "OncoVision AI Model Explainability Map",
        fontsize=13,
        fontweight="bold",
        color="darkblue"
    )

    axes[0].imshow(orig_np)
    axes[0].set_title("Original Ultrasound Scan", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Grad-CAM Attention Overlay", fontsize=10)
    axes[1].axis("off")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save to Bytes Buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)

    # Encode to Base64 String
    base64_encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    base64_data_uri = f"data:image/png;base64,{base64_encoded}"

    return base64_data_uri, pred_idx, confidence, probs
