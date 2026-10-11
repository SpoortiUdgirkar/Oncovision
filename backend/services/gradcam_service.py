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
from PIL import Image

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam.explain import GradCAM, MEDICAL_DISCLAIMER_TEXT


def generate_gradcam_base64(model, orig_pil: Image.Image, img_tensor, target_class=None, alpha=0.4):
    """
    Computes Grad-CAM overlay targeting target_class (e.g. from TTA prediction)
    and encodes side-by-side visualization into Base64 PNG string using lightweight OpenCV composition.

    Grad-CAM operates on the original un-flipped image forward pass using the target class index
    selected by TTA probability averaging.

    Args:
        model (nn.Module): Trained model instance.
        orig_pil (PIL.Image): Original uploaded image.
        img_tensor (torch.Tensor): Preprocessed input tensor [1, 3, H, W].
        target_class (int, optional): Target class index to generate activation map for.
        alpha (float): Heatmap opacity weight.

    Returns:
        tuple: (base64_png_uri, pred_class_idx, confidence, probabilities_list)
    """
    # Convert PIL Image to RGB NumPy array
    if orig_pil.mode != "RGB":
        orig_pil = orig_pil.convert("RGB")
    orig_np = np.array(orig_pil)
    orig_h, orig_w = orig_np.shape[:2]

    # Compute Grad-CAM Map
    gradcam_engine = GradCAM(model)
    cam_map_2d, pred_idx, confidence, probs = gradcam_engine.generate_map(img_tensor, target_class=target_class)

    # Resize Grad-CAM Map to Original Image Dimensions
    cam_resized = cv2.resize(cam_map_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Color Heatmap using JET Colormap
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha Blended Overlay
    overlay_rgb = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_rgb, alpha, 0)

    # Scale visualization height for side-by-side canvas while preserving aspect ratio
    target_h = min(orig_h, 480)
    target_w = max(1, int(orig_w * (target_h / float(orig_h))))

    if orig_h != target_h or orig_w != target_w:
        interp = cv2.INTER_AREA if orig_h > target_h else cv2.INTER_LINEAR
        orig_disp = cv2.resize(orig_np, (target_w, target_h), interpolation=interp)
        overlay_disp = cv2.resize(overlay_rgb, (target_w, target_h), interpolation=interp)
    else:
        orig_disp = orig_np
        overlay_disp = overlay_rgb

    header_h = 44
    label_h = 28
    gap = 8
    total_w = target_w * 2 + gap
    total_h = header_h + label_h + target_h + 10

    # Build clean canvas with header and side-by-side panels
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    # Top header bar (#f8fafc) and subtle border line (#e2e8f0)
    canvas[0:header_h, :] = [248, 250, 252]
    canvas[header_h - 1:header_h, :] = [226, 232, 240]

    # Main title
    cv2.putText(
        canvas,
        "OncoVision AI Model Explainability Map",
        (16, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (15, 23, 42),
        2,
        cv2.LINE_AA
    )

    # Panel subtitles
    cv2.putText(
        canvas,
        "Original Ultrasound Scan",
        (8, header_h + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (71, 85, 105),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Grad-CAM Attention Overlay",
        (target_w + gap + 8, header_h + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (71, 85, 105),
        1,
        cv2.LINE_AA
    )

    # Place image panels
    canvas[header_h + label_h : header_h + label_h + target_h, 0 : target_w] = orig_disp
    canvas[header_h + label_h : header_h + label_h + target_h, target_w + gap : target_w + gap + target_w] = overlay_disp

    # Fast PNG encoding via OpenCV
    success, encoded_buf = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode Grad-CAM visualization to PNG.")

    base64_encoded = base64.b64encode(encoded_buf.tobytes()).decode("utf-8")
    base64_data_uri = f"data:image/png;base64,{base64_encoded}"

    return base64_data_uri, pred_idx, confidence, probs
