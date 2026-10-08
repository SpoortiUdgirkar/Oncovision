"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Explainability Module.

This module provides:
1. GradCAM: PyTorch hook-based class for capturing feature maps and gradients from CNN backbones.
2. generate_gradcam: Reusable function to compute Grad-CAM heatmaps, create alpha-blended overlays,
   and output visual explanation figures with confidence metrics and medical disclaimers.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    CLASS_NAMES,
    IDX_TO_CLASS,
    DEVICE
)
from src.preprocessing.transforms import get_val_test_transforms
from src.models.resnet import build_resnet50


class GradCAM:
    """
    Grad-CAM class that registers forward and backward hooks on the target convolutional layer
    to extract activation maps and backpropagated gradients for a given class prediction.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        # Automatically locate final convolutional layer for ResNet50 if not specified
        if self.target_layer is None:
            if hasattr(self.model, "layer4"):
                self.target_layer = self.model.layer4[-1]
            else:
                raise ValueError("Target layer could not be automatically identified for model architecture.")

        self.activations = None
        self.gradients = None

        # Register PyTorch forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_map(self, input_tensor, target_class=None):
        """
        Computes normalized 2D Grad-CAM activation map for target class.

        Args:
            input_tensor (torch.Tensor): Preprocessed input image tensor [1, 3, 224, 224].
            target_class (int, optional): Class index to explain. Defaults to predicted class.

        Returns:
            tuple: (cam_map_2d, pred_class_idx, confidence, probabilities_list)
        """
        input_tensor = input_tensor.to(NEXT_DEVICE(self.model))
        input_tensor.requires_grad = True

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        pred_class_idx = torch.argmax(probs).item()
        confidence = probs[pred_class_idx].item()

        if target_class is None:
            target_class = pred_class_idx

        # Zero existing gradients
        self.model.zero_grad()

        # Backward pass for target class score
        score = logits[0, target_class]
        score.backward()

        # Fetch captured activations and gradients
        # Activations: [1, C, H, W], Gradients: [1, C, H, W]
        activations = self.activations[0]  # [C, H, W]
        gradients = self.gradients[0]      # [C, H, W]

        # Global Average Pooling of gradients across spatial dimensions
        weights = torch.mean(gradients, dim=(1, 2))  # [C]

        # Compute weighted sum of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU activation to keep features with positive influence
        cam = torch.clamp(cam, min=0)

        # Normalize activation map to [0.0, 1.0]
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, pred_class_idx, confidence, probs.cpu().tolist()


def NEXT_DEVICE(model):
    """Helper to retrieve device of model parameters."""
    return next(model.parameters()).device


MEDICAL_DISCLAIMER_TEXT = (
    "Model Explanation Notice: This Grad-CAM activation map highlights regions "
    "that influenced the deep learning model's prediction. It serves as an AI attention "
    "visualization tool and is NOT medically definitive or diagnostic proof."
)


def generate_gradcam(
    image_path,
    model_path=None,
    model=None,
    target_class=None,
    output_path=None,
    alpha=0.4,
    device=DEVICE
):
    """
    Reusable Grad-CAM explainability pipeline function.

    Args:
        image_path (str or Path): Path to ultrasound input image.
        model_path (str or Path, optional): Path to trained model checkpoint (.pth).
        model (nn.Module, optional): Pre-loaded PyTorch ResNet50 model instance.
        target_class (int, optional): Class index to explain (0: Normal, 1: Benign, 2: Malignant).
        output_path (str or Path, optional): Output filepath to save side-by-side visualization.
        alpha (float): Heatmap opacity blending weight (0.0 to 1.0).
        device (str): Compute device ('cuda' or 'cpu').

    Returns:
        dict: {
            "predicted_class": str,
            "predicted_class_idx": int,
            "confidence": float,
            "probabilities": list,
            "heatmap": np.ndarray (RGB),
            "overlay": np.ndarray (RGB),
            "output_path": Path
        }
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found at: '{image_path}'")

    # 1. Load Model if not provided
    if model is None:
        if model_path is None:
            model_path = MODELS_DIR / "resnet50_best.pth"
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: '{model_path}'")

        model = build_resnet50(pretrained=False, freeze_backbone=False)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # 2. Load and Preprocess Image
    orig_pil = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_pil)
    orig_h, orig_w = orig_np.shape[:2]

    transform = get_val_test_transforms()
    img_tensor = transform(orig_pil).unsqueeze(0).to(device)

    # 3. Compute Grad-CAM Map
    gradcam_engine = GradCAM(model)
    cam_map_2d, pred_idx, confidence, probs = gradcam_engine.generate_map(img_tensor, target_class=target_class)

    pred_class_name = IDX_TO_CLASS[pred_idx]

    # 4. Resize Grad-CAM Map to Original Image Dimensions
    cam_resized = cv2.resize(cam_map_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 5. Generate Color Heatmap using OpenCV JET Colormap
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 6. Create Alpha-Blended Overlay
    overlay_rgb = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_rgb, alpha, 0)

    # 7. Generate Side-by-Side Visualization Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"OncoVision Explainability - Prediction: {pred_class_name.upper()} ({confidence*100:.1f}% Confidence)",
        fontsize=14,
        fontweight="bold",
        color="darkblue"
    )

    # Panel 1: Original Image
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original Ultrasound Scan\nPath: {image_path.name}", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Grad-CAM Overlay
    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Grad-CAM Attention Map Overlay\nClass: {pred_class_name} ({confidence*100:.1f}%)", fontsize=11)
    axes[1].axis("off")

    # Footer Medical Disclaimer Notice
    plt.figtext(
        0.5, 0.02,
        MEDICAL_DISCLAIMER_TEXT,
        wrap=True,
        horizontalalignment="center",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8, edgecolor="orange")
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if output_path is None:
        output_dir = Path(OUTPUTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"gradcam_{image_path.stem}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print(f"        ONCOVISION GRAD-CAM EXPLAINABILITY GENERATED        ")
    print("=" * 70)
    print(f"Input Image        : {image_path.name}")
    print(f"Predicted Class    : {pred_class_name} (Class Index {pred_idx})")
    print(f"Confidence         : {confidence * 100:.2f}%")
    print(f"Class Probabilities: Normal: {probs[0]*100:.1f}%, Benign: {probs[1]*100:.1f}%, Malignant: {probs[2]*100:.1f}%")
    print(f"Output Saved To    : {output_path}")
    print(f"\n[NOTICE] {MEDICAL_DISCLAIMER_TEXT}")
    print("=" * 70 + "\n")

    return {
        "predicted_class": pred_class_name,
        "predicted_class_idx": pred_idx,
        "confidence": confidence,
        "probabilities": probs,
        "heatmap": heatmap_rgb,
        "overlay": overlay_rgb,
        "output_path": output_path
    }
