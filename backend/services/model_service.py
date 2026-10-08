"""
Model Service Singleton for OncoVision FastAPI Backend.

Loads the primary trained EfficientNet-B0 model once into memory during server startup,
prevents retraining or re-loading weights per request, and executes fast inference.
"""

import sys
from pathlib import Path
import torch

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, CLASS_NAMES, IDX_TO_CLASS, DEVICE
from src.models.factory import build_model


class ModelService:
    """Singleton service for model loading and inference."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.device = DEVICE
            cls._instance.model_path = None
            cls._instance.model_name = "EfficientNet-B0"
        return cls._instance

    def load_model(self, model_path=None, architecture="efficientnet_b0"):
        """
        Loads trained model checkpoint into memory. Defaults to Experiment 9 (256x256) model.
        """
        if self.model is not None:
            return  # Already loaded

        if model_path is None:
            exp9_eff_path = MODELS_DIR / "efficientnet_b0_exp1_256_best.pth"
            best_eff_path = MODELS_DIR / "efficientnet_b0_best.pth"
            final_eff_path = MODELS_DIR / "efficientnet_b0_final.pth"

            if exp9_eff_path.exists():
                model_path = exp9_eff_path
                architecture = "efficientnet_b0"
            elif best_eff_path.exists():
                model_path = best_eff_path
                architecture = "efficientnet_b0"
            elif final_eff_path.exists():
                model_path = final_eff_path
                architecture = "efficientnet_b0"
            else:
                model_path = MODELS_DIR / "resnet50_best.pth"
                architecture = "resnet50"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: '{model_path}'. "
                "Please ensure model training has been run before starting the API."
            )

        is_exp9 = "exp1_256" in str(model_path).lower() or "exp9" in str(model_path).lower()
        arch_display = "EfficientNet-B0 (256x256 + TTA)" if is_exp9 else "EfficientNet-B0"
        print(f"[INFO] Loading trained {arch_display} model into memory from: {model_path} ...")
        
        model = build_model(model_name=architecture, num_classes=3, pretrained=False, freeze_backbone=False)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self.model_path = model_path
        self.model_name = arch_display
        self.is_exp9 = is_exp9
        print(f"[SUCCESS] {arch_display} model loaded successfully on device '{self.device}'.")

    def is_loaded(self):
        """Checks if model is loaded into memory."""
        return self.model is not None

    def rollback_to_baseline(self):
        """
        Rolls back the active model to the 224x224 baseline checkpoint (models/efficientnet_b0_best.pth).
        """
        baseline_path = MODELS_DIR / "efficientnet_b0_best.pth"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found at: {baseline_path}")
        
        self.model = None
        self.load_model(model_path=baseline_path, architecture="efficientnet_b0")
        print(f"[INFO] Rolled back model service to baseline checkpoint: {baseline_path}")

    def load_exp9_model(self):
        """
        Loads the Experiment 9 256x256 checkpoint (models/efficientnet_b0_exp1_256_best.pth).
        """
        exp9_path = MODELS_DIR / "efficientnet_b0_exp1_256_best.pth"
        if not exp9_path.exists():
            raise FileNotFoundError(f"Experiment 9 checkpoint not found at: {exp9_path}")
        
        self.model = None
        self.load_model(model_path=exp9_path, architecture="efficientnet_b0")
        print(f"[INFO] Loaded Experiment 9 model checkpoint: {exp9_path}")

    def predict(self, img_tensor: torch.Tensor, use_tta: bool = True):
        """
        Runs model inference on preprocessed image tensor (with optional Horizontal-Flip TTA).

        Args:
            img_tensor (torch.Tensor): Preprocessed input tensor [1, 3, 256, 256].
            use_tta (bool): If True, applies horizontal-flip test-time augmentation (TTA)
                            averaging softmax probabilities of original and flipped images.

        Returns:
            tuple: (pred_class_name, confidence_float, class_probabilities_dict, pred_class_idx)
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized or loaded.")

        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            logits_orig = self.model(img_tensor)
            probs_orig = torch.softmax(logits_orig, dim=1)

            if use_tta:
                img_tensor_flip = torch.flip(img_tensor, dims=[3])
                logits_flip = self.model(img_tensor_flip)
                probs_flip = torch.softmax(logits_flip, dim=1)
                probs = (probs_orig + probs_flip) / 2.0
            else:
                probs = probs_orig

            probs = probs.squeeze(0)

        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx].item())
        pred_class_name = IDX_TO_CLASS[pred_idx]

        probs_dict = {
            CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))
        }

        return pred_class_name, confidence, probs_dict, pred_idx


# Singleton instance export
model_service = ModelService()
