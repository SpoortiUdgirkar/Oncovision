"""
Model Service Singleton for OncoVision FastAPI Backend.

Loads the trained ResNet50 model once into memory during server startup,
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
from src.models.resnet import build_resnet50


class ModelService:
    """Singleton service for model loading and inference."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.device = DEVICE
            cls._instance.model_path = None
        return cls._instance

    def load_model(self, model_path=None):
        """
        Loads trained ResNet50 checkpoint into memory.
        """
        if self.model is not None:
            return  # Already loaded

        if model_path is None:
            best_path = MODELS_DIR / "resnet50_best.pth"
            final_path = MODELS_DIR / "resnet50_final.pth"
            model_path = best_path if best_path.exists() else final_path

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: '{model_path}'. "
                "Please ensure Phase 3 model training has been run before starting the API."
            )

        print(f"[INFO] Loading trained ResNet50 model into memory from: {model_path} ...")
        model = build_resnet50(pretrained=False, freeze_backbone=False)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self.model_path = model_path
        print(f"[SUCCESS] Model loaded successfully on device '{self.device}'.")

    def is_loaded(self):
        """Checks if model is loaded into memory."""
        return self.model is not None

    def predict(self, img_tensor: torch.Tensor):
        """
        Runs model inference on preprocessed image tensor.

        Returns:
            tuple: (pred_class_name, confidence_float, class_probabilities_dict)
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized or loaded.")

        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx].item())
        pred_class_name = IDX_TO_CLASS[pred_idx]

        probs_dict = {
            CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))
        }

        return pred_class_name, confidence, probs_dict


# Singleton instance export
model_service = ModelService()
