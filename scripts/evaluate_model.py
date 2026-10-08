"""
Standalone Evaluation Runner Script for OncoVision Trained Checkpoints.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model_path models/resnet50_best.pth
"""

import argparse
import sys
from pathlib import Path
import torch

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, TEST_DIR, BATCH_SIZE, DEVICE
from src.preprocessing.dataset import create_dataloaders
from src.models.resnet import build_resnet50
from src.evaluation.metrics import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ResNet50 model on test dataset.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODELS_DIR / "resnet50_best.pth"),
        help="Path to trained model checkpoint (.pth file)."
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Test batch size.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: '{model_path}'. Please train the model first using 'python src/training/train.py'.")

    # 1. Create Test DataLoader
    _, _, test_loader, _, _, test_ds = create_dataloaders(
        test_dir=TEST_DIR,
        batch_size=args.batch_size
    )

    if len(test_ds) == 0:
        raise RuntimeError(f"Test dataset at '{TEST_DIR}' is empty!")

    # 2. Re-build ResNet50 model structure and load saved weights
    print(f"[INFO] Loading ResNet50 checkpoint from: {model_path} ...")
    model = build_resnet50(pretrained=False, freeze_backbone=False)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    # 3. Execute Evaluation
    evaluate_model(model, test_loader, device=DEVICE)


if __name__ == "__main__":
    main()
