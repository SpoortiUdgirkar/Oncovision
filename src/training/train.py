"""
Main Entry Point for Training ResNet50 on Breast Ultrasound Classification.

Usage:
    python src/training/train.py --epochs 15 --batch_size 32 --lr 1e-4 --fine_tune
"""

import argparse
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE
)
from src.preprocessing.dataset import create_dataloaders
from src.models.resnet import build_resnet50
from src.training.trainer import train_model


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 on Breast Ultrasound Classification dataset.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of initial feature extraction epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="DataLoader batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate.")
    parser.add_argument("--fine_tune", action="store_true", default=True, help="Enable fine-tuning stage.")
    parser.add_argument("--fine_tune_epochs", type=int, default=5, help="Number of fine-tuning epochs.")
    parser.add_argument("--unfreeze_layer", type=str, default="layer4", choices=["layer4", "layer3", "all"], help="Layer block to unfreeze for fine-tuning.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")

    args = parser.parse_args()

    # 1. Load DataLoaders
    print("[INFO] Initializing PyTorch DataLoaders...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        batch_size=args.batch_size
    )

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty! Please verify dataset/ directory before training.")

    # 2. Build Model
    print(f"[INFO] Building ResNet50 model (Pretrained=True, Frozen Backbone=True)...")
    model = build_resnet50(pretrained=True, freeze_backbone=True)

    # 3. Train Model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds,
        epochs=args.epochs,
        lr=args.lr,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        unfreeze_layer=args.unfreeze_layer,
        patience=args.patience,
        device=DEVICE,
        models_dir=MODELS_DIR,
        outputs_dir=OUTPUTS_DIR
    )


if __name__ == "__main__":
    main()
