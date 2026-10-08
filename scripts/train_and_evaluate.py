"""
Training and Evaluation Runner Script for OncoVision Benchmarking.

Usage:
    python scripts/train_and_evaluate.py --model efficientnet_b0
    python scripts/train_and_evaluate.py --model densenet121
"""

import argparse
import sys
from pathlib import Path
import torch

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from src.models.factory import build_model, count_parameters
from src.training.trainer import train_model
from src.evaluation.metrics import evaluate_model


def run_benchmark_for_model(
    model_name="efficientnet_b0",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    fine_tune=True,
    fine_tune_epochs=5,
    unfreeze_layer=None
):
    """
    Executes training and testing evaluation for a specific model architecture.
    """
    clean_name = model_name.lower().strip()
    if unfreeze_layer is None:
        if "efficientnet" in clean_name:
            unfreeze_layer = "features.7"
        elif "densenet" in clean_name:
            unfreeze_layer = "denseblock4"
        else:
            unfreeze_layer = "layer4"

    print("=" * 70)
    print(f"       ONCOVISION MODEL BENCHMARK: {model_name.upper()}       ")
    print("=" * 70)

    # 1. Load DataLoaders
    print("[INFO] Loading PyTorch DataLoaders...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        batch_size=batch_size
    )

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty! Please verify dataset/ directory.")

    # 2. Build Model Architecture
    print(f"[INFO] Building model architecture '{model_name}' (Pretrained=True, Frozen Backbone=True)...")
    model = build_model(
        model_name=clean_name,
        num_classes=3,
        pretrained=True,
        freeze_backbone=True
    )

    tot_params, train_params = count_parameters(model)
    print(f"[INFO] Model parameters: Total = {tot_params:,} | Trainable (Head) = {train_params:,}")

    # 3. Train Model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds,
        model_name=clean_name,
        epochs=epochs,
        lr=lr,
        fine_tune=fine_tune,
        fine_tune_epochs=fine_tune_epochs,
        unfreeze_layer=unfreeze_layer,
        patience=5,
        device=DEVICE,
        models_dir=MODELS_DIR,
        outputs_dir=OUTPUTS_DIR
    )

    # 4. Load Best Checkpoint for Test Set Evaluation
    best_checkpoint_path = MODELS_DIR / f"{clean_name}_best.pth"
    print(f"[INFO] Loading best checkpoint from: {best_checkpoint_path} for test set evaluation...")

    eval_model = build_model(model_name=clean_name, num_classes=3, pretrained=False, freeze_backbone=False)
    state_dict = torch.load(best_checkpoint_path, map_location=DEVICE)
    eval_model.load_state_dict(state_dict)
    eval_model.to(DEVICE)

    # 5. Evaluate on Independent Test Set
    metrics = evaluate_model(
        model=eval_model,
        dataloader=test_loader,
        device=DEVICE,
        model_name=clean_name,
        output_dir=OUTPUTS_DIR
    )

    metrics["training_time_minutes"] = history.get("training_time_minutes", 0.0)
    metrics["total_params"] = tot_params
    metrics["checkpoint_path"] = str(best_checkpoint_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a benchmark model.")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        choices=["resnet50", "efficientnet_b0", "densenet121"],
        help="Architecture to benchmark."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)

    args = parser.parse_args()
    run_benchmark_for_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
