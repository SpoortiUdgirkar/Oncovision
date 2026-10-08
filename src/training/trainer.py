"""
Training and Validation Pipeline with Early Stopping, Checkpointing, and Visualization.

This module contains:
1. EarlyStopping: Helper class for early termination and best model checkpointing.
2. train_one_epoch: Single training epoch execution loop.
3. validate_one_epoch: Single validation epoch execution loop.
4. train_model: Complete training pipeline featuring 2-stage transfer learning (Feature Extraction + Fine-Tuning),
   class-weighted CrossEntropy loss, learning rate scheduling, metric tracking, and loss/accuracy curve plotting.
"""

import sys
import copy
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    CLASS_NAMES
)
from src.models.resnet import build_resnet50, unfreeze_backbone_layers


class EarlyStopping:
    """
    Early Stopping monitor that halts training when validation loss stops improving
    and automatically saves the best model state dictionary.
    """

    def __init__(self, patience=5, min_delta=1e-4, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = Path(save_path) if save_path else None
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
            if self.save_path:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.save_path)
                print(f"  [CHECKPOINT] Validation loss improved to {val_loss:.4f}. Saved best model to {self.save_path.name}")
        else:
            self.counter += 1
            print(f"  [EARLY STOP] No loss improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True


def compute_class_weights(dataset, device=DEVICE):
    """
    Computes inverse class frequency weights for CrossEntropyLoss to address class imbalance.
    Formula: weight[c] = Total_Samples / (Num_Classes * Count_c)
    """
    counts = dataset.get_class_counts()
    total_samples = len(dataset)
    num_classes = len(CLASS_NAMES)

    if total_samples == 0:
        return None

    weights = []
    for cls_name in CLASS_NAMES:
        cnt = counts.get(cls_name, 0)
        w = total_samples / (num_classes * cnt) if cnt > 0 else 1.0
        weights.append(w)

    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[INFO] Computed Class Weights for CrossEntropyLoss: {dict(zip(CLASS_NAMES, [round(x, 3) for x in weights]))}")
    return weight_tensor


def train_one_epoch(model, dataloader, criterion, optimizer, device=DEVICE):
    """
    Executes one training epoch over the dataset.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total * 100.0) if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device=DEVICE):
    """
    Executes one validation epoch over the dataset.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total * 100.0) if total > 0 else 0.0
    return epoch_loss, epoch_acc


def plot_training_curves(history, output_dir=OUTPUTS_DIR):
    """
    Generates and saves Accuracy vs Epoch and Loss vs Epoch plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    plt.title("OncoVision ResNet50 - Training vs Validation Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("CrossEntropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    loss_path = output_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "b-o", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, history["val_acc"], "r--s", label="Validation Accuracy", linewidth=2)
    plt.title("OncoVision ResNet50 - Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    acc_path = output_dir / "accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()

    print(f"[INFO] Saved training loss curve to: {loss_path}")
    print(f"[INFO] Saved training accuracy curve to: {acc_path}")


def train_model(
    model,
    train_loader,
    val_loader,
    train_dataset,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    fine_tune=True,
    fine_tune_epochs=5,
    unfreeze_layer="layer4",
    patience=5,
    device=DEVICE,
    models_dir=MODELS_DIR,
    outputs_dir=OUTPUTS_DIR
):
    """
    Main training function managing 2-stage transfer learning, learning rate scheduling,
    early stopping, and metric tracking.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "resnet50_best.pth"
    final_model_path = models_dir / "resnet50_final.pth"

    model = model.to(device)

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(train_dataset, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initial Optimizer for frozen backbone (only classifier parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    early_stopping = EarlyStopping(patience=patience, save_path=best_model_path)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print("\n" + "=" * 70)
    print(f"       STARTING RESNET50 TRAINING ON DEVICE: {device.upper()}       ")
    print("=" * 70)
    print(f"Stage 1: Feature Extraction (Backbone Frozen) for {epochs} epochs | Initial LR={lr}\n")

    start_time = time.time()

    # Stage 1: Feature Extraction
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        v_loss, v_acc = validate_one_epoch(model, val_loader, criterion, device=device)
        scheduler.step(v_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:6.2f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:6.2f}% | LR: {curr_lr:.6f}")

        early_stopping(v_loss, model)
        if early_stopping.early_stop:
            print("\n[INFO] Stage 1 Early Stopping triggered.")
            break

    # Stage 2: Optional Fine-Tuning Stage
    if fine_tune and fine_tune_epochs > 0:
        print("\n" + "-" * 70)
        print(f"Stage 2: Fine-Tuning (Unfreezing '{unfreeze_layer}') for {fine_tune_epochs} epochs")
        print("-" * 70)

        # Unfreeze specified layers
        unfreeze_backbone_layers(model, unfreeze_from=unfreeze_layer)

        # Lower learning rate for fine-tuning to prevent destroying pretrained features
        fine_tune_lr = lr * 0.1
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=fine_tune_lr,
            weight_decay=1e-2
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        for epoch in range(epochs + 1, epochs + fine_tune_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
            v_loss, v_acc = validate_one_epoch(model, val_loader, criterion, device=device)
            scheduler.step(v_loss)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)

            curr_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch:02d}/{epochs + fine_tune_epochs:02d}] (FT) | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:6.2f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:6.2f}% | LR: {curr_lr:.6f}")

            early_stopping(v_loss, model)
            if early_stopping.early_stop:
                print("\n[INFO] Stage 2 Early Stopping triggered.")
                break

    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Training completed in {total_time / 60:.2f} minutes.")

    # Save final model state
    torch.save(model.state_dict(), final_model_path)
    print(f"[CHECKPOINT] Saved final model checkpoint to: {final_model_path}")

    # Plot training curves
    plot_training_curves(history, output_dir=outputs_dir)

    print("=" * 70)
    print("                    TRAINING PIPELINE COMPLETE                          ")
    print("=" * 70 + "\n")
    return history
