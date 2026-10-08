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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

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
from src.models.resnet import unfreeze_backbone_layers
from src.models.efficientnet import unfreeze_efficientnet_layers
from src.models.densenet import unfreeze_densenet_layers
from src.training.loss import FocalLoss


class EarlyStopping:
    """
    Early Stopping monitor that halts training when validation metric stops improving
    and automatically saves the best model state dictionary.
    """

    def __init__(self, patience=5, min_delta=1e-4, save_path=None, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = Path(save_path) if save_path else None
        self.mode = mode
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_metric, model, epoch=0):
        if self.mode == "min":
            improved = val_metric < self.best_score - self.min_delta
        else:
            improved = val_metric > self.best_score + self.min_delta

        if improved:
            self.best_score = val_metric
            self.best_epoch = epoch
            self.counter = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
            if self.save_path:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.save_path)
                metric_name = "loss" if self.mode == "min" else "Macro F1"
                fmt = f"{val_metric:.4f}" if self.mode == "min" else f"{val_metric:.2f}%"
                print(f"  [CHECKPOINT] Validation {metric_name} improved to {fmt} (Epoch {epoch}). Saved best model to {self.save_path.name}")
        else:
            self.counter += 1
            metric_name = "loss" if self.mode == "min" else "Macro F1"
            print(f"  [EARLY STOP] No {metric_name} improvement for {self.counter}/{self.patience} epochs (Best at Epoch {self.best_epoch}).")
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
    Returns: tuple (epoch_loss, epoch_acc, epoch_macro_f1)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

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

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total * 100.0) if total > 0 else 0.0

    from sklearn.metrics import f1_score
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100.0

    return epoch_loss, epoch_acc, epoch_macro_f1


def plot_training_curves(history, model_name="resnet50", output_dir=OUTPUTS_DIR):
    """
    Generates and saves Accuracy vs Epoch and Loss vs Epoch plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    clean_name = model_name.lower().strip()

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    plt.title(f"OncoVision {model_name} - Training vs Validation Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("CrossEntropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    
    loss_path = output_dir / f"{clean_name}_loss_curve.png" if clean_name != "resnet50" else output_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "b-o", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, history["val_acc"], "r--s", label="Validation Accuracy", linewidth=2)
    plt.title(f"OncoVision {model_name} - Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    
    acc_path = output_dir / f"{clean_name}_accuracy_curve.png" if clean_name != "resnet50" else output_dir / "accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()

    print(f"[INFO] Saved training loss curve to: {loss_path}")
    print(f"[INFO] Saved training accuracy curve to: {acc_path}")


def unfreeze_model_layers(model, model_name, unfreeze_layer):
    """Dispatches unfreeze strategy based on model architecture."""
    key = model_name.lower().strip()
    if "efficientnet" in key:
        unfreeze_efficientnet_layers(model, unfreeze_from=unfreeze_layer)
    elif "densenet" in key:
        unfreeze_densenet_layers(model, unfreeze_from=unfreeze_layer)
    else:
        unfreeze_backbone_layers(model, unfreeze_from=unfreeze_layer)


def train_model(
    model,
    train_loader,
    val_loader,
    train_dataset,
    model_name="resnet50",
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    fine_tune=True,
    fine_tune_epochs=5,
    unfreeze_layer="layer4",
    patience=5,
    checkpoint_metric="loss",
    label_smoothing=0.0,
    criterion=None,
    scheduler_type="plateau",
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
    
    clean_name = model_name.lower().strip()
    best_model_path = models_dir / f"{clean_name}_best.pth"
    final_model_path = models_dir / f"{clean_name}_final.pth"

    model = model.to(device)

    # Compute loss criterion if not explicitly provided
    if criterion is None:
        class_weights = compute_class_weights(train_dataset, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = criterion.to(device)
        print(f"[INFO] Using Custom Loss Criterion: {criterion.__class__.__name__}")

    # Initial Optimizer for frozen backbone (only classifier parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

    # Initialize Scheduler for Stage 1
    if scheduler_type.lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        print(f"[INFO] Initialized CosineAnnealingLR Scheduler for Stage 1 (T_max={epochs}, eta_min=1e-6)")
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        print(f"[INFO] Initialized ReduceLROnPlateau Scheduler for Stage 1")

    mode = "min" if checkpoint_metric == "loss" else "max"
    early_stopping = EarlyStopping(patience=patience, save_path=best_model_path, mode=mode)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "best_epoch": 0,
        "best_val_score": 0.0
    }

    print("\n" + "=" * 70)
    print(f"       STARTING {model_name.upper()} TRAINING ON DEVICE: {device.upper()}       ")
    print("=" * 70)
    print(f"Checkpoint Selection Metric: {checkpoint_metric.upper()} (Mode: {mode.upper()})")
    print(f"Stage 1: Feature Extraction (Backbone Frozen) for {epochs} epochs | Initial LR={lr}\n")

    start_time = time.time()

    # Stage 1: Feature Extraction
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        v_loss, v_acc, v_f1 = validate_one_epoch(model, val_loader, criterion, device=device)
        
        if scheduler_type.lower() == "cosine":
            scheduler.step()
        else:
            scheduler.step(v_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_macro_f1"].append(v_f1)

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:6.2f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:6.2f}% | Val F1: {v_f1:6.2f}% | LR: {curr_lr:.6f}", flush=True)

        metric_val = v_loss if checkpoint_metric == "loss" else v_f1
        early_stopping(metric_val, model, epoch=epoch)
        if early_stopping.early_stop:
            print("\n[INFO] Stage 1 Early Stopping triggered.")
            break

    # Stage 2: Optional Fine-Tuning Stage
    if fine_tune and fine_tune_epochs > 0:
        print("\n" + "-" * 70)
        print(f"Stage 2: Fine-Tuning (Unfreezing '{unfreeze_layer}') for {fine_tune_epochs} epochs")
        print("-" * 70)

        # Unfreeze specified layers
        unfreeze_model_layers(model, model_name=model_name, unfreeze_layer=unfreeze_layer)

        # Lower learning rate for fine-tuning to prevent destroying pretrained features
        fine_tune_lr = lr * 0.1
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=fine_tune_lr,
            weight_decay=1e-2
        )
        if scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-6)
            print(f"[INFO] Initialized CosineAnnealingLR Scheduler for Stage 2 (T_max={fine_tune_epochs}, eta_min=1e-6)")
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        for epoch in range(epochs + 1, epochs + fine_tune_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
            v_loss, v_acc, v_f1 = validate_one_epoch(model, val_loader, criterion, device=device)
            
            if scheduler_type.lower() == "cosine":
                scheduler.step()
            else:
                scheduler.step(v_loss)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
            history["val_macro_f1"].append(v_f1)

            curr_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch:02d}/{epochs + fine_tune_epochs:02d}] (FT) | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:6.2f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:6.2f}% | Val F1: {v_f1:6.2f}% | LR: {curr_lr:.6f}", flush=True)

            metric_val = v_loss if checkpoint_metric == "loss" else v_f1
            early_stopping(metric_val, model, epoch=epoch)
            if early_stopping.early_stop:
                print("\n[INFO] Stage 2 Early Stopping triggered.")
                break

    history["best_epoch"] = early_stopping.best_epoch
    history["best_val_score"] = early_stopping.best_score

    total_time = time.time() - start_time
    history["training_time_minutes"] = total_time / 60.0
    print(f"\n[SUCCESS] Training completed in {total_time / 60:.2f} minutes.")

    # Save final model state
    torch.save(model.state_dict(), final_model_path)
    print(f"[CHECKPOINT] Saved final model checkpoint to: {final_model_path}")

    # Plot training curves
    plot_training_curves(history, model_name=model_name, output_dir=outputs_dir)

    print("=" * 70)
    print(f"                    {model_name.upper()} TRAINING PIPELINE COMPLETE                          ")
    print("=" * 70 + "\n")
    return history
