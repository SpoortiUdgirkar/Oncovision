"""
Model Evaluation and Confusion Matrix Generation Module for OncoVision.

Computes:
1. Classification Accuracy
2. Precision, Recall, and F1-score (Macro and Weighted averages)
3. Per-class metrics breakdown (Normal, Benign, Malignant)
4. Confusion Matrix computation and annotated visualization plot.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, OUTPUTS_DIR, DEVICE


def evaluate_model(
    model,
    dataloader,
    device=DEVICE,
    class_names=CLASS_NAMES,
    model_name="resnet50",
    output_dir=OUTPUTS_DIR
):
    """
    Evaluates a trained model on a given DataLoader (e.g. test set) and returns detailed performance metrics.

    Args:
        model (nn.Module): Trained PyTorch model.
        dataloader (DataLoader): PyTorch DataLoader for evaluation.
        device (str): Compute device ('cuda' or 'cpu').
        class_names (list): List of class strings.
        model_name (str): Model architecture name for output plot titles and file names.
        output_dir (str or Path): Output directory for saving confusion matrix plots.

    Returns:
        dict: Evaluation metrics dictionary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Compute Global Classification Metrics
    accuracy = accuracy_score(y_true, y_pred) * 100.0
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0

    # Compute Per-Class Metrics Dictionary
    per_class_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True)

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print Formatted Evaluation Report
    print("=" * 70)
    print(f"         ONCOVISION - {model_name.upper()} TEST EVALUATION REPORT         ")
    print("=" * 70)
    print(f"Overall Accuracy          : {accuracy:.2f}%")
    print(f"Macro Precision           : {precision_macro:.2f}%")
    print(f"Macro Recall              : {recall_macro:.2f}%")
    print(f"Macro F1-Score            : {f1_macro:.2f}%")
    print(f"Weighted F1-Score         : {f1_weighted:.2f}%\n")

    print("--- Detailed Per-Class Classification Report ---")
    cls_report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(cls_report_str)

    print("--- Confusion Matrix ---")
    print(cm)

    # Plot & Save Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title(f"OncoVision {model_name} - Test Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()

    clean_name = model_name.lower().strip()
    cm_path = output_dir / f"{clean_name}_confusion_matrix.png" if clean_name != "resnet50" else output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n[SUCCESS] Confusion matrix plot saved to: {cm_path}")
    print("=" * 70 + "\n")

    metrics_summary = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "per_class": per_class_dict,
        "classification_report": cls_report_str,
        "cm_path": cm_path
    }

    return metrics_summary


def evaluate_model_tta(
    model,
    dataloader,
    device=DEVICE,
    class_names=CLASS_NAMES,
    model_name="efficientnet_b0_exp1_256_tta",
    output_dir=OUTPUTS_DIR
):
    """
    Evaluates a trained model using Test-Time Augmentation (Horizontal Flip) on a given DataLoader.

    For each batch:
      1. Original image tensor -> softmax probabilities
      2. Horizontally flipped image tensor (torch.flip(images, dims=[3])) -> softmax probabilities
      3. Averaged probabilities = (probs_orig + probs_flip) / 2.0
      4. Prediction = argmax(averaged probabilities)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            flipped_images = torch.flip(images, dims=[3])

            logits_orig = model(images)
            probs_orig = torch.softmax(logits_orig, dim=1)

            logits_flip = model(flipped_images)
            probs_flip = torch.softmax(logits_flip, dim=1)

            probs_tta = (probs_orig + probs_flip) / 2.0
            preds = torch.argmax(probs_tta, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Compute Global Classification Metrics
    accuracy = accuracy_score(y_true, y_pred) * 100.0
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0

    # Compute Per-Class Metrics Dictionary
    per_class_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True)

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print Formatted Evaluation Report
    print("=" * 70)
    print(f"         ONCOVISION - {model_name.upper()} (TTA) TEST EVALUATION REPORT         ")
    print("=" * 70)
    print(f"Overall Accuracy          : {accuracy:.2f}%")
    print(f"Macro Precision           : {precision_macro:.2f}%")
    print(f"Macro Recall              : {recall_macro:.2f}%")
    print(f"Macro F1-Score            : {f1_macro:.2f}%")
    print(f"Weighted F1-Score         : {f1_weighted:.2f}%\n")

    print("--- Detailed Per-Class Classification Report ---")
    cls_report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(cls_report_str)

    print("--- Confusion Matrix ---")
    print(cm)

    # Plot & Save Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title(f"OncoVision {model_name} (TTA) - Test Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()

    clean_name = model_name.lower().strip()
    cm_path = output_dir / f"{clean_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n[SUCCESS] Confusion matrix plot saved to: {cm_path}")
    print("=" * 70 + "\n")

    metrics_summary = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "per_class": per_class_dict,
        "classification_report": cls_report_str,
        "cm_path": cm_path
    }

    return metrics_summary
