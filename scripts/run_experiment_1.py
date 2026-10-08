"""
Experiment 1: Higher Input Resolution (256x256) Training and Evaluation.

Trains EfficientNet-B0 on 256x256 input resolution while maintaining:
- Same BUSI dataset split (545 train / 115 val / 120 test)
- SEED=42
- Class-weighted CrossEntropyLoss
- AdamW optimizer & ReduceLROnPlateau scheduler
- 20 epochs Stage 1 + 5 epochs Stage 2 fine-tuning

Saves checkpoint to: models/efficientnet_b0_exp1_256_best.pth
Saves report to: outputs/experiment_1_256_report.md
Updates: outputs/model_comparison.csv
"""

import sys
import time
from pathlib import Path
import pandas as pd
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
    DEVICE,
    CLASS_NAMES
)
from src.preprocessing.transforms import get_train_transforms, get_val_test_transforms
from src.preprocessing.dataset import create_dataloaders
from src.models.factory import build_model, count_parameters
from src.training.trainer import train_model
from src.evaluation.metrics import evaluate_model


def run_experiment_1():
    print("=" * 70)
    print("      ONCOVISION - EXPERIMENT 1: HIGHER INPUT RESOLUTION (256x256)      ")
    print("=" * 70)

    # 1. Custom 256x256 transforms
    train_transform = get_train_transforms(image_size=(256, 256))
    val_test_transform = get_val_test_transforms(image_size=(256, 256))

    print("[INFO] Creating DataLoaders with image_size=(256, 256)...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_test_transform=val_test_transform
    )

    print(f"[INFO] Dataset loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # 2. Build Model Architecture
    model_name = "efficientnet_b0_exp1_256"
    print(f"[INFO] Building EfficientNet-B0 for 256x256 resolution...")
    model = build_model(
        model_name="efficientnet_b0",
        num_classes=3,
        pretrained=True,
        freeze_backbone=True
    )

    tot_params, train_params = count_parameters(model)
    print(f"[INFO] Model parameters: Total = {tot_params:,} | Trainable (Head) = {train_params:,}")

    start_time = time.time()

    # 3. Train Model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds,
        model_name=model_name,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        fine_tune=True,
        fine_tune_epochs=5,
        unfreeze_layer="features.7",
        patience=5,
        device=DEVICE,
        models_dir=MODELS_DIR,
        outputs_dir=OUTPUTS_DIR
    )

    training_time_min = (time.time() - start_time) / 60.0

    # 4. Evaluate Best Checkpoint on 256x256 Test Set
    best_checkpoint_path = MODELS_DIR / f"{model_name}_best.pth"
    print(f"[INFO] Loading best checkpoint from: {best_checkpoint_path}...")

    eval_model = build_model("efficientnet_b0", num_classes=3, pretrained=False, freeze_backbone=False)
    state_dict = torch.load(best_checkpoint_path, map_location=DEVICE)
    eval_model.load_state_dict(state_dict)
    eval_model.to(DEVICE)

    metrics = evaluate_model(
        model=eval_model,
        dataloader=test_loader,
        device=DEVICE,
        model_name=model_name,
        output_dir=OUTPUTS_DIR
    )

    # Extract metrics
    acc = metrics["accuracy"]
    prec_m = metrics["precision_macro"]
    rec_m = metrics["recall_macro"]
    f1_m = metrics["f1_macro"]
    f1_w = metrics["f1_weighted"]
    cm = metrics["confusion_matrix"]
    per_class = metrics["per_class"]

    # Baseline comparison metrics
    base_acc = 72.50
    base_prec = 70.70
    base_rec = 68.88
    base_f1 = 69.47
    base_wf1 = 72.43

    diff_acc = acc - base_acc
    diff_f1 = f1_m - base_f1

    outcome = "IMPROVED" if diff_f1 > 0 else ("WORSENED" if diff_f1 < 0 else "NO CHANGE")

    print("\n" + "=" * 70)
    print(f"       EXPERIMENT 1 (256x256) VS BASELINE (224x224) COMPARISON       ")
    print("=" * 70)
    print(f"Metric                   | Baseline (224x224) | Exp 1 (256x256) | Difference")
    print(f"-------------------------------------------------------------------------")
    print(f"Accuracy                 | {base_acc:6.2f}%            | {acc:6.2f}%         | {diff_acc:+6.2f}%")
    print(f"Macro Precision          | {base_prec:6.2f}%            | {prec_m:6.2f}%         | {prec_m - base_prec:+6.2f}%")
    print(f"Macro Recall             | {base_rec:6.2f}%            | {rec_m:6.2f}%         | {rec_m - base_rec:+6.2f}%")
    print(f"Macro F1-Score           | {base_f1:6.2f}%            | {f1_m:6.2f}%         | {diff_f1:+6.2f}%")
    print(f"Weighted F1-Score        | {base_wf1:6.2f}%            | {f1_w:6.2f}%         | {f1_w - base_wf1:+6.2f}%")
    print(f"-------------------------------------------------------------------------")
    print(f"Outcome: Experiment 1 {outcome} the baseline Macro F1-Score.")
    print("=" * 70 + "\n")

    # 5. Create Markdown Report: outputs/experiment_1_256_report.md
    report_path = OUTPUTS_DIR / "experiment_1_256_report.md"
    report_content = f"""# OncoVision - Experiment 1 Report: Higher Input Resolution (256x256)

**Generated Date**: October 8, 2026  
**Model Architecture**: EfficientNet-B0  
**Resolution**: $256 \\times 256$ pixels (Baseline: $224 \\times 224$)  
**Checkpoint Path**: [`models/{model_name}_best.pth`](file:///{best_checkpoint_path.as_posix()})  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  

---

## 1. Executive Summary

Experiment 1 evaluated the performance impact of increasing input resolution from $224 \\times 224$ to $256 \\times 256$ pixels for EfficientNet-B0 while preserving all other training conditions (dataset split, seed, loss weights, optimizer, and learning rate schedules).

**Outcome**: Experiment 1 **{outcome}** the baseline performance.
- **Baseline (224x224) Macro F1**: {base_f1:.2f}% | **Accuracy**: {base_acc:.2f}%
- **Exp 1 (256x256) Macro F1**: **{f1_m:.2f}%** ({diff_f1:+.2f}%) | **Accuracy**: **{acc:.2f}%** ({diff_acc:+.2f}%)

---

## 2. Quantitative Comparison against Baseline

| Benchmark Metric | Baseline ($224 \\times 224$) | Exp 1 ($256 \\times 256$) | Delta (Abs Diff) | Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | {base_acc:.2f}% | **{acc:.2f}%** | {diff_acc:+.2f}% | {"Higher" if diff_acc > 0 else "Lower"} |
| **Macro Precision** | {base_prec:.2f}% | **{prec_m:.2f}%** | {prec_m - base_prec:+.2f}% | {"Higher" if prec_m > base_prec else "Lower"} |
| **Macro Recall** | {base_rec:.2f}% | **{rec_m:.2f}%** | {rec_m - base_rec:+.2f}% | {"Higher" if rec_m > base_rec else "Lower"} |
| **Macro F1-Score** | {base_f1:.2f}% | **{f1_m:.2f}%** | **{diff_f1:+.2f}%** | **{"IMPROVED" if diff_f1 > 0 else "WORSENED"}** |
| **Weighted F1-Score** | {base_wf1:.2f}% | **{f1_w:.2f}%** | {f1_w - base_wf1:+.2f}% | {"Higher" if f1_w > base_wf1 else "Lower"} |

---

## 3. Per-Class Performance Breakdown

| Target Class | Support | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | {per_class.get('Normal', {}).get('support', 21)} | {per_class.get('Normal', {}).get('precision', 0)*100:.2f}% | {per_class.get('Normal', {}).get('recall', 0)*100:.2f}% | {per_class.get('Normal', {}).get('f1-score', 0)*100:.2f}% |
| **Benign** | {per_class.get('Benign', {}).get('support', 67)} | {per_class.get('Benign', {}).get('precision', 0)*100:.2f}% | {per_class.get('Benign', {}).get('recall', 0)*100:.2f}% | {per_class.get('Benign', {}).get('f1-score', 0)*100:.2f}% |
| **Malignant** | {per_class.get('Malignant', {}).get('support', 32)} | {per_class.get('Malignant', {}).get('precision', 0)*100:.2f}% | {per_class.get('Malignant', {}).get('recall', 0)*100:.2f}% | {per_class.get('Malignant', {}).get('f1-score', 0)*100:.2f}% |

---

## 4. Confusion Matrix

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          {cm[0][0]:<17}{cm[0][1]:<19}{cm[0][2]}
True Benign:          {cm[1][0]:<17}{cm[1][1]:<19}{cm[1][2]}
True Malignant:       {cm[2][0]:<17}{cm[2][1]:<19}{cm[2][2]}
```

---

## 5. Technical Execution Details

- **Total Model Parameters**: {tot_params:,}
- **Trainable Parameters**: {train_params:,}
- **Total Training Time**: {training_time_min:.2f} minutes
- **Hardware Device**: {DEVICE.upper()}
- **Primary Model Status**: **Baseline preserved**. Production backend continues to serve `models/efficientnet_b0_best.pth` until final model selection.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[INFO] Wrote experiment report to: {report_path}")

    # 6. Update outputs/model_comparison.csv
    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    if csv_path.exists():
        df_comp = pd.read_csv(csv_path)
    else:
        df_comp = pd.DataFrame(columns=["Model", "Test Accuracy (%)", "Macro Precision (%)", "Macro Recall (%)", "Macro F1-Score (%)", "Weighted F1-Score (%)"])

    exp_row = {
        "Model": "EfficientNet-B0 (Exp1 256x256)",
        "Test Accuracy (%)": f"{acc:.2f}",
        "Macro Precision (%)": f"{prec_m:.2f}",
        "Macro Recall (%)": f"{rec_m:.2f}",
        "Macro F1-Score (%)": f"{f1_m:.2f}",
        "Weighted F1-Score (%)": f"{f1_w:.2f}"
    }

    # Remove existing Exp1 entry if present, then append
    df_comp = df_comp[~df_comp["Model"].str.contains("Exp1 256x256")]
    df_comp = pd.concat([df_comp, pd.DataFrame([exp_row])], ignore_index=True)
    df_comp.to_csv(csv_path, index=False)
    print(f"[INFO] Updated comparison table saved to: {csv_path}")

    return metrics


if __name__ == "__main__":
    run_experiment_1()
