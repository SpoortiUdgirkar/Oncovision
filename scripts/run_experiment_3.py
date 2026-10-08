"""
Experiment 3: Label Smoothing (0.1) Training and Evaluation.

Trains EfficientNet-B0 on 256x256 input resolution while maintaining:
- Same BUSI dataset split (545 train / 115 val / 120 test)
- SEED=42
- Class-weighted CrossEntropyLoss
- AdamW optimizer & ReduceLROnPlateau scheduler
- 20 epochs Stage 1 + 5 epochs Stage 2 fine-tuning
- Validation-loss checkpoint selection (mode="min")

KEY CHANGE:
- CrossEntropyLoss label_smoothing = 0.1

Saves checkpoint to: models/efficientnet_b0_exp3_label_smoothing_best.pth
Saves report to: outputs/experiment_3_label_smoothing_report.md
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


def run_experiment_3():
    print("=" * 70)
    print("      ONCOVISION - EXPERIMENT 3: LABEL SMOOTHING (0.1) (256x256)      ")
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
    model_name = "efficientnet_b0_exp3_label_smoothing"
    print(f"[INFO] Building EfficientNet-B0 for 256x256 resolution (Label Smoothing = 0.1)...")
    model = build_model(
        model_name="efficientnet_b0",
        num_classes=3,
        pretrained=True,
        freeze_backbone=True
    )

    tot_params, train_params = count_parameters(model)
    print(f"[INFO] Model parameters: Total = {tot_params:,} | Trainable (Head) = {train_params:,}")

    start_time = time.time()

    # 3. Train Model with label_smoothing=0.1 and checkpoint_metric="loss"
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
        checkpoint_metric="loss",
        label_smoothing=0.1,
        device=DEVICE,
        models_dir=MODELS_DIR,
        outputs_dir=OUTPUTS_DIR
    )

    training_time_min = (time.time() - start_time) / 60.0
    best_epoch = history.get("best_epoch", 0)
    best_val_loss = history.get("best_val_score", 0.0)

    # 4. Evaluate Best Checkpoint on 256x256 Test Set
    best_checkpoint_path = MODELS_DIR / f"{model_name}_best.pth"
    print(f"[INFO] Loading best checkpoint from: {best_checkpoint_path} (Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch})...")

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

    # Comparison metrics
    base_acc, base_f1 = 72.50, 69.47
    exp1_acc, exp1_f1 = 75.83, 71.90
    exp2_acc, exp2_f1 = 62.50, 57.61

    diff_exp1_f1 = f1_m - exp1_f1
    diff_exp1_acc = acc - exp1_acc

    decision = "KEEP" if (f1_m > exp1_f1 or (f1_m == exp1_f1 and acc > exp1_acc)) else "REJECT"

    print("\n" + "=" * 90)
    print(f"       4-MODEL BENCHMARK COMPARISON TABLE       ")
    print("=" * 90)
    print(f"Model / Configuration                    | Accuracy | Macro Prec | Macro Rec | Macro F1 | Weighted F1")
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"1. Baseline (224x224, Val Loss)           | 72.50%   | 70.70%     | 68.88%    | 69.47%   | 72.43%")
    print(f"2. Exp 1 (256x256, Val Loss)              | 75.83%   | 74.96%     | 70.33%    | 71.90%   | 75.16%")
    print(f"3. Exp 2 (256x256, Val Macro F1) [REJECTED]| 62.50%   | 60.83%     | 56.37%    | 57.61%   | 61.92%")
    print(f"4. Exp 3 (256x256 + Label Smooth 0.1)     | {acc:6.2f}%  | {prec_m:6.2f}%    | {rec_m:6.2f}%   | {f1_m:6.2f}%  | {f1_w:6.2f}%")
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Delta Exp 3 vs Exp 1 (Current Best)      | {diff_exp1_acc:+6.2f}%  | {prec_m - 74.96:+6.2f}%    | {rec_m - 70.33:+6.2f}%   | {diff_exp1_f1:+6.2f}%  | {f1_w - 75.16:+6.2f}%")
    print(f"--------------------------------------------------------------------------------------------------")
    print(f"Selected Checkpoint Epoch: {best_epoch} | Best Validation Loss: {best_val_loss:.4f}")
    print(f"DECISION: {decision} Experiment 3 ({'Improved over Exp 1' if decision == 'KEEP' else 'Did not outperform Exp 1'}).")
    print("=" * 90 + "\n")

    # 5. Create Markdown Report: outputs/experiment_3_label_smoothing_report.md
    report_path = OUTPUTS_DIR / "experiment_3_label_smoothing_report.md"
    report_content = f"""# OncoVision - Experiment 3 Report: Label Smoothing (0.1)

**Generated Date**: October 8, 2026  
**Model Architecture**: EfficientNet-B0  
**Input Resolution**: $256 \\times 256$ pixels  
**Loss Function**: `CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)`  
**Checkpoint Path**: [`models/{model_name}_best.pth`](file:///{best_checkpoint_path.as_posix()})  
**Selected Best Checkpoint Epoch**: Epoch **{best_epoch}** (Best Validation Loss: **{best_val_loss:.4f}**)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  

---

## 1. Executive Summary & Decision

Experiment 3 evaluated the impact of adding **Label Smoothing ($0.1$)** to the class-weighted CrossEntropy loss function on top of the successful $256 \\times 256$ resolution configuration (Experiment 1).

**DECISION: {decision}**

- **Experiment 1 (Current Best)**: Accuracy = **75.83%** | Macro F1 = **71.90%**
- **Experiment 3 (Label Smoothing 0.1)**: Accuracy = **{acc:.2f}%** ({diff_exp1_acc:+.2f}%) | Macro F1 = **{f1_m:.2f}%** ({diff_exp1_f1:+.2f}%)

---

## 2. 4-Way Benchmark Comparison Table

| Model Architecture / Experiment | Resolution | Selection Metric | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1-Score (%) | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | $224 \\times 224$ | Val Loss | 72.50% | 70.70% | 68.88% | 69.47% | 72.43% | Baseline |
| **Experiment 1** | $256 \\times 256$ | Val Loss | **75.83%** | **74.96%** | **70.33%** | **71.90%** | **75.16%** | **BEST MODEL** |
| **Experiment 2** | $256 \\times 256$ | Val Macro F1 | 62.50% | 60.83% | 56.37% | 57.61% | 61.92% | REJECTED |
| **Experiment 3** | $256 \\times 256$ | Val Loss (Smooth 0.1) | **{acc:.2f}%** | **{prec_m:.2f}%** | **{rec_m:.2f}%** | **{f1_m:.2f}%** | **{f1_w:.2f}%** | **{decision}** |

---

## 3. Per-Class Performance Breakdown (Exp 3)

| Target Class | Support | Precision (%) | Recall (%) | F1-Score (%) | Exp 1 F1 (%) | Baseline F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | {per_class.get('Normal', {}).get('support', 21)} | {per_class.get('Normal', {}).get('precision', 0)*100:.2f}% | {per_class.get('Normal', {}).get('recall', 0)*100:.2f}% | **{per_class.get('Normal', {}).get('f1-score', 0)*100:.2f}%** | 68.29% | 63.16% |
| **Benign** | {per_class.get('Benign', {}).get('support', 67)} | {per_class.get('Benign', {}).get('precision', 0)*100:.2f}% | {per_class.get('Benign', {}).get('recall', 0)*100:.2f}% | **{per_class.get('Benign', {}).get('f1-score', 0)*100:.2f}%** | 81.94% | 77.61% |
| **Malignant** | {per_class.get('Malignant', {}).get('support', 32)} | {per_class.get('Malignant', {}).get('precision', 0)*100:.2f}% | {per_class.get('Malignant', {}).get('recall', 0)*100:.2f}% | **{per_class.get('Malignant', {}).get('f1-score', 0)*100:.2f}%** | 65.45% | 67.65% |

---

## 4. Confusion Matrix (Exp 3)

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          {cm[0][0]:<17}{cm[0][1]:<19}{cm[0][2]}
True Benign:          {cm[1][0]:<17}{cm[1][1]:<19}{cm[1][2]}
True Malignant:       {cm[2][0]:<17}{cm[2][1]:<19}{cm[2][2]}
```

---

## 5. Technical Execution Details

- **Selected Checkpoint Epoch**: Epoch {best_epoch}
- **Validation Loss at Selected Checkpoint**: {best_val_loss:.4f}
- **Total Model Parameters**: {tot_params:,}
- **Trainable Parameters**: {train_params:,}
- **Training Time**: {training_time_min:.2f} minutes
- **Hardware Device**: {DEVICE.upper()}
- **Production Primary Model Status**: **Baseline preserved**. Production backend continues serving `models/efficientnet_b0_best.pth`.
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
        "Model": "EfficientNet-B0 (Exp3 256x256 + Label Smooth 0.1)",
        "Test Accuracy (%)": f"{acc:.2f}",
        "Macro Precision (%)": f"{prec_m:.2f}",
        "Macro Recall (%)": f"{rec_m:.2f}",
        "Macro F1-Score (%)": f"{f1_m:.2f}",
        "Weighted F1-Score (%)": f"{f1_w:.2f}"
    }

    # Remove existing Exp3 entry if present, then append
    df_comp = df_comp[~df_comp["Model"].str.contains("Exp3")]
    df_comp = pd.concat([df_comp, pd.DataFrame([exp_row])], ignore_index=True)
    df_comp.to_csv(csv_path, index=False)
    print(f"[INFO] Updated comparison table saved to: {csv_path}")

    return metrics


if __name__ == "__main__":
    run_experiment_3()
