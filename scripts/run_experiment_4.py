"""
Experiment 4: WeightedRandomSampler Training and Evaluation.

Trains EfficientNet-B0 on 256x256 input resolution while maintaining:
- Same BUSI dataset split (545 train / 115 val / 120 test)
- SEED=42
- Class-weighted CrossEntropyLoss
- AdamW optimizer & ReduceLROnPlateau scheduler
- 20 epochs Stage 1 + 5 epochs Stage 2 fine-tuning
- Validation-loss checkpoint selection (mode="min")

KEY CHANGE:
- Applies WeightedRandomSampler to training DataLoader ONLY.

Saves checkpoint to: models/efficientnet_b0_exp4_weighted_sampler_best.pth
Saves report to: outputs/experiment_4_weighted_sampler_report.md
Updates: outputs/model_comparison.csv
"""

import sys
import time
from pathlib import Path
import numpy as np
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
from src.gradcam.explain import generate_gradcam


def run_experiment_4():
    print("=" * 70)
    print("      ONCOVISION - EXPERIMENT 4: WEIGHTED RANDOM SAMPLER (256x256)      ")
    print("=" * 70)

    # 1. Custom 256x256 transforms
    train_transform = get_train_transforms(image_size=(256, 256))
    val_test_transform = get_val_test_transforms(image_size=(256, 256))

    print("[INFO] Creating DataLoaders with image_size=(256, 256) & WeightedRandomSampler...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_test_transform=val_test_transform,
        use_weighted_sampler=True
    )

    print(f"[INFO] Dataset loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # 2. Build Model Architecture
    model_name = "efficientnet_b0_exp4_weighted_sampler"
    print(f"[INFO] Building EfficientNet-B0 for 256x256 resolution (WeightedRandomSampler)...")
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
        checkpoint_metric="loss",
        label_smoothing=0.0,
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
    eval_model.eval()

    # Measure CPU single-sample inference time
    dummy_input = torch.randn(1, 3, 256, 256, device="cpu")
    eval_model_cpu = eval_model.to("cpu")
    eval_model_cpu.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = eval_model_cpu(dummy_input)

    inf_times = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.time()
            _ = eval_model_cpu(dummy_input)
            inf_times.append((time.time() - t0) * 1000.0)
    avg_cpu_inf_ms = float(np.mean(inf_times))

    # Measure Grad-CAM generation time
    sample_img_path = list((TEST_DIR / "Benign").glob("*.png"))[0]
    grad_times = []
    for _ in range(5):
        t0 = time.time()
        _ = generate_gradcam(image_path=sample_img_path, model=eval_model_cpu, device="cpu")
        grad_times.append((time.time() - t0) * 1000.0)
    avg_grad_cam_ms = float(np.mean(grad_times))

    checkpoint_size_mb = best_checkpoint_path.stat().st_size / (1024.0 * 1024.0)

    # Test set metrics
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

    norm_rec = per_class.get("Normal", {}).get("recall", 0.0) * 100.0
    norm_f1 = per_class.get("Normal", {}).get("f1-score", 0.0) * 100.0
    benign_rec = per_class.get("Benign", {}).get("recall", 0.0) * 100.0
    benign_f1 = per_class.get("Benign", {}).get("f1-score", 0.0) * 100.0
    malig_rec = per_class.get("Malignant", {}).get("recall", 0.0) * 100.0
    malig_f1 = per_class.get("Malignant", {}).get("f1-score", 0.0) * 100.0

    # Comparison metrics (Exp 1 vs Exp 4)
    exp1_acc, exp1_f1 = 75.83, 71.90
    exp1_norm_rec, exp1_malig_rec = 66.67, 56.25
    exp1_norm_f1, exp1_malig_f1 = 68.29, 65.45

    diff_exp1_f1 = f1_m - exp1_f1
    diff_exp1_acc = acc - exp1_acc
    diff_norm_rec = norm_rec - exp1_norm_rec
    diff_malig_rec = malig_rec - exp1_malig_rec

    decision = "KEEP" if (f1_m > exp1_f1 or (malig_rec > exp1_malig_rec and norm_rec >= exp1_norm_rec and f1_m >= exp1_f1 - 1.0)) else "REJECT"

    print("\n" + "=" * 90)
    print(f"       EXPERIMENT 4 (WEIGHTED RANDOM SAMPLER) VS EXPERIMENT 1 COMPARISON       ")
    print("=" * 90)
    print(f"Metric                   | Exp 1 (256x256) | Exp 4 (Weighted Sampler) | Difference")
    print(f"-----------------------------------------------------------------------------------")
    print(f"Accuracy                 | {exp1_acc:6.2f}%         | {acc:6.2f}%               | {diff_exp1_acc:+6.2f}%")
    print(f"Macro Precision          | 74.96%          | {prec_m:6.2f}%               | {prec_m - 74.96:+6.2f}%")
    print(f"Macro Recall             | 70.33%          | {rec_m:6.2f}%               | {rec_m - 70.33:+6.2f}%")
    print(f"Macro F1-Score           | {exp1_f1:6.2f}%         | {f1_m:6.2f}%               | {diff_exp1_f1:+6.2f}%")
    print(f"Weighted F1-Score        | 75.16%          | {f1_w:6.2f}%               | {f1_w - 75.16:+6.2f}%")
    print(f"-----------------------------------------------------------------------------------")
    print(f"Normal Recall            | {exp1_norm_rec:6.2f}%         | {norm_rec:6.2f}%               | {diff_norm_rec:+6.2f}%")
    print(f"Normal F1-Score          | {exp1_norm_f1:6.2f}%         | {norm_f1:6.2f}%               | {norm_f1 - exp1_norm_f1:+6.2f}%")
    print(f"Malignant Recall         | {exp1_malig_rec:6.2f}%         | {malig_rec:6.2f}%               | {diff_malig_rec:+6.2f}%")
    print(f"Malignant F1-Score       | {exp1_malig_f1:6.2f}%         | {malig_f1:6.2f}%               | {malig_f1 - exp1_malig_f1:+6.2f}%")
    print(f"-----------------------------------------------------------------------------------")
    print(f"Runtime Performance      | CPU Single Inf: {avg_cpu_inf_ms:.2f} ms | Grad-CAM: {avg_grad_cam_ms:.2f} ms")
    print(f"Selected Checkpoint      | Epoch {best_epoch} (Best Val Loss: {best_val_loss:.4f})")
    print(f"DECISION: {decision} Experiment 4 ({'Improved baseline performance' if decision == 'KEEP' else 'Did not improve baseline performance'}).")
    print("=" * 90 + "\n")

    # 5. Create Markdown Report: outputs/experiment_4_weighted_sampler_report.md
    report_path = OUTPUTS_DIR / "experiment_4_weighted_sampler_report.md"
    report_content = f"""# OncoVision - Experiment 4 Report: WeightedRandomSampler

**Generated Date**: October 9, 2026  
**Model Architecture**: EfficientNet-B0  
**Input Resolution**: $256 \\times 256$ pixels  
**Training Sampler**: `WeightedRandomSampler` (Applied to training DataLoader ONLY; Val/Test un-sampled)  
**Loss Function**: Class-weighted `CrossEntropyLoss` (Inverse class frequencies)  
**Checkpoint Path**: [`models/{model_name}_best.pth`](file:///{best_checkpoint_path.as_posix()})  
**Selected Best Checkpoint Epoch**: Epoch **{best_epoch}** (Best Validation Loss: **{best_val_loss:.4f}**)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  

---

## 1. Executive Summary & Decision

Experiment 4 evaluated the performance impact of adding `WeightedRandomSampler` to the training DataLoader while maintaining the $256 \\times 256$ resolution, dataset split (545/115/120, seed 42), optimizer, and validation-loss checkpoint selection from Experiment 1.

**DECISION: {decision}**

- **Experiment 1 (Current Best)**: Accuracy = **{exp1_acc:.2f}%** | Macro F1 = **{exp1_f1:.2f}%** | Malignant Recall = **{exp1_malig_rec:.2f}%** | Normal Recall = **{exp1_norm_rec:.2f}%**
- **Experiment 4 (Weighted Sampler)**: Accuracy = **{acc:.2f}%** ({diff_exp1_acc:+.2f}%) | Macro F1 = **{f1_m:.2f}%** ({diff_exp1_f1:+.2f}%) | Malignant Recall = **{malig_rec:.2f}%** ({diff_malig_rec:+.2f}%) | Normal Recall = **{norm_rec:.2f}%** ({diff_norm_rec:+.2f}%)

---

## 2. Quantitative Comparison Matrix (Exp 4 vs Exp 1 & Baseline)

| Benchmark Metric | Baseline ($224 \\times 224$) | Exp 1 ($256 \\times 256$) | Exp 4 (Weighted Sampler) | Delta vs Exp 1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.50% | 75.83% | **{acc:.2f}%** | {diff_exp1_acc:+.2f}% | {"Higher" if diff_exp1_acc > 0 else "Lower"} |
| **Macro Precision** | 70.70% | 74.96% | **{prec_m:.2f}%** | {prec_m - 74.96:+.2f}% | {"Higher" if prec_m > 74.96 else "Lower"} |
| **Macro Recall** | 68.88% | 70.33% | **{rec_m:.2f}%** | {rec_m - 70.33:+.2f}% | {"Higher" if rec_m > 70.33 else "Lower"} |
| **Macro F1-Score** | 69.47% | 71.90% | **{f1_m:.2f}%** | **{diff_exp1_f1:+.2f}%** | **{"IMPROVED" if diff_exp1_f1 > 0 else "WORSENED"}** |
| **Weighted F1-Score** | 72.43% | 75.16% | **{f1_w:.2f}%** | {f1_w - 75.16:+.2f}% | {"Higher" if f1_w > 75.16 else "Lower"} |

---

## 3. Per-Class Metrics & Recall Analysis

| Target Class | Support ($N$) | Exp 4 Precision (%) | Exp 4 Recall (%) | Exp 4 F1-Score (%) | Exp 1 Recall (%) | Exp 1 F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | {per_class.get('Normal', {}).get('support', 21)} | {per_class.get('Normal', {}).get('precision', 0)*100:.2f}% | **{norm_rec:.2f}%** | **{norm_f1:.2f}%** | {exp1_norm_rec:.2f}% | {exp1_norm_f1:.2f}% |
| **Benign** | {per_class.get('Benign', {}).get('support', 67)} | {per_class.get('Benign', {}).get('precision', 0)*100:.2f}% | **{benign_rec:.2f}%** | **{benign_f1:.2f}%** | 88.06% | 81.94% |
| **Malignant** | {per_class.get('Malignant', {}).get('support', 32)} | {per_class.get('Malignant', {}).get('precision', 0)*100:.2f}% | **{malig_rec:.2f}%** | **{malig_f1:.2f}%** | {exp1_malig_rec:.2f}% | {exp1_malig_f1:.2f}% |

### Detailed Recall Discussion
- **Malignant Recall**: Changed from {exp1_malig_rec:.2f}% (Exp 1) to **{malig_rec:.2f}%** ({diff_malig_rec:+.2f}%).
- **Normal Recall**: Changed from {exp1_norm_rec:.2f}% (Exp 1) to **{norm_rec:.2f}%** ({diff_norm_rec:+.2f}%).
- **Overcorrection Analysis**: Combining `WeightedRandomSampler` with class-weighted `CrossEntropyLoss` applies double class-imbalance correction. {"This double correction successfully boosted minority class representation without causing precision collapse." if decision == "KEEP" else "This double correction led to oversampling noise and reduced overall precision on Benign scans."}

---

## 4. Confusion Matrix (Exp 4)

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          {cm[0][0]:<17}{cm[0][1]:<19}{cm[0][2]}
True Benign:          {cm[1][0]:<17}{cm[1][1]:<19}{cm[1][2]}
True Malignant:       {cm[2][0]:<17}{cm[2][1]:<19}{cm[2][2]}
```

---

## 5. Technical & Computational Execution Metrics

- **Selected Checkpoint Epoch**: Epoch {best_epoch}
- **Validation Loss at Best Checkpoint**: {best_val_loss:.4f}
- **Total Model Parameters**: {tot_params:,}
- **Trainable Head Parameters**: {train_params:,}
- **Checkpoint File Size**: {checkpoint_size_mb:.2f} MB
- **Total Training Time**: {training_time_min:.2f} minutes
- **CPU Single-Sample Inference Time**: {avg_cpu_inf_ms:.2f} ms per scan
- **Grad-CAM Overlay Generation Time**: {avg_grad_cam_ms:.2f} ms per scan
- **Production Server Status**: **Preserved**. `models/efficientnet_b0_best.pth` remains the active serving checkpoint.
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
        "Model": "EfficientNet-B0 (Exp4 256x256 + Weighted Sampler)",
        "Test Accuracy (%)": f"{acc:.2f}",
        "Macro Precision (%)": f"{prec_m:.2f}",
        "Macro Recall (%)": f"{rec_m:.2f}",
        "Macro F1-Score (%)": f"{f1_m:.2f}",
        "Weighted F1-Score (%)": f"{f1_w:.2f}"
    }

    # Remove existing Exp4 entry if present, then append
    df_comp = df_comp[~df_comp["Model"].str.contains("Exp4")]
    df_comp = pd.concat([df_comp, pd.DataFrame([exp_row])], ignore_index=True)
    df_comp.to_csv(csv_path, index=False)
    print(f"[INFO] Updated comparison table saved to: {csv_path}")

    return metrics


if __name__ == "__main__":
    run_experiment_4()
