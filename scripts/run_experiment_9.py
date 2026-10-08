"""
Experiment 9: Horizontal-Flip Test-Time Augmentation (TTA) Evaluation.

Evaluates Test-Time Augmentation (TTA) on the trained Experiment 1 checkpoint:
- Checkpoint: models/efficientnet_b0_exp1_256_best.pth
- Resolution: 256x256
- Test Dataset: BUSI Held-Out Test Set (N=120)

NO RETRAINING IS PERFORMED.

TTA Procedure per sample:
1. Apply deterministic validation/test preprocessing (256x256, ToTensor, Normalize)
2. Model forward pass on original image -> obtain class probabilities p_orig = softmax(logits)
3. Model forward pass on horizontally flipped image (torch.flip(img, dims=[3])) -> obtain class probabilities p_flip = softmax(logits_flip)
4. Average probabilities: p_tta = (p_orig + p_flip) / 2.0
5. Prediction: class with highest averaged probability

Saves report to: outputs/experiment_9_tta_report.md
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
    DEVICE,
    CLASS_NAMES
)
from src.preprocessing.transforms import get_val_test_transforms
from src.preprocessing.dataset import create_dataloaders
from src.models.factory import build_model, count_parameters
from src.evaluation.metrics import evaluate_model, evaluate_model_tta


def run_experiment_9():
    print("=" * 70)
    print("      ONCOVISION - EXPERIMENT 9: HORIZONTAL-FLIP TTA EVALUATION      ")
    print("=" * 70)

    # 1. Load Test DataLoader (256x256)
    val_test_transform = get_val_test_transforms(image_size=(256, 256))
    _, _, test_loader, _, _, test_ds = create_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        train_transform=val_test_transform,
        val_test_transform=val_test_transform,
        use_weighted_sampler=False
    )

    print(f"[INFO] Loaded BUSI Test Dataset ($N={len(test_ds)}$ scans) at 256x256 resolution.")

    # 2. Load Trained Experiment 1 Checkpoint
    model_name = "efficientnet_b0_exp1_256"
    checkpoint_path = MODELS_DIR / f"{model_name}_best.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Experiment 1 checkpoint not found at: {checkpoint_path}")

    print(f"[INFO] Loading trained Exp 1 checkpoint from: {checkpoint_path}...")
    model = build_model("efficientnet_b0", num_classes=3, pretrained=False, freeze_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    tot_params, _ = count_parameters(model)
    print(f"[INFO] Model loaded successfully: Total parameters = {tot_params:,}")

    # 3. Evaluate Baseline (Single Pass - No TTA)
    print("\n[INFO] Evaluating Experiment 1 Baseline (Single Pass, No TTA)...")
    baseline_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=DEVICE,
        model_name="exp1_baseline_single_pass",
        output_dir=OUTPUTS_DIR
    )

    # 4. Evaluate Experiment 1 + Horizontal-Flip TTA
    print("\n[INFO] Evaluating Experiment 1 with Horizontal-Flip TTA...")
    tta_metrics = evaluate_model_tta(
        model=model,
        dataloader=test_loader,
        device=DEVICE,
        model_name="exp9_exp1_hflip_tta",
        output_dir=OUTPUTS_DIR
    )

    # 5. Measure CPU Single-Sample Latency Comparison
    dummy_input = torch.randn(1, 3, 256, 256, device="cpu")
    model_cpu = model.to("cpu")
    model_cpu.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model_cpu(dummy_input)
            _ = model_cpu(torch.flip(dummy_input, dims=[3]))

    # Measure Single Pass Latency
    single_pass_times = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.time()
            _ = model_cpu(dummy_input)
            single_pass_times.append((time.time() - t0) * 1000.0)
    avg_single_ms = float(np.mean(single_pass_times))

    # Measure TTA Latency (Original + Flip)
    tta_times = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.time()
            p1 = torch.softmax(model_cpu(dummy_input), dim=1)
            p2 = torch.softmax(model_cpu(torch.flip(dummy_input, dims=[3])), dim=1)
            _ = (p1 + p2) / 2.0
            tta_times.append((time.time() - t0) * 1000.0)
    avg_tta_ms = float(np.mean(tta_times))

    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024.0 * 1024.0)

    # Extract comparison metrics
    b_acc, b_prec, b_rec, b_f1, b_wf1 = baseline_metrics["accuracy"], baseline_metrics["precision_macro"], baseline_metrics["recall_macro"], baseline_metrics["f1_macro"], baseline_metrics["f1_weighted"]
    t_acc, t_prec, t_rec, t_f1, t_wf1 = tta_metrics["accuracy"], tta_metrics["precision_macro"], tta_metrics["recall_macro"], tta_metrics["f1_macro"], tta_metrics["f1_weighted"]

    b_cm = baseline_metrics["confusion_matrix"]
    t_cm = tta_metrics["confusion_matrix"]

    b_mal_r = baseline_metrics["per_class"].get('Malignant', {}).get('recall', 0) * 100.0
    t_mal_r = tta_metrics["per_class"].get('Malignant', {}).get('recall', 0) * 100.0

    b_norm_r = baseline_metrics["per_class"].get('Normal', {}).get('recall', 0) * 100.0
    t_norm_r = tta_metrics["per_class"].get('Normal', {}).get('recall', 0) * 100.0

    b_ben_r = baseline_metrics["per_class"].get('Benign', {}).get('recall', 0) * 100.0
    t_ben_r = tta_metrics["per_class"].get('Benign', {}).get('recall', 0) * 100.0

    b_mal_fn = b_cm[2][0] + b_cm[2][1]
    t_mal_fn = t_cm[2][0] + t_cm[2][1]

    diff_acc = t_acc - b_acc
    diff_f1 = t_f1 - b_f1

    # Render Decision
    if t_f1 > b_f1 or (t_acc >= b_acc and t_mal_r > b_mal_r):
        decision = "KEEP"
        decision_reason = f"Horizontal-Flip TTA improved or maintained Macro F1 ({t_f1:.2f}% vs {b_f1:.2f}%) and Accuracy ({t_acc:.2f}% vs {b_acc:.2f}%)."
    else:
        decision = "REJECT"
        decision_reason = f"Horizontal-Flip TTA did not improve Macro F1 ({t_f1:.2f}% vs {b_f1:.2f}%) or Accuracy ({t_acc:.2f}% vs {b_acc:.2f}%), despite doubling inference latency."

    print("\n" + "=" * 70)
    print(f"       EXPERIMENT 9: EXP 1 BASELINE (NO TTA) VS EXP 1 + H-FLIP TTA       ")
    print("=" * 70)
    print(f"Metric                   | Exp 1 Baseline (No TTA) | Exp 1 + H-Flip TTA | Difference")
    print(f"--------------------------------------------------------------------------------------")
    print(f"Accuracy                 | {b_acc:6.2f}%                 | {t_acc:6.2f}%            | {diff_acc:+6.2f}%")
    print(f"Macro Precision          | {b_prec:6.2f}%                 | {t_prec:6.2f}%            | {t_prec - b_prec:+6.2f}%")
    print(f"Macro Recall             | {b_rec:6.2f}%                 | {t_rec:6.2f}%            | {t_rec - b_rec:+6.2f}%")
    print(f"Macro F1-Score           | {b_f1:6.2f}%                 | {t_f1:6.2f}%            | {diff_f1:+6.2f}%")
    print(f"Weighted F1-Score        | {b_wf1:6.2f}%                 | {t_wf1:6.2f}%            | {t_wf1 - b_wf1:+6.2f}%")
    print(f"Normal Recall            | {b_norm_r:6.2f}%                 | {t_norm_r:6.2f}%            | {t_norm_r - b_norm_r:+6.2f}%")
    print(f"Malignant Recall         | {b_mal_r:6.2f}%                 | {t_mal_r:6.2f}%            | {t_mal_r - b_mal_r:+6.2f}%")
    print(f"Malignant False Negatives| {b_mal_fn:<23}| {t_mal_fn:<19}| {t_mal_fn - b_mal_fn:+d}")
    print(f"CPU Single Pass Latency  | {avg_single_ms:6.2f} ms              | {avg_tta_ms:6.2f} ms         | {avg_tta_ms - avg_single_ms:+6.2f} ms")
    print(f"--------------------------------------------------------------------------------------")
    print(f"DECISION: {decision}")
    print(f"Reason: {decision_reason}")
    print("=" * 70 + "\n")

    # 6. Create Markdown Report: outputs/experiment_9_tta_report.md
    report_path = OUTPUTS_DIR / "experiment_9_tta_report.md"
    report_content = f"""# OncoVision - Experiment 9 Report: Horizontal-Flip Test-Time Augmentation (TTA)

**Generated Date**: October 9, 2026  
**Model Architecture**: EfficientNet-B0  
**Evaluated Checkpoint**: [`models/{model_name}_best.pth`](file:///{checkpoint_path.as_posix()}) (Experiment 1 Baseline, $256 \\times 256$)  
**Inference Strategy**: Horizontal-Flip Test-Time Augmentation (TTA) vs Single-Pass Baseline  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  
**Experiment Decision**: **{decision}**  

---

## 1. Executive Summary & TTA Procedure

Experiment 9 evaluated whether applying **Horizontal-Flip Test-Time Augmentation (TTA)** at inference time improves classification performance on the BUSI test set using the existing trained Experiment 1 checkpoint, without any model retraining or parameter tuning.

### TTA Algorithmic Flow
For each input ultrasound scan $X$:
1. Compute original prediction logits $Z_1 = f_\theta(X)$ and probabilities $P_1 = \\text{{softmax}}(Z_1)$.
2. Compute horizontally flipped tensor $X_\\text{{flip}} = \\text{{torch.flip}}(X, \\text{{dims}}=[3])$.
3. Compute flipped prediction logits $Z_2 = f_\theta(X_\\text{{flip}})$ and probabilities $P_2 = \\text{{softmax}}(Z_2)$.
4. Average probability vectors: $P_\\text{{TTA}} = \\frac{{P_1 + P_2}}{{2}}$.
5. Output final prediction $\\hat{{y}} = \\arg\\max(P_\\text{{TTA}})$.

**Decision**: **{decision}**  
*Rationale*: {decision_reason}

- **Accuracy**: Baseline: **{b_acc:.2f}%** | TTA: **{t_acc:.2f}%** ({diff_acc:+.2f}%)
- **Macro F1-Score**: Baseline: **{b_f1:.2f}%** | TTA: **{t_f1:.2f}%** ({diff_f1:+.2f}%)
- **Weighted F1-Score**: Baseline: **{b_wf1:.2f}%** | TTA: **{t_wf1:.2f}%** ({t_wf1 - b_wf1:+.2f}%)
- **Malignant Recall**: Baseline: **{b_mal_r:.2f}%** | TTA: **{t_mal_r:.2f}%** ({t_mal_r - b_mal_r:+.2f}%) | **FNs**: **{t_mal_fn}** (vs {b_mal_fn} baseline)
- **Normal Recall**: Baseline: **{b_norm_r:.2f}%** | TTA: **{t_norm_r:.2f}%** ({t_norm_r - b_norm_r:+.2f}%)

---

## 2. Quantitative Comparison Matrix

| Metric | Exp 1 Baseline (Single Pass) | Exp 9 (Exp 1 + H-Flip TTA) | Difference | Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | {b_acc:.2f}% | **{t_acc:.2f}%** | {diff_acc:+.2f}% | {"Higher" if diff_acc > 0 else ("Equal" if diff_acc == 0 else "Lower")} |
| **Macro Precision** | {b_prec:.2f}% | **{t_prec:.2f}%** | {t_prec - b_prec:+.2f}% | {"Higher" if t_prec > b_prec else "Lower"} |
| **Macro Recall** | {b_rec:.2f}% | **{t_rec:.2f}%** | {t_rec - b_rec:+.2f}% | {"Higher" if t_rec > b_rec else "Lower"} |
| **Macro F1-Score** | {b_f1:.2f}% | **{t_f1:.2f}%** | **{diff_f1:+.2f}%** | **{"IMPROVED" if diff_f1 > 0 else ("NO CHANGE" if diff_f1 == 0 else "WORSENED")}** |
| **Weighted F1-Score** | {b_wf1:.2f}% | **{t_wf1:.2f}%** | {t_wf1 - b_wf1:+.2f}% | {"Higher" if t_wf1 > b_wf1 else "Lower"} |
| **Normal Recall** | {b_norm_r:.2f}% | **{t_norm_r:.2f}%** | {t_norm_r - b_norm_r:+.2f}% | {"Higher" if t_norm_r > b_norm_r else "Lower"} |
| **Malignant Recall** | {b_mal_r:.2f}% | **{t_mal_r:.2f}%** | {t_mal_r - b_mal_r:+.2f}% | {"Higher" if t_mal_r > b_mal_r else "Lower"} |
| **Malignant FNs** | {b_mal_fn} | **{t_mal_fn}** | {t_mal_fn - b_mal_fn:+d} | {"Fewer FNs" if t_mal_fn < b_mal_fn else ("Equal" if t_mal_fn == b_mal_fn else "More FNs")} |
| **CPU Latency** | `{avg_single_ms:.2f}` ms | `{avg_tta_ms:.2f}` ms | `{avg_tta_ms - avg_single_ms:+.2f}` ms | ~2x Inference Time |

---

## 3. Detailed Per-Class Breakdown

### Baseline (No TTA) Per-Class Metrics
- **Normal**: Precision: {baseline_metrics['per_class']['Normal']['precision']*100:.2f}% | Recall: {b_norm_r:.2f}% | F1: {baseline_metrics['per_class']['Normal']['f1-score']*100:.2f}%
- **Benign**: Precision: {baseline_metrics['per_class']['Benign']['precision']*100:.2f}% | Recall: {b_ben_r:.2f}% | F1: {baseline_metrics['per_class']['Benign']['f1-score']*100:.2f}%
- **Malignant**: Precision: {baseline_metrics['per_class']['Malignant']['precision']*100:.2f}% | Recall: {b_mal_r:.2f}% | F1: {baseline_metrics['per_class']['Malignant']['f1-score']*100:.2f}%

### TTA Per-Class Metrics
- **Normal**: Precision: {tta_metrics['per_class']['Normal']['precision']*100:.2f}% | Recall: {t_norm_r:.2f}% | F1: {tta_metrics['per_class']['Normal']['f1-score']*100:.2f}%
- **Benign**: Precision: {tta_metrics['per_class']['Benign']['precision']*100:.2f}% | Recall: {t_ben_r:.2f}% | F1: {tta_metrics['per_class']['Benign']['f1-score']*100:.2f}%
- **Malignant**: Precision: {tta_metrics['per_class']['Malignant']['precision']*100:.2f}% | Recall: {t_mal_r:.2f}% | F1: {tta_metrics['per_class']['Malignant']['f1-score']*100:.2f}%

---

## 4. Confusion Matrices Comparison

### Baseline Confusion Matrix (No TTA)
```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          {b_cm[0][0]:<17}{b_cm[0][1]:<19}{b_cm[0][2]}
True Benign:          {b_cm[1][0]:<17}{b_cm[1][1]:<19}{b_cm[1][2]}
True Malignant:       {b_cm[2][0]:<17}{b_cm[2][1]:<19}{b_cm[2][2]}
```

### TTA Confusion Matrix (Horizontal-Flip TTA)
```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          {t_cm[0][0]:<17}{t_cm[0][1]:<19}{t_cm[0][2]}
True Benign:          {t_cm[1][0]:<17}{t_cm[1][1]:<19}{t_cm[1][2]}
True Malignant:       {t_cm[2][0]:<17}{t_cm[2][1]:<19}{t_cm[2][2]}
```

---

## 5. Implementation & Safety Safeguards

1. **No Checkpoint Created**: This is an inference-time experiment. No model weights were retrained or modified.
2. **Production Backend Preserved**: The FastAPI backend continues serving predictions using single-pass inference (`models/efficientnet_b0_best.pth`).
3. **Reproducibility**: Experiment 1's top-performing checkpoint ([`models/efficientnet_b0_exp1_256_best.pth`](file:///{checkpoint_path.as_posix()})) remains preserved untouched.

---

## 6. Recommendation for Next Experiment

- **Experiment 10: Model Architecture Scaling (EfficientNet-B2 at $256 \\times 256$)**
  - **Hypothesis**: Scaling the model capacity from EfficientNet-B0 to EfficientNet-B2 (increasing feature dimension and receptive field) will enhance fine feature representation for subtle malignant boundaries, helping improve accuracy and recall beyond Experiment 1's 75.83% baseline.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[INFO] Wrote experiment report to: {report_path}")

    # 7. Update outputs/model_comparison.csv
    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    if csv_path.exists():
        df_comp = pd.read_csv(csv_path)
    else:
        df_comp = pd.DataFrame(columns=["Model", "Test Accuracy (%)", "Macro Precision (%)", "Macro Recall (%)", "Macro F1-Score (%)", "Weighted F1-Score (%)"])

    exp_row = {
        "Model": "EfficientNet-B0 (Exp9 Exp1 + H-Flip TTA)",
        "Test Accuracy (%)": f"{t_acc:.2f}",
        "Macro Precision (%)": f"{t_prec:.2f}",
        "Macro Recall (%)": f"{t_rec:.2f}",
        "Macro F1-Score (%)": f"{t_f1:.2f}",
        "Weighted F1-Score (%)": f"{t_wf1:.2f}"
    }

    # Remove existing Exp9 entry if present, then append
    df_comp = df_comp[~df_comp["Model"].str.contains("Exp9 Exp1 + H-Flip TTA")]
    df_comp = pd.concat([df_comp, pd.DataFrame([exp_row])], ignore_index=True)
    df_comp.to_csv(csv_path, index=False)
    print(f"[INFO] Updated comparison table saved to: {csv_path}")

    return tta_metrics


if __name__ == "__main__":
    run_experiment_9()
