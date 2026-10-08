# OncoVision - Experiment 9 Verification & Technical Audit Report

**Generated Date**: October 9, 2026  
**Evaluated Model**: EfficientNet-B0 ($256 \times 256$ input resolution)  
**Checkpoint Verified**: [`models/efficientnet_b0_exp1_256_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp1_256_best.pth)  
**Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  
**Hardware Device**: CPU (`CPU`)  

---

## 1. Verification Summary & Methodological Integrity

This audit re-evaluated **Experiment 1 (Single Pass)** and **Experiment 9 (Horizontal-Flip TTA)** to verify mathematical correctness, confusion matrix alignment, probability vector averaging symmetry, and execution latency.

### Key Audit Findings
1. **Zero Retraining & Weight Stability**: Confirmed model weights remained 100% frozen during evaluation (`model.eval()`). No checkpoint was overwritten or modified.
2. **Fixed Evaluation Dataset**: All evaluations used the exact same 120-image test set ($N=120$: Normal = 21, Benign = 67, Malignant = 32).
3. **Probability Vector Averaging**: Confirmed TTA executes $\frac{\text{softmax}(f(X)) + \text{softmax}(f(\text{flip}(X)))}{2.0}$, producing a valid probability distribution (summing to $1.0$).
4. **Mathematical Alignment**: Recomputed all metrics directly from confusion matrices. All reported class supports, row sums, column sums, per-class F1-scores, and global averages align with 100% precision.

---

## 2. Re-Evaluated Performance Metrics ($N=120$ Scans)

| Benchmark Metric | Exp 1 Baseline (Single Pass) | Exp 9 (Exp 1 + H-Flip TTA) | Absolute Difference | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 75.83% | **75.83%** | +0.00% | Verified |
| **Macro Precision** | 74.96% | **74.48%** | -0.48% | Verified |
| **Macro Recall** | 70.33% | **71.96%** | **+1.63%** | Verified |
| **Macro F1-Score** | 71.90% | **73.00%** | **+1.10%** | **Verified (Improved)** |
| **Weighted F1-Score** | 75.16% | **75.44%** | **+0.28%** | Verified |
| **Normal Recall** | 66.67% | **71.43%** | **+4.76%** | Verified |
| **Benign Recall** | 88.06% | **85.07%** | -2.99% | Verified |
| **Malignant Recall** | 56.25% | **59.38%** | **+3.12%** | Verified |
| **Malignant False Negatives** | 14 | **13** | **-1 FN** | Verified |

---

## 3. Recomputed Confusion Matrices & Math Alignment

### Exp 1 Baseline Confusion Matrix (Single Pass)
```text
               Predicted Normal   Predicted Benign   Predicted Malignant    Row Total (True Support)
True Normal:          14                 6                  1                    21
True Benign:           4                59                  4                    67
True Malignant:        2                12                 18                    32
Col Total:            20                77                 23                   120
```

### Exp 9 TTA Confusion Matrix (Horizontal-Flip TTA)
```text
               Predicted Normal   Predicted Benign   Predicted Malignant    Row Total (True Support)
True Normal:          15                 5                  1                    21
True Benign:           3                57                  7                    67
True Malignant:        2                11                 19                    32
Col Total:            20                73                 27                   120
```

- **Row Sum Check**: Row sums equal {21, 67, 32} for both single-pass and TTA confusion matrices.
- **Accuracy Calculation**:
  - Baseline Acc $= (14 + 59 + 18) / 120 = 91 / 120 = 75.83\%$
  - TTA Acc $= (15 + 57 + 19) / 120 = 91 / 120 = 75.83\%$
- **Macro F1 Calculation**:
  - Baseline Macro F1 $= (0.6829 + 0.8194 + 0.6545) / 3 = 71.90\%$
  - TTA Macro F1 $= (0.7317 + 0.8143 + 0.6441) / 3 = 73.00\%$

---

## 4. Latency Benchmark & Resource Breakdown

Benchmarked using 50 repetitions on CPU with 10 warmup iterations.

| Execution Mode | Mean Latency (ms) | Median Latency (ms) | Preprocessing Included? | Grad-CAM Included? | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-Pass Inference** | `43.50` ms | `42.77` ms | No | No | Pure model forward pass |
| **Horizontal-Flip TTA** | `91.88` ms | `84.69` ms | No | No | 2 forward passes + softmax avg |
| **Full Pipeline (Single Pass)** | `998.95` ms | `998.95` ms | Yes | Yes | Image decode + Tensors + Grad-CAM |

*Note: Latency is reported purely as an engineering resource metric. No claims regarding clinical real-time deployment or diagnostic safety are made.*

---

## 5. Model Comparison CSV Audit

Audited [`outputs/model_comparison.csv`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/outputs/model_comparison.csv):
- Column headers match standard schema: `['Model', 'Test Accuracy (%)', 'Macro Precision (%)', 'Macro Recall (%)', 'Macro F1-Score (%)', 'Weighted F1-Score (%)']`.
- Exp 9 entry correctly listed as `"EfficientNet-B0 (Exp9 Exp1 + H-Flip TTA)"`.
- All metrics formatted consistently to 2 decimal places.

---

## 6. Disclaimers & Operational Notice

1. **Non-Diagnostic Tool**: OncoVision and its associated model artifacts serve strictly as an AI research demonstration and decision-support prototype. It is **NOT** a certified medical device, diagnostic software, or clinical solution.
2. **Threshold Sensitivity**: Probability thresholds were kept un-tuned at default $0.5$ argmax selection to prevent test-set overfitting.
3. **Production State Preserved**: The FastAPI backend continues to load `models/efficientnet_b0_best.pth` as the primary production checkpoint.
