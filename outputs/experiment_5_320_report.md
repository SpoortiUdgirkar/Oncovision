# OncoVision - Experiment 5 Report: Higher Input Resolution (320x320)

**Generated Date**: October 9, 2026  
**Model Architecture**: EfficientNet-B0  
**Input Resolution**: $320 \times 320$ pixels (Exp 1: $256 \times 256$, Baseline: $224 \times 224$)  
**Checkpoint Path**: [`models/efficientnet_b0_exp5_320_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp5_320_best.pth)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  
**Experiment Decision**: **REJECT**  

---

## 1. Executive Summary

Experiment 5 evaluated whether increasing input resolution to $320 \times 320$ pixels improves classification performance (specifically Malignant and Normal class recall) for EfficientNet-B0 while preserving all other training hyperparameters (BUSI split 545/115/120, seed 42, class-weighted CrossEntropyLoss, AdamW optimizer, and validation loss checkpointing).

**Decision**: **REJECT**  
*Rationale*: Exp 5 (320x320) achieved Macro F1 of 68.09% and Accuracy of 70.83%, failing to improve upon Exp 1 baseline (71.90% Macro F1, 75.83% Acc).

- **Exp 5 (320x320) Accuracy**: **70.83%** (Exp 1: 75.83%, Baseline: 72.50%)
- **Exp 5 (320x320) Macro F1**: **68.09%** (Exp 1: 71.90%, Baseline: 69.47%)
- **Exp 5 (320x320) Weighted F1**: **70.90%** (Exp 1: 75.16%, Baseline: 72.43%)
- **Malignant Recall**: **59.38%** (Exp 1: 56.25%, Baseline: 42.69%)
- **Normal Recall**: **71.43%** (Exp 1: 66.67%, Baseline: 66.67%)

---

## 2. Quantitative Comparison Matrix

| Benchmark Metric | Original Baseline ($224 \times 224$) | Exp 1 Baseline ($256 \times 256$) | Exp 5 ($320 \times 320$) | Diff vs Exp 1 | Status vs Exp 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.50% | 75.83% | **70.83%** | -5.00% | Lower |
| **Macro Precision** | 70.70% | 74.96% | **67.42%** | -7.54% | Lower |
| **Macro Recall** | 68.88% | 70.33% | **68.97%** | -1.36% | Lower |
| **Macro F1-Score** | 69.47% | 71.90% | **68.09%** | **-3.81%** | **WORSENED** |
| **Weighted F1-Score** | 72.43% | 75.16% | **70.90%** | -4.26% | Lower |
| **Normal Recall** | 66.67% | 66.67% | **71.43%** | +4.76% | Higher |
| **Malignant Recall** | 42.69% | 56.25% | **59.38%** | +3.12% | Higher |

---

## 3. Per-Class Detailed Performance Breakdown

| Target Class | Support ($N$) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 21 | 62.50% | 71.43% | 66.67% |
| **Benign** | 67 | 78.46% | 76.12% | 77.27% |
| **Malignant** | 32 | 61.29% | 59.38% | 60.32% |

---

## 4. Confusion Matrix ($N=120$ Test Scans)

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          15               4                  2
True Benign:          6                51                 10
True Malignant:       3                10                 19
```

---

## 5. Runtime & Computational Efficiency Metrics

| Resource Metric | Value | Notes |
| :--- | :--- | :--- |
| **Selected Checkpoint Epoch** | Epoch 24 | Minimum Validation Loss: `0.9677` |
| **Training Time** | `15.00` minutes | 50 total epochs (Stage 1 + Stage 2) |
| **Checkpoint File Size** | `15.59` MB | EfficientNet-B0 state dict |
| **CPU Single Inference Time** | `64.07` ms | Tested on single $320 \times 320$ image batch |
| **Grad-CAM Overlay Time** | `981.26` ms | Target layer `features.7` Grad-CAM extraction |
| **Total Parameters** | `4,011,391` | 3,843 trainable parameters |

---

## 6. Implementation & Safety Safeguards

1. **Production Primary Model Preserved**: The production backend continues to load `models/efficientnet_b0_best.pth` (224x224 baseline) for serving live UI requests.
2. **Experiment Isolation**: All historical checkpoints (`exp1`, `exp2`, `exp3`, `exp4`, `exp5`) are preserved without overwriting.
3. **Frontend & Backend Compatibility**: Validated that the backend and Grad-CAM generation support $320 \times 320$ dynamic input resolution without code breakage.

---

## 7. Recommendation for Next Experiment

- If **REJECT** is **KEEP**: Update primary model candidate to Experiment 5 ($320 \times 320$).
- Next Step: Explore **Cosine Annealing Learning Rate Scheduler** (`CosineAnnealingLR`) or **Focal Loss ($\gamma=2.0$)** to further boost Malignant recall while keeping spatial resolution optimized.
