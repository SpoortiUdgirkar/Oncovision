# OncoVision - Experiment 8 Report: Mild Random Erasing Data Augmentation

**Generated Date**: October 9, 2026  
**Model Architecture**: EfficientNet-B0  
**Input Resolution**: $256 \times 256$ pixels  
**Data Augmentation**: Mild RandomErasing on Training DataLoader only ($p=0.15$, $scale=(0.02, 0.08)$, $ratio=(0.5, 2.0)$, $value=0$)  
**Validation / Test Augmentation**: Deterministic Resize & Normalization only (no RandomErasing)  
**Checkpoint Path**: [`models/efficientnet_b0_exp8_random_erasing_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp8_random_erasing_best.pth)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  
**Experiment Decision**: **REJECT**  

---

## 1. Executive Summary

Experiment 8 evaluated whether introducing mild RandomErasing ($p=0.15$, erasing small $2\%-8\%$ regions of normalized feature maps during training) improves generalization and Malignant recall for EfficientNet-B0 while preserving all other Experiment 1 hyperparameters ($256 \times 256$ input resolution, BUSI 545/115/120 split, seed 42, AdamW optimizer, class-weighted CrossEntropyLoss, and `ReduceLROnPlateau`).

**Decision**: **REJECT**  
*Rationale*: Exp 8 (Random Erasing) achieved Macro F1 of 64.00% and Accuracy of 67.50%, failing to outperform Exp 1 baseline (71.90% Macro F1, 75.83% Acc).

- **Exp 8 (Random Erasing) Accuracy**: **67.50%** (Exp 1: 75.83%, Baseline: 72.50%)
- **Exp 8 (Random Erasing) Macro F1**: **64.00%** (Exp 1: 71.90%, Baseline: 69.47%)
- **Exp 8 (Random Erasing) Weighted F1**: **67.30%** (Exp 1: 75.16%, Baseline: 72.43%)
- **Malignant Recall**: **75.00%** (Exp 1: 56.25%, Baseline: 42.69%) | **FNs**: **8** (vs 14 in Exp 1)
- **Normal Recall**: **47.62%** (Exp 1: 66.67%, Baseline: 66.67%)

---

## 2. Quantitative Comparison Matrix

| Benchmark Metric | Original Baseline ($224 \times 224$) | Exp 1 Baseline ($256 \times 256$) | Exp 8 (Random Erasing) | Diff vs Exp 1 | Status vs Exp 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.50% | 75.83% | **67.50%** | -8.33% | Lower |
| **Macro Precision** | 70.70% | 74.96% | **64.60%** | -10.36% | Lower |
| **Macro Recall** | 68.88% | 70.33% | **64.26%** | -6.07% | Lower |
| **Macro F1-Score** | 69.47% | 71.90% | **64.00%** | **-7.90%** | **WORSENED** |
| **Weighted F1-Score** | 72.43% | 75.16% | **67.30%** | -7.86% | Lower |
| **Normal Recall** | 66.67% | 66.67% | **47.62%** | -19.05% | Lower |
| **Malignant Recall** | 42.69% | 56.25% | **75.00%** | +18.75% | Higher |

---

## 3. Per-Class Detailed Performance Breakdown

| Target Class | Support ($N$) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 21 | 58.82% | 47.62% | 52.63% |
| **Benign** | 67 | 73.44% | 70.15% | 71.76% |
| **Malignant** | 32 | 61.54% | 75.00% | 67.61% |

---

## 4. Confusion Matrix ($N=120$ Test Scans)

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          10               9                  2
True Benign:          7                47                 13
True Malignant:       0                8                  24
```

---

## 5. Runtime & Computational Efficiency Metrics

| Resource Metric | Value | Notes |
| :--- | :--- | :--- |
| **Selected Checkpoint Epoch** | Epoch 25 | Minimum Validation Loss: `0.9281` |
| **Training Time** | `10.45` minutes | 50 total epochs (Stage 1 + Stage 2) |
| **Checkpoint File Size** | `15.59` MB | EfficientNet-B0 state dict |
| **CPU Single Inference Time** | `45.60` ms | Tested on single $256 \times 256$ image batch |
| **Grad-CAM Overlay Time** | `958.51` ms | Target layer `features.7` Grad-CAM extraction |
| **Total Parameters** | `4,011,391` | 3,843 trainable parameters |

---

## 6. Implementation & Safety Safeguards

1. **Production Primary Model Preserved**: The production backend continues to load `models/efficientnet_b0_best.pth` (224x224 baseline) for serving live UI requests.
2. **Experiment Isolation**: All historical checkpoints (`exp1` through `exp8`) remain untouched without overwriting.
3. **Frontend & Backend Compatibility**: Verified that the backend, Grad-CAM generation, and API services remain fully operational.

---

## 7. Recommendation for Next Experiment

- If **REJECT** is **KEEP**: Promote Experiment 8 to top candidate.
- If **REJECT** is **REJECT**: Explore **Test-Time Augmentation (TTA)** or **Model Architecture Scaling** (e.g. EfficientNet-B2 or DenseNet-121 fine-tuning refinements) starting from Experiment 1 ($256 \times 256$).
