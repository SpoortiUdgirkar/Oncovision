# OncoVision - Experiment 7 Report: Cosine Annealing Learning Rate Scheduler

**Generated Date**: October 9, 2026  
**Model Architecture**: EfficientNet-B0  
**Input Resolution**: $256 \times 256$ pixels  
**Learning Rate Scheduler**: `CosineAnnealingLR` (Stage 1 $T_\text{max}=20$, Stage 2 $T_\text{max}=5$, $\eta_\text{min}=10^{-6}$)  
**Loss Function**: Class-Weighted CrossEntropyLoss (`label_smoothing=0.0`)  
**Checkpoint Path**: [`models/efficientnet_b0_exp7_cosine_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp7_cosine_best.pth)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  
**Experiment Decision**: **REJECT**  

---

## 1. Executive Summary

Experiment 7 evaluated whether replacing `ReduceLROnPlateau` with a smooth `CosineAnnealingLR` schedule improves overall classification performance, Malignant recall, and Normal recall for EfficientNet-B0 while keeping all other Experiment 1 parameters constant ($256 \times 256$ input size, BUSI 545/115/120 split, seed 42, AdamW optimizer, class-weighted CrossEntropyLoss).

**Decision**: **REJECT**  
*Rationale*: Exp 7 (Cosine Scheduler) achieved Macro F1 of 58.74% and Accuracy of 64.17%, failing to outperform Exp 1 baseline (71.90% Macro F1, 75.83% Acc).

- **Exp 7 (Cosine Scheduler) Accuracy**: **64.17%** (Exp 1: 75.83%, Baseline: 72.50%)
- **Exp 7 (Cosine Scheduler) Macro F1**: **58.74%** (Exp 1: 71.90%, Baseline: 69.47%)
- **Exp 7 (Cosine Scheduler) Weighted F1**: **63.46%** (Exp 1: 75.16%, Baseline: 72.43%)
- **Malignant Recall**: **43.75%** (Exp 1: 56.25%, Baseline: 42.69%) | **FNs**: **18** (vs 14 in Exp 1)
- **Normal Recall**: **52.38%** (Exp 1: 66.67%, Baseline: 66.67%)

---

## 2. Quantitative Comparison Matrix

| Benchmark Metric | Original Baseline ($224 \times 224$) | Exp 1 Baseline ($256 \times 256$) | Exp 7 (Cosine) | Diff vs Exp 1 | Status vs Exp 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.50% | 75.83% | **64.17%** | -11.66% | Lower |
| **Macro Precision** | 70.70% | 74.96% | **60.01%** | -14.95% | Lower |
| **Macro Recall** | 68.88% | 70.33% | **57.91%** | -12.42% | Lower |
| **Macro F1-Score** | 69.47% | 71.90% | **58.74%** | **-13.16%** | **WORSENED** |
| **Weighted F1-Score** | 72.43% | 75.16% | **63.46%** | -11.70% | Lower |
| **Normal Recall** | 66.67% | 66.67% | **52.38%** | -14.29% | Lower |
| **Malignant Recall** | 42.69% | 56.25% | **43.75%** | -12.50% | Lower |

---

## 3. Per-Class Detailed Performance Breakdown

| Target Class | Support ($N$) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 21 | 57.89% | 52.38% | 55.00% |
| **Benign** | 67 | 70.27% | 77.61% | 73.76% |
| **Malignant** | 32 | 51.85% | 43.75% | 47.46% |

---

## 4. Confusion Matrix ($N=120$ Test Scans)

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          11               7                  3
True Benign:          5                52                 10
True Malignant:       3                15                 14
```

---

## 5. Runtime & Computational Efficiency Metrics

| Resource Metric | Value | Notes |
| :--- | :--- | :--- |
| **Selected Checkpoint Epoch** | Epoch 19 | Minimum Validation Loss: `0.9954` |
| **Training Time** | `10.20` minutes | 50 total epochs (Stage 1 + Stage 2) |
| **Checkpoint File Size** | `15.59` MB | EfficientNet-B0 state dict |
| **CPU Single Inference Time** | `47.02` ms | Tested on single $256 \times 256$ image batch |
| **Grad-CAM Overlay Time** | `946.68` ms | Target layer `features.7` Grad-CAM extraction |
| **Total Parameters** | `4,011,391` | 3,843 trainable parameters |

---

## 6. Implementation & Safety Safeguards

1. **Production Primary Model Preserved**: The production backend continues to load `models/efficientnet_b0_best.pth` (224x224 baseline) for serving live UI requests.
2. **Experiment Isolation**: All historical checkpoints (`exp1` through `exp7`) remain untouched without overwriting.
3. **Frontend & Backend Compatibility**: Verified that the backend, Grad-CAM generation, and API services remain fully operational.

---

## 7. Recommendation for Next Experiment

- If **REJECT** is **KEEP**: Promote Experiment 7 to top candidate.
- If **REJECT** is **REJECT**: Explore **Data Augmentation Enhancements** (e.g. RandAugment / Elastic Transformations / Random Erasing) starting from Experiment 1 ($256 \times 256$, `ReduceLROnPlateau`).
