# OncoVision - Experiment 3 Report: Label Smoothing (0.1)

**Generated Date**: October 8, 2026  
**Model Architecture**: EfficientNet-B0  
**Input Resolution**: $256 \times 256$ pixels  
**Loss Function**: `CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)`  
**Checkpoint Path**: [`models/efficientnet_b0_exp3_label_smoothing_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp3_label_smoothing_best.pth)  
**Selected Best Checkpoint Epoch**: Epoch **23** (Best Validation Loss: **1.0266**)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  

---

## 1. Executive Summary & Decision

Experiment 3 evaluated the impact of adding **Label Smoothing ($0.1$)** to the class-weighted CrossEntropy loss function on top of the successful $256 \times 256$ resolution configuration (Experiment 1).

**DECISION: REJECT**

- **Experiment 1 (Current Best)**: Accuracy = **75.83%** | Macro F1 = **71.90%**
- **Experiment 3 (Label Smoothing 0.1)**: Accuracy = **64.17%** (-11.66%) | Macro F1 = **62.39%** (-9.51%)

---

## 2. 4-Way Benchmark Comparison Table

| Model Architecture / Experiment | Resolution | Selection Metric | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1-Score (%) | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | $224 \times 224$ | Val Loss | 72.50% | 70.70% | 68.88% | 69.47% | 72.43% | Baseline |
| **Experiment 1** | $256 \times 256$ | Val Loss | **75.83%** | **74.96%** | **70.33%** | **71.90%** | **75.16%** | **BEST MODEL** |
| **Experiment 2** | $256 \times 256$ | Val Macro F1 | 62.50% | 60.83% | 56.37% | 57.61% | 61.92% | REJECTED |
| **Experiment 3** | $256 \times 256$ | Val Loss (Smooth 0.1) | **64.17%** | **62.05%** | **66.63%** | **62.39%** | **65.12%** | **REJECT** |

---

## 3. Per-Class Performance Breakdown (Exp 3)

| Target Class | Support | Precision (%) | Recall (%) | F1-Score (%) | Exp 1 F1 (%) | Baseline F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 21.0 | 43.24% | 76.19% | **55.17%** | 68.29% | 63.16% |
| **Benign** | 67.0 | 80.39% | 61.19% | **69.49%** | 81.94% | 77.61% |
| **Malignant** | 32.0 | 62.50% | 62.50% | **62.50%** | 65.45% | 67.65% |

---

## 4. Confusion Matrix (Exp 3)

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          16               4                  1
True Benign:          15               41                 11
True Malignant:       6                6                  20
```

---

## 5. Technical Execution Details

- **Selected Checkpoint Epoch**: Epoch 23
- **Validation Loss at Selected Checkpoint**: 1.0266
- **Total Model Parameters**: 4,011,391
- **Trainable Parameters**: 3,843
- **Training Time**: 12.70 minutes
- **Hardware Device**: CPU
- **Production Primary Model Status**: **Baseline preserved**. Production backend continues serving `models/efficientnet_b0_best.pth`.
