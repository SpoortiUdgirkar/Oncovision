# OncoVision - Experiment 2 Report: Validation Macro F1 Checkpoint Selection

**Generated Date**: October 8, 2026  
**Model Architecture**: EfficientNet-B0  
**Input Resolution**: $256 \times 256$ pixels  
**Checkpoint Selection Criterion**: Maximum Validation Macro F1 (`val_macro_f1`)  
**Checkpoint Path**: [`models/efficientnet_b0_exp2_val_macro_f1_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp2_val_macro_f1_best.pth)  
**Selected Best Checkpoint Epoch**: Epoch **9** (Best Validation Macro F1: **60.43%**)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  

---

## 1. Executive Summary

Experiment 2 tested whether selecting the best model checkpoint based on **maximum Validation Macro F1-Score** (`val_macro_f1`) rather than minimum validation loss (`val_loss`) improves classification balance, specifically addressing the Malignant class recall drop observed in Experiment 1.

**Key Findings**:
- **Selected Checkpoint Epoch**: Epoch **9** (Achieved **60.43%** validation Macro F1).
- **Test Set Macro F1**: **57.61%** (-11.86% vs Baseline, -14.29% vs Exp 1).
- **Test Set Accuracy**: **62.50%** (-10.00% vs Baseline).
- **Malignant Recall Impact**: Malignant recall is **59.38%** (Malignant F1: **56.72%**).

---

## 2. 3-Way Benchmark Comparison Table

| Benchmark Metric | Baseline ($224 \times 224$) | Exp 1 ($256 \times 256$, Val Loss) | Exp 2 ($256 \times 256$, Val F1) | Exp 2 vs Baseline | Exp 2 vs Exp 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.50% | 75.83% | **62.50%** | -10.00% | -13.33% |
| **Macro Precision** | 70.70% | 74.96% | **60.83%** | -9.87% | -14.13% |
| **Macro Recall** | 68.88% | 70.33% | **56.37%** | -12.51% | -13.96% |
| **Macro F1-Score** | 69.47% | 71.90% | **57.61%** | **-11.86%** | **-14.29%** |
| **Weighted F1-Score** | 72.43% | 75.16% | **61.92%** | -10.51% | -13.24% |

---

## 3. Per-Class Performance Breakdown (Exp 2)

| Target Class | Support | Precision (%) | Recall (%) | F1-Score (%) | Exp 1 F1 (%) | Baseline F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 21.0 | 61.54% | 38.10% | **47.06%** | 68.29% | 63.16% |
| **Benign** | 67.0 | 66.67% | 71.64% | **69.06%** | 81.94% | 77.61% |
| **Malignant** | 32.0 | 54.29% | 59.38% | **56.72%** | 65.45% | 67.65% |

---

## 4. Confusion Matrix (Exp 2)

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          8                11                 2
True Benign:          5                48                 14
True Malignant:       0                13                 19
```

---

## 5. Technical Execution Details

- **Selected Checkpoint Epoch**: Epoch 9
- **Validation Macro F1 at Selected Checkpoint**: 60.43%
- **Total Model Parameters**: 4,011,391
- **Trainable Parameters**: 3,843
- **Training Time**: 7.23 minutes
- **Hardware Device**: CPU
- **Production Model Status**: **Baseline preserved**. Production backend continues serving `models/efficientnet_b0_best.pth` pending final model selection.
