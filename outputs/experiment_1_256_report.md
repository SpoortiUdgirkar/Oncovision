# OncoVision - Experiment 1 Report: Higher Input Resolution (256x256)

**Generated Date**: October 8, 2026  
**Model Architecture**: EfficientNet-B0  
**Resolution**: $256 \times 256$ pixels (Baseline: $224 \times 224$)  
**Checkpoint Path**: [`models/efficientnet_b0_exp1_256_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp1_256_best.pth)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  

---

## 1. Executive Summary

Experiment 1 evaluated the performance impact of increasing input resolution from $224 \times 224$ to $256 \times 256$ pixels for EfficientNet-B0 while preserving all other training conditions (dataset split, seed, loss weights, optimizer, and learning rate schedules).

**Outcome**: Experiment 1 **IMPROVED** the baseline performance.
- **Baseline (224x224) Macro F1**: 69.47% | **Accuracy**: 72.50%
- **Exp 1 (256x256) Macro F1**: **71.90%** (+2.43%) | **Accuracy**: **75.83%** (+3.33%)

---

## 2. Quantitative Comparison against Baseline

| Benchmark Metric | Baseline ($224 \times 224$) | Exp 1 ($256 \times 256$) | Delta (Abs Diff) | Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.50% | **75.83%** | +3.33% | Higher |
| **Macro Precision** | 70.70% | **74.96%** | +4.26% | Higher |
| **Macro Recall** | 68.88% | **70.33%** | +1.45% | Higher |
| **Macro F1-Score** | 69.47% | **71.90%** | **+2.43%** | **IMPROVED** |
| **Weighted F1-Score** | 72.43% | **75.16%** | +2.73% | Higher |

---

## 3. Per-Class Performance Breakdown

| Target Class | Support | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 21.0 | 70.00% | 66.67% | 68.29% |
| **Benign** | 67.0 | 76.62% | 88.06% | 81.94% |
| **Malignant** | 32.0 | 78.26% | 56.25% | 65.45% |

---

## 4. Confusion Matrix

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          14               6                  1
True Benign:          4                59                 4
True Malignant:       2                12                 18
```

---

## 5. Technical Execution Details

- **Total Model Parameters**: 4,011,391
- **Trainable Parameters**: 3,843
- **Total Training Time**: 13.66 minutes
- **Hardware Device**: CPU
- **Primary Model Status**: **Baseline preserved**. Production backend continues to serve `models/efficientnet_b0_best.pth` until final model selection.
