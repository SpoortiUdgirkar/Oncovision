# OncoVision - Experiment 4 Report: WeightedRandomSampler

**Generated Date**: October 9, 2026  
**Model Architecture**: EfficientNet-B0  
**Input Resolution**: $256 \times 256$ pixels  
**Training Sampler**: `WeightedRandomSampler` (Applied to training DataLoader ONLY; Val/Test un-sampled)  
**Loss Function**: Class-weighted `CrossEntropyLoss` (Inverse class frequencies)  
**Checkpoint Path**: [`models/efficientnet_b0_exp4_weighted_sampler_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp4_weighted_sampler_best.pth)  
**Selected Best Checkpoint Epoch**: Epoch **3** (Best Validation Loss: **1.1277**)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  

---

## 1. Executive Summary & Decision

Experiment 4 evaluated the performance impact of adding `WeightedRandomSampler` to the training DataLoader while maintaining the $256 \times 256$ resolution, dataset split (545/115/120, seed 42), optimizer, and validation-loss checkpoint selection from Experiment 1.

**DECISION: REJECT**

- **Experiment 1 (Current Best)**: Accuracy = **75.83%** | Macro F1 = **71.90%** | Malignant Recall = **56.25%** | Normal Recall = **66.67%**
- **Experiment 4 (Weighted Sampler)**: Accuracy = **26.67%** (-49.16%) | Macro F1 = **26.97%** (-44.93%) | Malignant Recall = **31.25%** (-25.00%) | Normal Recall = **80.95%** (+14.28%)

---

## 2. Quantitative Comparison Matrix (Exp 4 vs Exp 1 & Baseline)

| Benchmark Metric | Baseline ($224 \times 224$) | Exp 1 ($256 \times 256$) | Exp 4 (Weighted Sampler) | Delta vs Exp 1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.50% | 75.83% | **26.67%** | -49.16% | Lower |
| **Macro Precision** | 70.70% | 74.96% | **45.19%** | -29.77% | Lower |
| **Macro Recall** | 68.88% | 70.33% | **39.89%** | -30.44% | Lower |
| **Macro F1-Score** | 69.47% | 71.90% | **26.97%** | **-44.93%** | **WORSENED** |
| **Weighted F1-Score** | 72.43% | 75.16% | **22.73%** | -52.43% | Lower |

---

## 3. Per-Class Metrics & Recall Analysis

| Target Class | Support ($N$) | Exp 4 Precision (%) | Exp 4 Recall (%) | Exp 4 F1-Score (%) | Exp 1 Recall (%) | Exp 1 F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 21.0 | 18.68% | **80.95%** | **30.36%** | 66.67% | 68.29% |
| **Benign** | 67.0 | 71.43% | **7.46%** | **13.51%** | 88.06% | 81.94% |
| **Malignant** | 32.0 | 45.45% | **31.25%** | **37.04%** | 56.25% | 65.45% |

### Detailed Recall Discussion
- **Malignant Recall**: Changed from 56.25% (Exp 1) to **31.25%** (-25.00%).
- **Normal Recall**: Changed from 66.67% (Exp 1) to **80.95%** (+14.28%).
- **Overcorrection Analysis**: Combining `WeightedRandomSampler` with class-weighted `CrossEntropyLoss` applies double class-imbalance correction. This double correction led to oversampling noise and reduced overall precision on Benign scans.

---

## 4. Confusion Matrix (Exp 4)

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          17               0                  4
True Benign:          54               5                  8
True Malignant:       20               2                  10
```

---

## 5. Technical & Computational Execution Metrics

- **Selected Checkpoint Epoch**: Epoch 3
- **Validation Loss at Best Checkpoint**: 1.1277
- **Total Model Parameters**: 4,011,391
- **Trainable Head Parameters**: 3,843
- **Checkpoint File Size**: 15.60 MB
- **Total Training Time**: 3.89 minutes
- **CPU Single-Sample Inference Time**: 65.36 ms per scan
- **Grad-CAM Overlay Generation Time**: 1039.96 ms per scan
- **Production Server Status**: **Preserved**. `models/efficientnet_b0_best.pth` remains the active serving checkpoint.
