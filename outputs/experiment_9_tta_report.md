# OncoVision - Experiment 9 Report: Horizontal-Flip Test-Time Augmentation (TTA)

**Generated Date**: October 9, 2026  
**Model Architecture**: EfficientNet-B0  
**Evaluated Checkpoint**: [`models/efficientnet_b0_exp1_256_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp1_256_best.pth) (Experiment 1 Baseline, $256 \times 256$)  
**Inference Strategy**: Horizontal-Flip Test-Time Augmentation (TTA) vs Single-Pass Baseline  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  
**Experiment Decision**: **KEEP**  

---

## 1. Executive Summary & TTA Procedure

Experiment 9 evaluated whether applying **Horizontal-Flip Test-Time Augmentation (TTA)** at inference time improves classification performance on the BUSI test set using the existing trained Experiment 1 checkpoint, without any model retraining or parameter tuning.

### TTA Algorithmic Flow
For each input ultrasound scan $X$:
1. Compute original prediction logits $Z_1 = f_	heta(X)$ and probabilities $P_1 = \text{softmax}(Z_1)$.
2. Compute horizontally flipped tensor $X_\text{flip} = \text{torch.flip}(X, \text{dims}=[3])$.
3. Compute flipped prediction logits $Z_2 = f_	heta(X_\text{flip})$ and probabilities $P_2 = \text{softmax}(Z_2)$.
4. Average probability vectors: $P_\text{TTA} = \frac{P_1 + P_2}{2}$.
5. Output final prediction $\hat{y} = \arg\max(P_\text{TTA})$.

**Decision**: **KEEP**  
*Rationale*: Horizontal-Flip TTA improved or maintained Macro F1 (73.00% vs 71.90%) and Accuracy (75.83% vs 75.83%).

- **Accuracy**: Baseline: **75.83%** | TTA: **75.83%** (+0.00%)
- **Macro F1-Score**: Baseline: **71.90%** | TTA: **73.00%** (+1.10%)
- **Weighted F1-Score**: Baseline: **75.16%** | TTA: **75.44%** (+0.29%)
- **Malignant Recall**: Baseline: **56.25%** | TTA: **59.38%** (+3.12%) | **FNs**: **13** (vs 14 baseline)
- **Normal Recall**: Baseline: **66.67%** | TTA: **71.43%** (+4.76%)

---

## 2. Quantitative Comparison Matrix

| Metric | Exp 1 Baseline (Single Pass) | Exp 9 (Exp 1 + H-Flip TTA) | Difference | Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 75.83% | **75.83%** | +0.00% | Equal |
| **Macro Precision** | 74.96% | **74.48%** | -0.48% | Lower |
| **Macro Recall** | 70.33% | **71.96%** | +1.63% | Higher |
| **Macro F1-Score** | 71.90% | **73.00%** | **+1.10%** | **IMPROVED** |
| **Weighted F1-Score** | 75.16% | **75.44%** | +0.29% | Higher |
| **Normal Recall** | 66.67% | **71.43%** | +4.76% | Higher |
| **Malignant Recall** | 56.25% | **59.38%** | +3.12% | Higher |
| **Malignant FNs** | 14 | **13** | -1 | Fewer FNs |
| **CPU Latency** | `41.28` ms | `76.38` ms | `+35.10` ms | ~2x Inference Time |

---

## 3. Detailed Per-Class Breakdown

### Baseline (No TTA) Per-Class Metrics
- **Normal**: Precision: 70.00% | Recall: 66.67% | F1: 68.29%
- **Benign**: Precision: 76.62% | Recall: 88.06% | F1: 81.94%
- **Malignant**: Precision: 78.26% | Recall: 56.25% | F1: 65.45%

### TTA Per-Class Metrics
- **Normal**: Precision: 75.00% | Recall: 71.43% | F1: 73.17%
- **Benign**: Precision: 78.08% | Recall: 85.07% | F1: 81.43%
- **Malignant**: Precision: 70.37% | Recall: 59.38% | F1: 64.41%

---

## 4. Confusion Matrices Comparison

### Baseline Confusion Matrix (No TTA)
```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          14               6                  1
True Benign:          4                59                 4
True Malignant:       2                12                 18
```

### TTA Confusion Matrix (Horizontal-Flip TTA)
```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          15               5                  1
True Benign:          3                57                 7
True Malignant:       2                11                 19
```

---

## 5. Implementation & Safety Safeguards

1. **No Checkpoint Created**: This is an inference-time experiment. No model weights were retrained or modified.
2. **Production Backend Preserved**: The FastAPI backend continues serving predictions using single-pass inference (`models/efficientnet_b0_best.pth`).
3. **Reproducibility**: Experiment 1's top-performing checkpoint ([`models/efficientnet_b0_exp1_256_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp1_256_best.pth)) remains preserved untouched.

---

## 6. Recommendation for Next Experiment

- **Experiment 10: Model Architecture Scaling (EfficientNet-B2 at $256 \times 256$)**
  - **Hypothesis**: Scaling the model capacity from EfficientNet-B0 to EfficientNet-B2 (increasing feature dimension and receptive field) will enhance fine feature representation for subtle malignant boundaries, helping improve accuracy and recall beyond Experiment 1's 75.83% baseline.
