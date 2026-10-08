# OncoVision - Experiment 6 Report: Focal Loss (gamma=2.0)

**Generated Date**: October 9, 2026  
**Model Architecture**: EfficientNet-B0  
**Input Resolution**: $256 \times 256$ pixels  
**Loss Function**: Unweighted Multiclass Focal Loss ($\gamma = 2.0$, $\alpha = \text{None}$)  
**Checkpoint Path**: [`models/efficientnet_b0_exp6_focal_loss_best.pth`](file:///C:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp6_focal_loss_best.pth)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  
**Experiment Decision**: **REJECT**  

---

## 1. Focal Loss Mathematical Specification & Implementation

Standard Cross-Entropy Loss evaluates sample loss as:
$$CE(p_t) = -\log(p_t)$$

The Multiclass Focal Loss (Lin et al., 2017) introduces a modulating factor $(1 - p_t)^\gamma$ to down-weight easy examples ($p_t \to 1$) and focus training gradients on hard, misclassified samples ($p_t \to 0$):
$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t = \text{softmax}(\text{logits})_t$ is the model's estimated probability for the ground-truth target class $t$.
- $\gamma = 2.0$ is the focusing parameter.
- $\alpha = \text{None}$ specifies equal class weighting ($1.0$ for all classes) to evaluate Focal Loss independently without double-weighting.

---

## 2. Executive Summary

Experiment 6 evaluated whether replacing class-weighted CrossEntropyLoss with unweighted Multiclass Focal Loss ($\gamma=2.0$) improves overall classification performance, Malignant recall, and Normal recall while maintaining the Experiment 1 baseline setup ($256 \times 256$ input size, BUSI 545/115/120 split, seed 42, AdamW, ReduceLROnPlateau).

**Decision**: **REJECT**  
*Rationale*: Exp 6 (Focal Loss) achieved Macro F1 of 59.85% and Accuracy of 70.83%, failing to outperform Exp 1 baseline (71.90% Macro F1, 75.83% Acc).

- **Exp 6 (Focal Loss) Accuracy**: **70.83%** (Exp 1: 75.83%, Baseline: 72.50%)
- **Exp 6 (Focal Loss) Macro F1**: **59.85%** (Exp 1: 71.90%, Baseline: 69.47%)
- **Exp 6 (Focal Loss) Weighted F1**: **67.39%** (Exp 1: 75.16%, Baseline: 72.43%)
- **Malignant Recall**: **50.00%** (Exp 1: 56.25%, Baseline: 42.69%) | **FNs**: **16** (vs 14 in Exp 1)
- **Normal Recall**: **23.81%** (Exp 1: 66.67%, Baseline: 66.67%)

---

## 3. Quantitative Comparison Matrix

| Benchmark Metric | Original Baseline ($224 \times 224$) | Exp 1 Baseline ($256 \times 256$) | Exp 6 (Focal Loss) | Diff vs Exp 1 | Status vs Exp 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 72.50% | 75.83% | **70.83%** | -5.00% | Lower |
| **Macro Precision** | 70.70% | 74.96% | **79.63%** | +4.67% | Higher |
| **Macro Recall** | 68.88% | 70.33% | **56.44%** | -13.89% | Lower |
| **Macro F1-Score** | 69.47% | 71.90% | **59.85%** | **-12.05%** | **WORSENED** |
| **Weighted F1-Score** | 72.43% | 75.16% | **67.39%** | -7.77% | Lower |
| **Normal Recall** | 66.67% | 66.67% | **23.81%** | -42.86% | Lower |
| **Malignant Recall** | 42.69% | 56.25% | **50.00%** | -6.25% | Lower |

---

## 4. Per-Class Detailed Performance Breakdown

| Target Class | Support ($N$) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 21 | 83.33% | 23.81% | 37.04% |
| **Benign** | 67 | 66.67% | 95.52% | 78.53% |
| **Malignant** | 32 | 88.89% | 50.00% | 64.00% |

---

## 5. Confusion Matrix ($N=120$ Test Scans)

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal:          5                16                 0
True Benign:          1                64                 2
True Malignant:       0                16                 16
```

---

## 6. Runtime & Computational Efficiency Metrics

| Resource Metric | Value | Notes |
| :--- | :--- | :--- |
| **Selected Checkpoint Epoch** | Epoch 25 | Minimum Validation Loss: `0.3366` |
| **Training Time** | `11.02` minutes | 50 total epochs (Stage 1 + Stage 2) |
| **Checkpoint File Size** | `15.59` MB | EfficientNet-B0 state dict |
| **CPU Single Inference Time** | `61.11` ms | Tested on single $256 \times 256$ image batch |
| **Grad-CAM Overlay Time** | `953.04` ms | Target layer `features.7` Grad-CAM extraction |
| **Total Parameters** | `4,011,391` | 3,843 trainable parameters |

---

## 7. Implementation & Safety Safeguards

1. **Production Primary Model Preserved**: The production backend continues to load `models/efficientnet_b0_best.pth` (224x224 baseline) for serving live UI requests.
2. **Experiment Isolation**: All historical checkpoints (`exp1` through `exp6`) remain untouched without overwriting.
3. **Frontend & Backend Compatibility**: Verified that the backend, Grad-CAM generation, and API services remain fully operational.

---

## 8. Recommendation for Next Experiment

- If **REJECT** is **KEEP**: Promote Experiment 6 to leading candidate.
- If **REJECT** is **REJECT**: Explore **Class-Weighted Focal Loss** or **Cosine Annealing Learning Rate Scheduler** (`CosineAnnealingLR`) starting from the successful Experiment 1 baseline ($256 \times 256$).
