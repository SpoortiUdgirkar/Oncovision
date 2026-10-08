# OncoVision - Deep Learning Model & Transfer Learning Specification

This document details the neural network architecture, transfer learning strategy, 2-stage training methodology, loss functions, optimization algorithms, and empirical evaluation results for **OncoVision**.

---

## 1. Architecture Overview: ResNet50 Transfer Learning

OncoVision utilizes a **ResNet50 (Residual Network with 50 layers)** backbone pretrained on the ImageNet dataset as its primary vision feature extractor.

### Network Adaptation Architecture

```text
Input Ultrasound Image [3 x 224 x 224]
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ ResNet50 Conv Backbone (Conv1 -> Layer1 -> Layer2 ->   │
│ Layer3 -> Layer4)                                       │
│ Output Feature Map: [2048 x 7 x 7]                     │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Adaptive Global Average Pooling                        │
│ Output Vector: [2048]                                  │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Custom Classification Head (model.fc)                  │
│  ├── Dropout (p = 0.3)                                 │
│  └── Linear (in_features = 2048, out_features = 3)     │
└────────────────────────────────────────────────────────┘
        │
        ▼
Logits Vector [3] -> Softmax -> Class Probabilities:
  - Class 0: Normal
  - Class 1: Benign
  - Class 2: Malignant
```

---

## 2. 2-Stage Transfer Learning Strategy

To preserve low-level visual filters (edges, textures, speckle noise) learned on ImageNet while adapting high-level semantic features to breast ultrasound lesions, training is conducted in **2 distinct stages**:

### Stage 1: Feature Extraction (Backbone Frozen)
* **Backbone Layers**: `layer1`, `layer2`, `layer3`, `layer4` parameter gradients frozen (`requires_grad = False`).
* **Classification Head**: `model.fc` parameters trainable (`requires_grad = True`).
* **Learning Rate**: $1 \times 10^{-4}$
* **Objective**: Train the randomly initialized final classification head to map 2048-dimensional pooling vectors to class logits without altering deep feature extractors.

### Stage 2: Fine-Tuning (Selected Deep Layers Unfrozen)
* **Backbone Unfreezing**: Deeper residual blocks in `layer4` (the final residual stage) are unfrozen (`requires_grad = True`).
* **Learning Rate**: Reduced by $10\times$ to $1 \times 10^{-5}$ to prevent catastrophic forgetting of pretrained weights.
* **Objective**: Adapt high-level receptive fields to specialized ultrasound acoustic properties (hypoechoic masses, spiculation, posterior shadowing).

---

## 3. Loss Function & Class Imbalance Weighting

The BUSI dataset contains a **3.29 : 1** class imbalance ratio (437 Benign vs 133 Normal). To prevent the model from biasing predictions toward the majority class, inverse class frequency weights are computed and injected into `torch.nn.CrossEntropyLoss`:

$$w_c = \frac{N}{K \cdot N_c}$$

Where:
* $N$ = Total training images ($545$)
* $K$ = Number of target classes ($3$)
* $N_c$ = Training count for class $c$

### Calculated Class Weights
* **Normal ($N_0 = 93$)**: $w_0 = 1.953$
* **Benign ($N_1 = 305$)**: $w_1 = 0.596$
* **Malignant ($N_2 = 147$)**: $w_2 = 1.236$

---

## 4. Optimization & Hyperparameters

| Hyperparameter | Configuration Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` | Decoupled weight decay provides superior generalization over standard Adam. |
| **Weight Decay** | $1 \times 10^{-2}$ | L2 regularization prevents overfitting on medical dataset splits. |
| **Initial Learning Rate** | $1 \times 10^{-4}$ | Standard transfer learning rate for classification head. |
| **Fine-Tuning LR** | $1 \times 10^{-5}$ | Conservative rate to prevent gradient destruction during fine-tuning. |
| **LR Scheduler** | `ReduceLROnPlateau` | Factor $0.5$, Patience $2$. Reduces LR when validation loss plateaus. |
| **Batch Size** | `32` (or `16` on CPU) | Balanced memory utilization and gradient estimation stability. |
| **Early Stopping** | Patience: $5$ epochs | Terminates training when validation loss stops improving. |

---

## 5. Empirical Evaluation Results (Test Set N=120)

The model was evaluated on the independent test set ($120$ unseen real ultrasound scans).

### Performance Summary Table

| Metric | Overall Score (%) |
| :--- | :---: |
| **Accuracy** | **44.17%** |
| **Macro Precision** | **41.19%** |
| **Macro Recall** | **42.69%** |
| **Macro F1-Score** | **34.92%** |
| **Weighted F1-Score** | **40.95%** |

### Per-Class Detailed Metrics

| Class Name | Support (Scans) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 21 | 0.2000 | 0.0476 | 0.0769 |
| **Benign** | 67 | 0.6857 | 0.3582 | 0.4706 |
| **Malignant** | 32 | 0.3500 | 0.8750 | 0.5000 |

### Confusion Matrix Array

```text
               Predicted Normal   Predicted Benign   Predicted Malignant
True Normal           1                  7                   13
True Benign           4                 24                   39
True Malignant        0                  4                   28
```

*(Visual plot saved to `outputs/confusion_matrix.png`).*
