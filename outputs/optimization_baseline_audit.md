# OncoVision - EfficientNet-B0 Baseline Training & Optimization Audit

**Date**: October 8, 2026  
**Primary Model**: EfficientNet-B0  
**Baseline Checkpoint**: [`models/efficientnet_b0_best.pth`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_best.pth)  
**Evaluation Dataset**: BUSI Held-Out Test Split ($N=120$ scans)  

---

## 1. Baseline Performance Summary

The current baseline model (**EfficientNet-B0**) achieved the top performance among all benchmarked single-model architectures on the held-out BUSI test split:

| Metric | EfficientNet-B0 Baseline Value |
| :--- | :---: |
| **Test Accuracy** | **72.50%** |
| **Macro Precision** | **70.70%** |
| **Macro Recall** | **68.88%** |
| **Macro F1-Score** | **69.47%** |
| **Weighted F1-Score** | **72.43%** |

### Class-Wise Baseline Metrics
- **Normal ($N=21$)**: Precision: 70.59% | Recall: 57.14% | F1-Score: 63.16%
- **Benign ($N=67$)**: Precision: 77.61% | Recall: 77.61% | F1-Score: 77.61%
- **Malignant ($N=32$)**: Precision: 63.89% | Recall: 71.88% | F1-Score: 67.65%

---

## 2. Current Training & Model Configuration

| Component / Pipeline Step | Current Implementation | File / Source Location |
| :--- | :--- | :--- |
| **1. Dataset & Split Logic** | BUSI dataset ($780$ total images: Normal=133, Benign=437, Malignant=210). Stratified 70/15/15 split using `SEED=42`. Train: 545, Val: 115, Test: 120. Image-level split. | [`scripts/prepare_busi_dataset.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/scripts/prepare_busi_dataset.py) |
| **2. Image Preprocessing** | $224 \times 224$ resolution (`IMAGE_SIZE = (224, 224)`), Bilinear interpolation, standard ImageNet mean (`[0.485, 0.456, 0.406]`) and std (`[0.229, 0.224, 0.225]`). | [`src/config.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/config.py), [`src/preprocessing/transforms.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/preprocessing/transforms.py) |
| **3. Data Augmentation** | `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=15)`, `ColorJitter(brightness=0.1, contrast=0.1)`. No vertical flip (preserves acoustic depth). | [`src/preprocessing/transforms.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/preprocessing/transforms.py) |
| **4. Class Imbalance Handling** | Weighted `CrossEntropyLoss` with inverse class frequencies ($\text{weight}[c] = \frac{N_{\text{total}}}{C \cdot N_c}$). Weights: Normal=1.953, Benign=0.596, Malignant=1.236. `WeightedRandomSampler` is NOT used. | [`src/training/trainer.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/training/trainer.py) |
| **5. EfficientNet-B0 Architecture** | Pretrained ImageNet weights (`EfficientNet_B0_Weights.DEFAULT`). Custom classifier head: `Dropout(p=0.3)` + `Linear(1280, 3)`. | [`src/models/efficientnet.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/models/efficientnet.py) |
| **6. Optimizer** | `AdamW` with `weight_decay=1e-2`. | [`src/training/trainer.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/training/trainer.py) |
| **7. Learning Rate** | Initial LR = `1e-4` (Stage 1). Fine-tuning LR = `1e-5` (`lr * 0.1`) for Stage 2. | [`src/config.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/config.py), [`src/training/trainer.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/training/trainer.py) |
| **8. Scheduler** | `ReduceLROnPlateau(mode="min", factor=0.5, patience=2)` monitoring validation loss. | [`src/training/trainer.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/training/trainer.py) |
| **9. Epochs** | Stage 1: 20 max epochs. Stage 2: 5 fine-tuning max epochs (25 max total). | [`src/config.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/config.py), [`src/training/trainer.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/training/trainer.py) |
| **10. Layer Freezing** | Stage 1: Entire backbone frozen (`features[0..8]`). Stage 2: Unfreezes `features.7` onwards (or all). | [`src/models/efficientnet.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/models/efficientnet.py), [`src/training/trainer.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/training/trainer.py) |
| **11. Early Stopping** | `EarlyStopping(patience=5, min_delta=1e-4)` monitoring validation loss. | [`src/training/trainer.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/training/trainer.py) |
| **12. Checkpoint Selection** | Saved best model state when validation loss (`val_loss`) achieves a new minimum. | [`src/training/trainer.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/training/trainer.py) |
| **13. Evaluation Pipeline** | Evaluated on 120 held-out test scans with `src/training/evaluator.py`. Computes classification report, confusion matrix, macro/weighted metrics. | [`src/training/evaluator.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/training/evaluator.py) |

---

## 3. Current Strengths

1. **Top Single-Model Accuracy & Macro F1**: Achieved 72.50% test accuracy and 69.47% Macro F1, outperforming ResNet50 (44.17% Acc) and DenseNet121 (40.00% Acc) by a substantial margin (+28.33% accuracy).
2. **Anatomically Sound Augmentation**: Restricts transformations to horizontal flips, small rotations ($\pm 15^\circ$), and gain/contrast adjustments, preserving acoustic shadowing and vertical tissue depth orientations.
3. **Class-Weighted Loss Function**: Uses inverse class frequency weighting in CrossEntropyLoss to prevent bias toward the majority Benign class ($N=305$ train scans).
4. **2-Stage Transfer Learning Strategy**: First trains the classification head while keeping backbone weights frozen, then fine-tunes deeper blocks at $0.1\times$ lower learning rate to prevent feature destruction.
5. **Clean Validation & Evaluation Tracking**: Uses held-out test set ($N=120$) exclusively for reporting final performance without data leakage.

---

## 4. Current Weaknesses & Bottlenecks

1. **Input Resolution Constraint ($224 \times 224$)**: EfficientNet architectures perform significantly better at higher resolutions. $224 \times 224$ limits detailed micro-calcification and lesion boundary feature extraction.
2. **Suboptimal Checkpoint Selection Metric**: Early stopping and model saving monitor `validation_loss` rather than `validation Macro F1`. On imbalanced ultrasound datasets, validation loss can decrease while class-level recall degrades.
3. **Normal Class Under-Performance**: Normal class recall is only 57.14% ($12/21$ correct), leading to misclassification of non-lesion scans as Benign.
4. **Lack of Advanced Augmentation & Regularization**: Missing techniques such as Mixup, CutMix, Label Smoothing, or Random Erasing, which reduce overfitting on small biomedical datasets ($N=545$ training samples).
5. **Fixed Cosine vs. Plateau Learning Rate Scheduling**: `ReduceLROnPlateau` relies on noisy validation loss fluctuations. Cosine Annealing with Warm Restarts often yields smoother convergence and higher peak Macro F1 scores.
6. **No Oversampling / Weighted Random Sampling**: Weighted loss penalizes mistakes on minority classes during backprop, but batches remain imbalanced during forward passes. `WeightedRandomSampler` ensures equal class representation in every batch.

---

## 5. Recommended Optimization Experiments

The following specific experiments are prioritized by their likelihood of improving the EfficientNet-B0 baseline:

### Experiment 1: Higher Input Resolution ($256 \times 256$ or $300 \times 300$)
- **Rationale**: EfficientNet-B0 native resolution is $224 \times 224$, but scaling to $256 \times 256$ or $300 \times 300$ preserves fine acoustic texture patterns and lesion boundary details.
- **Implementation**: Update `IMAGE_SIZE = (256, 256)` or `(300, 300)` in config/transforms.

### Experiment 2: Checkpoint Selection via Validation Macro F1
- **Rationale**: Saving the best checkpoint based on `val_macro_f1` instead of `val_loss` directly optimizes the primary benchmark evaluation metric and improves minority class recall (Normal & Malignant).
- **Implementation**: Modify `EarlyStopping` / checkpoint saving logic to monitor `val_macro_f1` (maximizing mode).

### Experiment 3: Label Smoothing ($0.1$) & Focal Loss
- **Rationale**: Over-confident predictions on noisy ultrasound boundaries harm generalization. Label smoothing ($0.1$) prevents over-confidence, while Focal Loss ($\gamma=2.0$) forces the model to focus on hard misclassified samples.
- **Implementation**: Replace standard weighted CrossEntropyLoss with `CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)` or Focal Loss.

### Experiment 4: Weighted Random Sampler & Advanced Augmentations
- **Rationale**: Using `WeightedRandomSampler` guarantees equal probability of sampling Normal, Benign, and Malignant scans per batch. Adding mild `RandomErasing(p=0.2)` or `AutoAugment/TrivialAugment` reduces overfitting.
- **Implementation**: Pass `sampler=WeightedRandomSampler(...)` to `DataLoader` and include `RandomErasing` in `get_train_transforms()`.

### Experiment 5: Cosine Annealing Learning Rate Scheduler with Warmup
- **Rationale**: `CosineAnnealingLR` or `CosineAnnealingWarmRestarts` provides smooth, continuous decay down to `1e-6`, avoiding premature plateaus.
- **Implementation**: Replace `ReduceLROnPlateau` with `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)`.

---

## 6. Risk Assessment Matrix

| Experiment | Expected Benefit | Risk Level | Mitigation Strategy |
| :--- | :--- | :---: | :--- |
| **Exp 1: Higher Resolution ($256 \times 256$)** | High (+2-4% F1) | Low | Slightly higher GPU/CPU memory consumption during training. |
| **Exp 2: Val Macro F1 Checkpointing** | High (+2-3% F1) | Very Low | None. Directly aligns checkpoint saving with macro evaluation goal. |
| **Exp 3: Label Smoothing ($0.1$)** | Medium (+1-2% F1) | Low | Ensure smoothing parameter is mild ($0.1$) to prevent underfitting. |
| **Exp 4: WeightedRandomSampler** | Medium (+1-3% Recall) | Low | Disable `shuffle=True` when sampler is enabled in PyTorch DataLoader. |
| **Exp 5: Cosine Annealing Scheduler** | Medium (+1-2% F1) | Low | Set `eta_min=1e-6` to avoid learning rate decaying to exact 0. |

---

## 7. Verification Status

- **Model Checkpoints**: All checkpoints (`models/efficientnet_b0_best.pth`, `resnet50_best.pth`, `densenet121_best.pth`) remain untouched and preserved.
- **Automated Tests**: Master test suite script [`scripts/run_full_test_suite.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/scripts/run_full_test_suite.py) executed cleanly with **20/20 tests passing**.
- **Working Application**: FastAPI backend and React frontend services are operating normally without breaking changes.
