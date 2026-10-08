# OncoVision - Deep Learning Model & Transfer Learning Specification

This document details the neural network architecture, transfer learning strategy, benchmarking results, Grad-CAM explainability, and quantitative localization metrics for **OncoVision**.

---

## 1. Primary Model Architecture: EfficientNet-B0

OncoVision utilizes **EfficientNet-B0** as its primary production feature extractor, selected through empirical multi-architecture benchmarking over ResNet50 and DenseNet121.

### Network Architecture Specification

```text
Input Ultrasound Scan [3 x 224 x 224]
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ EfficientNet-B0 Feature Backbone (features[0..8])      │
│ Target Layer for Grad-CAM: model.features[-1][0]       │
│ Output Spatial Feature Map: [1280 x 7 x 7]             │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Adaptive Global Average Pooling                        │
│ Output Feature Vector: [1280]                          │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Custom Classification Head (model.classifier)          │
│  ├── Dropout (p = 0.3)                                 │
│  └── Linear (in_features = 1280, out_features = 3)     │
└────────────────────────────────────────────────────────┘
        │
        ▼
Logits Vector [3] -> Softmax -> Class Probabilities:
  - Class 0: Normal
  - Class 1: Benign
  - Class 2: Malignant
```

---

## 2. Multi-Model Benchmark Results (Test Set N=120)

All three models were evaluated on the held-out BUSI test split ($120$ unseen scans) under identical experimental conditions:

| Model Architecture | Status | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | **PRIMARY** | **72.50%** | **70.70%** | **68.88%** | **69.47%** | **72.43%** |
| **ResNet50** | Benchmark | 44.17% | 41.19% | 42.69% | 34.92% | 40.95% |
| **DenseNet121** | Benchmark | 40.00% | 33.05% | 33.69% | 32.01% | 39.50% |

### Primary Model (EfficientNet-B0) Per-Class Metrics

| Class Name | Support (Scans) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 21 | 0.7059 (70.59%) | 0.5714 (57.14%) | 0.6316 (63.16%) |
| **Benign** | 67 | 0.7761 (77.61%) | 0.7761 (77.61%) | 0.7761 (77.61%) |
| **Malignant** | 32 | 0.6389 (63.89%) | 0.7188 (71.88%) | 0.6765 (67.65%) |

---

## 3. Quantitative Grad-CAM Localization IoU

Grad-CAM attention heatmaps derived from `model.features[-1][0]` (`Conv2d(320, 1280)`) were thresholded at $\text{heatmap} \ge 0.5$ and evaluated against ground-truth BUSI lesion masks using Intersection over Union (IoU):

| Target Category | Evaluated Scans ($N$) | Mean Grad-CAM IoU (%) | Median Grad-CAM IoU |
| :--- | :---: | :---: | :---: |
| **Benign Scans** | 67 | **8.10%** ($0.0810$) | **0.0375** |
| **Malignant Scans** | 32 | **13.32%** ($0.1332$) | **0.0695** |
| **Normal Scans** | 21 | **N/A** (No lesion expected) | **N/A** |
| **Overall Lesion Scans** | **99** | **9.78%** ($0.0978$) | **0.0519** |

---

## 4. Historical Checkpoints Preserved

All benchmark checkpoints are preserved under `models/`:
* `models/efficientnet_b0_best.pth` (16.34 MB) — Primary production model
* `models/resnet50_best.pth` (94.37 MB) — Benchmark candidate
* `models/densenet121_best.pth` (28.43 MB) — Benchmark candidate
