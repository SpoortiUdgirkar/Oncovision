# OncoVision - Deep Learning Architecture Benchmark Report

**Generated Date**: October 8, 2026  
**Evaluation Dataset**: BUSI Independent Test Split ($N=120$ scans)  
**Evaluated Architectures**: ResNet50, EfficientNet-B0, DenseNet121  

---

## 1. Executive Summary

This report documents the comparative empirical benchmark evaluation of three deep convolutional neural network architectures (**ResNet50**, **EfficientNet-B0**, and **DenseNet121**) for 3-class breast ultrasound scan classification (Normal, Benign, Malignant).

All models were evaluated under identical experimental conditions on the exact same test dataset ($N=120$ unseen scans) using standard ImageNet normalization, 2-stage transfer learning, and class frequency loss weighting.

### Master Model Comparison Table

| Model Architecture | Total Parameters | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | 23,514,243 | 44.17% | 41.19% | 42.69% | 34.92% | 40.95% |
| **EfficientNet-B0** | **4,011,391** | **72.50%** | **70.70%** | **68.88%** | **69.47%** | **72.43%** |
| **DenseNet121** | 6,956,931 | 40.00% | 33.05% | 33.69% | 32.01% | 39.50% |

*(Visual bar chart comparison saved to `outputs/model_comparison.png`).*

---

## 2. Experimental Setup & Preprocessing

To ensure rigorous benchmarking integrity, all three models were trained and tested under standardized hyperparameters:

* **Dataset Split**: 545 Training / 115 Validation / 120 Test ($70\% / 15\% / 15\%$).
* **Input Resolution**: $224 \times 224 \times 3$ RGB.
* **Loss Function**: `torch.nn.CrossEntropyLoss` with inverse class frequency weights ($w_0 = 1.953, w_1 = 0.596, w_2 = 1.236$).
* **Optimization Strategy**: 2-stage transfer learning using `AdamW` ($\text{weight\_decay}=10^{-2}$) and `ReduceLROnPlateau` scheduler ($\text{factor}=0.5, \text{patience}=2$).
  * *Stage 1 (Feature Extraction)*: Backbone parameters frozen; initial head trained at $\text{LR}=10^{-4}$.
  * *Stage 2 (Fine-Tuning)*: Deeper feature blocks unfrozen; fine-tuned at reduced $\text{LR}=10^{-5}$.
* **Evaluation Target**: Independent test set of 120 scans (21 Normal, 67 Benign, 32 Malignant).

---

## 3. Individual Architecture Performance Breakdown

### A. ResNet50 (Baseline Model)
* **Overall Accuracy**: **44.17%** | **Macro F1**: **34.92%** | **Weighted F1**: **40.95%**
* **Behavioral Analysis**: Suffers from heavy class bias toward Malignant predictions ($87.50\%$ recall on Malignant, but only $4.76\%$ recall on Normal and $35.82\%$ recall on Benign).
* **Confusion Matrix**:
  ```text
                 Predicted Normal   Predicted Benign   Predicted Malignant
  True Normal           1                  7                   13
  True Benign           4                 24                   39
  True Malignant        0                  4                   28
  ```

### B. EfficientNet-B0 (Compound Scaling Architecture)
* **Overall Accuracy**: **72.50%** | **Macro F1**: **69.47%** | **Weighted F1**: **72.43%**
* **Behavioral Analysis**: Demonstrates superior feature extraction efficiency with $83\%$ fewer parameters than ResNet50. Achieves well-balanced precision and recall across all three classes without collapsing onto majority predictions.
* **Confusion Matrix**:
  ```text
                 Predicted Normal   Predicted Benign   Predicted Malignant
  True Normal          12                  7                   2
  True Benign           4                 52                  11
  True Malignant        1                  8                  23
  ```

### C. DenseNet121 (Dense Feature Reuse Architecture)
* **Overall Accuracy**: **40.00%** | **Macro F1**: **32.01%** | **Weighted F1**: **39.50%**
* **Behavioral Analysis**: Exhibited early validation plateauing. High feature propagation density without sufficient depth fine-tuning led to high confusion between Benign and Normal/Malignant categories ($19$ Benign scans misclassified as Normal).
* **Confusion Matrix**:
  ```text
                 Predicted Normal   Predicted Benign   Predicted Malignant
  True Normal           7                 13                   1
  True Benign          19                 37                  11
  True Malignant       11                 17                   4
  ```

---

## 4. Per-Class Performance Comparison

| Target Class | Support | Metric | ResNet50 | EfficientNet-B0 | DenseNet121 |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Normal** | 21 | **Precision** | 20.00% | **70.59%** | 18.92% |
| | | **Recall** | 4.76% | **57.14%** | 33.33% |
| | | **F1-Score** | 7.69% | **63.16%** | 24.14% |
| **Benign** | 67 | **Precision** | 68.57% | **77.61%** | 55.22% |
| | | **Recall** | 35.82% | **77.61%** | 55.22% |
| | | **F1-Score** | 47.06% | **77.61%** | 55.22% |
| **Malignant** | 32 | **Precision** | 35.00% | **63.89%** | 25.00% |
| | | **Recall** | **87.50%** | 71.88% | 12.50% |
| | | **F1-Score** | 50.00% | **67.65%** | 16.67% |

---

## 5. Confusion Matrix Observations

1. **Class Balance Improvement**: EfficientNet-B0 correctly identified $12/21$ Normal scans ($57.14\%$), compared to only $1/21$ ($4.76\%$) for ResNet50 and $7/21$ ($33.33\%$) for DenseNet121.
2. **False Negative Reduction**: EfficientNet-B0 correctly classified $23/32$ Malignant tumors ($71.88\%$), with only 1 Malignant scan misclassified as Normal. ResNet50 correctly identified 28 Malignant scans but misclassified 52 Benign scans as Malignant ($39/67$). DenseNet121 failed severely on Malignant detection ($4/32$ correct).
3. **Overall Consistency**: EfficientNet-B0 yielded the highest true diagonal concentration ($87/120$ correct predictions) across all categories.

---

## 6. Model Selection Rationale

Using a multi-metric clinical evaluation priority order:
1. **Macro F1-Score**: **EfficientNet-B0 (69.47%)** > ResNet50 (34.92%) > DenseNet121 (32.01%).
2. **Macro Recall**: **EfficientNet-B0 (68.88%)** > ResNet50 (42.69%) > DenseNet121 (33.69%).
3. **Test Accuracy**: **EfficientNet-B0 (72.50%)** > ResNet50 (44.17%) > DenseNet121 (40.00%).
4. **Per-Class Balance**: EfficientNet-B0 achieves balanced recall ($57.14\%$ Normal, $77.61\%$ Benign, $71.88\%$ Malignant), whereas ResNet50 suffers from severe skew ($4.76\%$ Normal recall).

### Official Selection
**EfficientNet-B0** is selected as the **best-performing model on the BUSI test split**.

*(Note: This designation reflects empirical performance on the BUSI test split for research demonstration purposes and does not constitute clinical diagnostic validation).*

---

## 7. Limitations & Recommendations

* **Sample Size**: Test set contains 120 images. Evaluation on larger multi-center cohorts is recommended before deployment.
* **Hardware Execution**: CPU-bound fine-tuning required early stopping limits; further fine-tuning on GPU resources may further optimize DenseNet121 and ResNet50 performance.
