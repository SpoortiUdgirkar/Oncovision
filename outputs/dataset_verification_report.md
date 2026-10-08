# OncoVision - Dataset Verification & Integrity Report

**Generated Date**: October 8, 2026  
**Dataset Name**: BUSI (Dataset of Breast Ultrasound Images)  
**Dataset Purpose**: Deep learning classification of B-mode breast ultrasound scans into three diagnostic categories: Normal, Benign, and Malignant.

---

## 1. Executive Summary

| Attribute | Quantitative Value / Status |
| :--- | :--- |
| **Dataset Source** | BUSI (Breast Ultrasound Dataset, Al-Dhabyani et al., 2020) |
| **Total Valid Images** | **780** scans (excluding binary segmentation masks) |
| **Target Classes** | 3 (`Normal`, `Benign`, `Malignant`) |
| **Train / Val / Test Split** | **545** Train (69.9%) / **115** Val (14.7%) / **120** Test (15.4%) |
| **File Format Integrity** | **0** corrupted or unreadable images found |
| **Exact Duplicate Files** | **0** MD5 hash collisions between splits (Passed) |
| **Class Imbalance Ratio** | **3.29 : 1** (Benign to Normal ratio) |
| **Augmentation Status** | Implemented (Horizontal Flip, $\pm 15^\circ$ Rotation, Color Jitter) |

---

## 2. Split & Class Distribution

The dataset was partitioned using a **70% / 15% / 15%** stratified split strategy with a fixed random seed (`SEED=42`).

### Class Breakdown Across Splits

| Class Name | Train (70%) | Validation (15%) | Test (15%) | Total Scans | Overall Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 93 | 19 | 21 | **133** | 17.05% |
| **Benign** | 305 | 65 | 67 | **437** | 56.03% |
| **Malignant** | 147 | 31 | 32 | **210** | 26.92% |
| **Total** | **545** | **115** | **120** | **780** | **100.00%** |

---

## 3. Data Integrity & Leakage Assessment

### A. Corrupted & Unsupported File Check
* All 780 files were verified via PIL header inspection (`Image.verify()`).
* **Result**: 0 corrupted image files, 0 unreadable files, and 0 unsupported file extensions detected.

### B. MD5 Hash Duplicate Verification
* To prevent data leakage, MD5 checksums were computed for every image file across `dataset/train`, `dataset/val`, and `dataset/test`.
* **Result**: **0 duplicate file hashes** were found between splits (`Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`).

### C. Patient-Level Leakage Assessment
* **Metadata Availability**: Filenames in the BUSI release follow standard numbered patterns (`benign (1).png`, `benign (2).png`, `malignant (1).png`). Public metadata lacks explicit patient ID tags (e.g. `patient_001_scan_1.png`).
* **Leakage Risk**: While no identical image files exist across splits, image-level splitting could theoretically allow different scan views from the same patient to reside in different splits if a patient contributed multiple scans.
* **Policy Compliance**: Per project guidelines, synthetic patient IDs were **NOT** fabricated. The existing stratified split is preserved, and the potential for patient-level overlap is transparently acknowledged as a dataset limitation.

---

## 4. Preprocessing & Data Augmentation Pipeline

### Preprocessing Pipeline
1. **Format Conversion**: All images converted to 3-channel RGB (`Image.convert("RGB")`).
2. **Resizing**: Standardized to $224 \times 224$ pixels using bilinear interpolation.
3. **Normalization**: Standard ImageNet mean $\mu = [0.485, 0.456, 0.406]$ and standard deviation $\sigma = [0.229, 0.224, 0.225]$.

### Training Data Augmentations
* **Random Horizontal Flip ($p=0.5$)**: Safe for ultrasound imaging as lateral transducer orientation does not alter acoustic shadowing or margin spiculation.
* **Random Rotation ($\pm 15^\circ$)**: Simulates realistic transducer positioning angles during clinical examination.
* **Color Jitter (Brightness 0.1, Contrast 0.1)**: Simulates minor gain and dynamic range setting variations across ultrasound scanners.
* **Excluded Augmentations**: Vertical flips (violates anatomical acoustic depth from skin line to chest wall) and extreme elastic warping (distorts tumor boundary geometry).

---

## 5. Mask Correspondence & Explainability Evaluation Potential

* **Mask Exclusion**: Binary segmentation masks (`*_mask.png`) were filtered out of training/testing classification directories to prevent model training on binary mask images.
* **Ground-Truth Correspondence**: Ground-truth lesion segmentation masks exist in the raw BUSI source directory with 1:1 filename correspondence (e.g., `benign (100).png` $\leftrightarrow$ `benign (100)_mask.png`).
* **Future Localization Evaluation**: This verified correspondence allows for future quantitative Grad-CAM localization evaluation (such as Intersection over Union [IoU] or Pointing Game score) once ground-truth mask loading is integrated into evaluation scripts.

---

## 6. Dataset Limitations & Mitigation Strategies

1. **Class Imbalance**: Benign scans represent $56.03\%$ of the dataset, while Normal scans represent only $17.05\%$.
   * *Mitigation*: Inverse class frequency weighting ($w_0 = 1.953, w_1 = 0.596, w_2 = 1.236$) injected into `torch.nn.CrossEntropyLoss`.
2. **Dataset Size**: Total count of 780 images is modest for training deep convolutional neural networks from scratch.
   * *Mitigation*: Transfer learning with ImageNet pretrained backbones and 2-stage fine-tuning.
