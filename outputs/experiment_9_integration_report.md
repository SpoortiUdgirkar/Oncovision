# OncoVision Experiment 9 Integration Report

**Date:** October 9, 2026  
**Status:** Successfully Integrated & Verified  
**Active Model:** EfficientNet-B0 ($256 \times 256$ + Horizontal-Flip TTA)  
**Checkpoint Path:** `models/efficientnet_b0_exp1_256_best.pth`  
**Master Automated Test Suite:** 23/23 Tests Passing (100%)  

---

## Executive Summary

This report documents the controlled integration of **Experiment 9** (EfficientNet-B0 at $256 \times 256$ resolution with horizontal-flip Test-Time Augmentation) as the primary active research model for the OncoVision application.

Experiment 9 achieved superior classification performance on the fixed 120-image BUSI test set:
- **Test Accuracy:** $75.83\%$ (vs. $72.50\%$ baseline)
- **Macro F1-Score:** $73.00\%$ (vs. $69.47\%$ baseline)
- **Malignant Recall:** $68.75\%$ (vs. $56.25\%$ baseline)

All backend preprocessing pipelines, inference routines, Grad-CAM interpretability hooks, and automated tests have been updated, verified, and benchmarked without retraining model weights or altering dataset splits.

---

## 1. Summary of System Modifications

### 1.1 Preprocessing & Resolution Configuration (`src/config.py`)
- Updated `IMAGE_SIZE` default from `(224, 224)` to `(256, 256)`.
- Applied standard ImageNet RGB normalization parameters (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### 1.2 Model Service Singleton (`backend/services/model_service.py`)
- **Default Checkpoint:** Updated primary checkpoint path to `models/efficientnet_b0_exp1_256_best.pth`.
- **Test-Time Augmentation (TTA) Routine:**
  1. Forward pass on original preprocessed tensor $[1, 3, 256, 256] \rightarrow p_{\text{orig}} = \text{softmax}(\text{logits}_{\text{orig}})$.
  2. Forward pass on horizontally flipped tensor $\text{flip}(x, \text{dim}=3) \rightarrow p_{\text{flip}} = \text{softmax}(\text{logits}_{\text{flip}})$.
  3. Softmax probability vector averaging: $p_{\text{TTA}} = \frac{p_{\text{orig}} + p_{\text{flip}}}{2.0}$.
  4. Class prediction: $\text{argmax}(p_{\text{TTA}})$ mapping to `CLASS_NAMES` (`["Normal", "Benign", "Malignant"]`).
- **Rollback Helper:** Implemented `rollback_to_baseline()` method to restore the $224 \times 224$ baseline checkpoint (`models/efficientnet_b0_best.pth`) seamlessly.

### 1.3 Grad-CAM Interpretability Service (`backend/services/gradcam_service.py`)
- Updated `generate_gradcam_base64` signature to accept `target_class` index derived from TTA probability averaging.
- **Explainability Scope Documentation:** Grad-CAM operates on the original (un-flipped) image forward pass backpropagating gradients for the TTA-selected target class. Heatmaps highlight influential acoustic features, **not** precise anatomical tumor segmentation boundaries.

### 1.4 Prediction Router Endpoint (`backend/routes/predict.py`)
- Accepts uploaded ultrasound scans and applies $256 \times 256$ deterministic validation transforms.
- Calls `model_service.predict(img_tensor, use_tta=True)`.
- Passes the TTA target class index to `generate_gradcam_base64`.
- Returns metadata `model_name="EfficientNet-B0 (256x256 + TTA)"` alongside confidence scores, class probabilities, and medical disclaimers.

---

## 2. File Change Audit

| File | Change Description |
| :--- | :--- |
| [`src/config.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/config.py) | Updated default `IMAGE_SIZE` to `(256, 256)`. |
| [`backend/services/model_service.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/backend/services/model_service.py) | Updated default model loading to Exp 9 checkpoint, implemented TTA inference logic, added `rollback_to_baseline()`. |
| [`backend/services/gradcam_service.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/backend/services/gradcam_service.py) | Updated `generate_gradcam_base64` to support `target_class` index parameter. |
| [`backend/routes/predict.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/backend/routes/predict.py) | Integrated $256 \times 256$ TTA inference and explicit target class passing for Grad-CAM overlay generation. |
| [`scripts/run_full_test_suite.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/scripts/run_full_test_suite.py) | Updated Test 2 & Test 3 to validate $256 \times 256$ tensor dimensions, Exp 9 checkpoint loading, and baseline rollback state. |

---

## 3. Automated Test Suite Results

The master test suite was re-executed following Experiment 9 integration. All 23 tests passed cleanly.

```
======================================================================
OncoVision Master Test Suite Execution
======================================================================
[PASS] Test 01: Dataset Loading & Structure
[PASS] Test 02: Image Preprocessing (256x256 Tensor Shape)
[PASS] Test 03: Primary Model Checkpoint Loading & Rollback State
[PASS] Test 04: Model Inference & Class Probabilities
[PASS] Test 05: Grad-CAM Generation & Target Layer Auto-Detection
[PASS] Test 06: FastAPI /health Endpoint
[PASS] Test 07: FastAPI /predict Endpoint & Model Metadata
[PASS] Test 08: React Frontend Build & Layout
[PASS] Test 09: Frontend-Backend Communication & CORS
[PASS] Test 10: Invalid Image Upload Handling
[PASS] Test 11: Very Large Image File Handling (>15 MB)
[PASS] Test 12: Unsupported File Format Handling
[PASS] Test 13: Missing Model Checkpoint Handling
[PASS] Test 14: Backend Unavailable Error Handling
[PASS] Test 15: Multiple Consecutive Predictions (5/5)
[PASS] Test 16: Heatmap Non-NaN & Non-Empty Validation
[PASS] Test 17: BUSI Mask Loading & Manifest Matching
[PASS] Test 18: Binary Mask Preprocessing
[PASS] Test 19: Grad-CAM Localization IoU Calculation
[PASS] Test 20: Normal-Class N/A Handling
[PASS] Test 21: Focal Loss Unit & Mathematical Verification
[PASS] Test 22: Random Erasing Transform Verification
[PASS] Test 23: TTA Probability Averaging & Determinism

----------------------------------------------------------------------
Result: 23 PASSED, 0 FAILED (100% Success Rate)
----------------------------------------------------------------------
```

---

## 4. End-to-End Sample Output Verification

Live backend API predictions (`POST /api/predict`) were compared with standalone Experiment 9 evaluation across sample test scans:

| Sample Scan | True Class | Model Prediction | Model Confidence | Softmax Probabilities (Normal, Benign, Malignant) | Match Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `normal (101).png` | Normal | **Normal** | $48.33\%$ | $[0.4833, 0.3245, 0.1923]$ | EXACT MATCH |
| `benign (1).png` | Benign | **Benign** | $42.69\%$ | $[0.2382, 0.4269, 0.3350]$ | EXACT MATCH |
| `malignant (104).png` | Malignant | **Malignant** | $55.48\%$ | $[0.2141, 0.2311, 0.5548]$ | EXACT MATCH |

---

## 5. Rollback Protocol

The original baseline model (`models/efficientnet_b0_best.pth`) has been preserved untouched on disk.

To roll back the active backend to the $224 \times 224$ baseline model:
1. In Python / Service script:
   ```python
   from backend.services.model_service import model_service
   model_service.rollback_to_baseline()
   ```
2. Or in `src/config.py`:
   Set `IMAGE_SIZE = (224, 224)` and restart the FastAPI backend server.

---

## 6. Medical Framing & Disclaimer Notice

All API responses and UI panels maintain strict medical research disclaimers:
> *"OncoVision is a research tool for computer-aided ultrasound analysis and does not provide medical diagnosis or treatment recommendations."*

Model confidence scores are explicitly labeled as **"Model Confidence"** and **"Class Probabilities"**, avoiding misleading diagnostic claims or assertions of biopsy-level accuracy.
