# OncoVision Final Project-Readiness Audit Report

**Date:** October 9, 2026  
**Status:** Audit Complete — Ready for Review  
**Automated Test Suite Status:** 23/23 Tests Passing (100%)  

---

## Executive Summary

This report documents the final project-readiness audit for **OncoVision**, an AI-assisted breast ultrasound classification and interpretability platform. Across 9 completed model experiments, significant performance insights were gained, culminating in Experiment 9 (EfficientNet-B0 at 256×256 with horizontal-flip Test-Time Augmentation).

Per project instructions, all model optimization and retraining have been paused. No code modifications or production model swaps were performed during this audit. The current production backend continues to run the baseline model (`models/efficientnet_b0_best.pth`), preserving full experimental isolation.

---

## 1. Active Checkpoint & Inference Pipeline Inspection

### 1.1 Backend Inspection (`backend/services/model_service.py` & `backend/routes/predict.py`)
- **Active Model Checkpoint:** `models/efficientnet_b0_best.pth`
- **Architecture:** EfficientNet-B0 (3-class output: Normal, Benign, Malignant)
- **Input Resolution:** $224 \times 224$ pixels
- **Normalization:** ImageNet standard mean (`[0.485, 0.456, 0.406]`) and std (`[0.229, 0.224, 0.225]`)
- **Inference Pipeline:**
  1. Image byte stream received via REST endpoint POST `/api/predict`.
  2. Image converted to RGB format (`PIL.Image.convert("RGB")`).
  3. Preprocessing transform applied (Resize to $224 \times 224$, ToTensor, Normalize).
  4. Single-pass evaluation pass (`torch.no_grad()`).
  5. Softmax activation applied to raw logits to compute multi-class probabilities.
  6. Grad-CAM heatmaps computed targeting layer `features[7][1]` using target class activation gradient backpropagation.

### 1.2 Comparison with Leading Candidate (Experiment 9)
- **Production Active Model:** Baseline EfficientNet-B0 ($224 \times 224$), Test Accuracy: $72.50\%$, Macro F1: $69.47\%$.
- **Leading Experimental Model (Exp 9):** EfficientNet-B0 ($256 \times 256$) + Horizontal-Flip TTA, Test Accuracy: $75.83\%$, Macro F1: $73.00\%$, Malignant Recall: $68.75\%$.
- **Promotion Status:** Exp 9 candidate is stored separately at `models/efficientnet_b0_exp1_256_best.pth` and has **NOT** been swapped into production, maintaining explicit approval workflows.

---

## 2. Model Outputs & Labeling Verification

### 2.1 Backend Terminology
- Output fields from `/api/predict`:
  - `pred_class`: Predicted class string (`"Normal"`, `"Benign"`, `"Malignant"`).
  - `confidence`: Highest softmax probability (float between 0.0 and 1.0).
  - `probabilities`: Softmax probability distribution dictionary (`{"Normal": p1, "Benign": p2, "Malignant": p3}`).

### 2.2 Frontend Display Audit (`frontend/src/App.jsx`)
- **Main Confidence Callout:** Labeled as **"Model Confidence"** (`(confidence * 100).toFixed(2)%`).
- **Class Breakdown:** Labeled as **"Class Probabilities"** with individual progress bars for Normal, Benign, and Malignant.
- **Verification:** Probabilities are **NOT** mislabeled as "Accuracy", "Diagnostic Certainty", or "Biopsy-grade Diagnosis".

---

## 3. Grad-CAM Interpretability & Disclaimer Audit

### 3.1 Map Generation (`src/gradcam/localization.py`)
- Grad-CAM extracts spatial activation gradients from layer `features[7][1]`.
- Map is resized to original image dimensions, normalized $[0, 1]$, colormapped with OpenCV `COLORMAP_JET`, and blended ($40\%$ heatmap, $60\%$ original grayscale).

### 3.2 UI Presentation & Medical Framing
- Heatmap is rendered adjacent to the original image under the section **"Grad-CAM Attention Map"**.
- Accompanied by explanation banner:
  > *"Grad-CAM highlights spatial acoustic regions that contributed most significantly to the model's classification decision."*
- Prominent Footer Disclaimer:
  > *"OncoVision is a research tool for computer-aided ultrasound analysis and does not provide medical diagnosis or treatment recommendations."*
- **Verification:** Grad-CAM is explicitly framed as an attention visualization tool, not a clinical tumor boundary segmentation or diagnostic scanner.

---

## 4. Robustness & Error Handling Audit

| Scenario / Edge Case | Handled By | Expected & Verified Behavior | Status |
| :--- | :--- | :--- | :--- |
| **File Type Validation** | Frontend & Backend | Non-image upload rejected with `400 Bad Request` ("Invalid file format"). Supported: PNG, JPG, JPEG, BMP, TIF, TIFF. | PASS |
| **File Size Limits** | Backend | Files exceeding 15MB are rejected with HTTP 400. | PASS |
| **Corrupted Images** | Backend | Unreadable image bytes trigger PIL `UnidentifiedImageError` caught and returned as HTTP 400. | PASS |
| **Network / API Failures** | Frontend | `try/catch` wrapper catches fetch errors and displays error banner: *"Prediction service unavailable. Please check server status."* | PASS |
| **Loading Indicators** | Frontend | Upload button disables and displays spinning indicator with *"Analyzing image..."* text during request. | PASS |
| **Empty Results** | Frontend | Initial UI state renders clear drop zone prompt prior to upload; error state cleanly resets view. | PASS |

---

## 5. Automated Test Suite Execution Results

The master automated test suite was executed via `python scripts/run_full_test_suite.py`. All 23 unit, integration, and verification tests passed without errors.

```
======================================================================
OncoVision Master Test Suite - Audit Execution Summary
======================================================================
1. BUSI Data Splitting & Leakage Verification ................ PASS
2. EfficientNet-B0 Baseline Model Architecture .............. PASS
3. Baseline Evaluation Metrics & Matrix Integrity .......... PASS
4. Baseline Latency & Throughput Benchmark .................. PASS
5. Grad-CAM Gradient Map Generation ........................ PASS
6. FastAPI Predict API Integration & Payloads .............. PASS
7. End-to-End Synthetic Image Prediction Pipeline ............ PASS
8. Upload File Format & Corrupted Byte Validation .......... PASS
9. Model File Integrity & Parameter State Validation ........ PASS
10. Model Comparison Log & Metrics Continuity .............. PASS
11. Focal Loss Computation & Multiclass Gradient Flow ........ PASS
12. TTA Softmax Probability Vector Averaging ............... PASS
13. Experiment 9 Verification Report Integrity ............. PASS
14. Final Project-Readiness Audit Verification .............. PASS
... (Total 23 tests executed)

----------------------------------------------------------------------
Result: 23 PASSED, 0 FAILED (100% Success Rate)
Execution Time: 18.42s
----------------------------------------------------------------------
```

---

## 6. Manual End-to-End Testing Protocol

Use this manual verification protocol for qualitative QA testing across all three clinical categories.

### 6.1 Testing Checklist

| Step | Action | Expected Normal Behavior | Expected Benign Behavior | Expected Malignant Behavior |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Select test ultrasound image from `data/busi_images/` | File `normal (1).png` loaded into uploader | File `benign (1).png` loaded into uploader | File `malignant (1).png` loaded into uploader |
| **2** | Submit image for analysis | Loader spinner appears; submit button disabled | Loader spinner appears; submit button disabled | Loader spinner appears; submit button disabled |
| **3** | Inspect Class Prediction | Predicted Class: `Normal` (or high Normal probability) | Predicted Class: `Benign` (or high Benign probability) | Predicted Class: `Malignant` (or high Malignant probability) |
| **4** | Verify Confidence Display | Numeric % display (e.g., $84.20\%$) labeled **Model Confidence** | Numeric % display (e.g., $91.15\%$) labeled **Model Confidence** | Numeric % display (e.g., $79.80\%$) labeled **Model Confidence** |
| **5** | Verify Grad-CAM Map | Attention map highlights diffuse tissue / background | Attention map focuses on well-circumscribed focal lesion | Attention map focuses on irregular hypoechoic tissue regions |
| **6** | Verify Framing & Disclaimers | Interpretability banner & research disclaimer visible | Interpretability banner & research disclaimer visible | Interpretability banner & research disclaimer visible |

> **Note on Model Uncertainty:** Ultrasound interpretation involves inherent visual ambiguities. Misclassifications (e.g., a complex cyst predicted as Malignant or a subtle lesion predicted as Benign) must present valid confidence scores and Grad-CAM maps without UI crashes or invalid state assertions.

---

## 7. Audit Conclusion & Production Blockers Audit

### 7.1 Production Blockers Summary
- **Critical Blockers:** **NONE**. All automated tests pass, endpoints are resilient, and front/backend contracts are aligned.
- **Model Promotion Pending:** Experiment 9 (EfficientNet-B0 256×256 TTA) achieved superior accuracy ($75.83\%$ vs. $72.50\%$) and Macro F1 ($73.00\%$ vs. $69.47\%$). Promoting Exp 9 to production requires updating `models/efficientnet_b0_best.pth` and backend resolution parameters. Approval is pending user review.

### 7.2 Recommendations for Next Release
1. **User Approval Request:** Request approval to promote Experiment 9 to production to unlock +3.33% accuracy improvement.
2. **Clinical Validation:** Conduct external validation on independent ultrasound datasets (e.g., OASBUD) prior to clinical user trials.
