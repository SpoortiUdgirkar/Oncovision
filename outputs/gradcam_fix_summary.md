# EfficientNet-B0 Grad-CAM Fix & Verification Summary

**Date**: October 8, 2026  
**Project**: OncoVision  

---

## 1. Files Changed

1. [`src/gradcam/explain.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/gradcam/explain.py) — Updated `_detect_target_layer` to select `model.features[-1][0]` (`Conv2d(320, 1280)`) for EfficientNet-B0 and print the selected target layer.
2. [`src/gradcam/localization.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/gradcam/localization.py) — Refactored image evaluation loop to generate Grad-CAM only once per image, eliminating redundant computation and aligning visualization overlays with IoU thresholding.
3. [`src/preprocessing/mask_loader.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/src/preprocessing/mask_loader.py) — Replaced hardcoded personal Windows path `C:\Users\spoor\Downloads\archive.zip` with dynamic project-configurable resolution (`BUSI_MASK_ARCHIVE` env var, `DATASET_DIR / "archive.zip"`, or `Path.home() / "Downloads" / "archive.zip"`).
4. [`outputs/gradcam_localization_report.md`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/outputs/gradcam_localization_report.md) — Updated target layer specification, mask loading notes, and quantitative IoU metrics.
5. [`README.md`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/README.md) & [`MODEL_DETAILS.md`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/MODEL_DETAILS.md) — Documented primary EfficientNet-B0 model selection, target layer `model.features[-1][0]`, dynamic dataset path configuration, and quantitative localization IoU metrics.
6. [`scripts/run_full_test_suite.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/scripts/run_full_test_suite.py) — Updated target layer expectations and assertions.

---

## 2. Summary of Fixes

- **Target Layer Correction**: EfficientNet-B0 Grad-CAM now targets `model.features[-1][0]` (`Conv2d(320, 1280)`), targeting the final convolutional layer directly. ResNet50 (`model.layer4[-1]`) and DenseNet121 (`model.features.denseblock4`) target layers remain unchanged.
- **Redundant Computation Removal**: Replaced duplicate `generate_gradcam` and `gradcam_engine.generate_map` calls with a single-pass `gradcam_engine.generate_map` per test image.
- **Hardcoded Path Elimination**: Removed personal Windows directory references from `src/preprocessing/mask_loader.py`. Archive location is dynamically resolved without hardcoded user strings.
- **Robust Mask Loader**: Preserved stem matching, multi-lesion bitwise OR merging, binary `{0, 255}` output, and nearest-neighbor resizing.

---

## 3. Selected Target Layers

- **EfficientNet-B0**: `model.features[-1][0]` (`Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)`)
- **ResNet50**: `model.layer4[-1]` (`Bottleneck`)
- **DenseNet121**: `model.features.denseblock4` (`_DenseBlock`)

---

## 4. Quantitative Localization IoU (Fixed $\text{heatmap} \ge 0.5$)

- **Did IoU Value Change?**: YES. Updating the target layer directly to `model.features[-1][0]` and eliminating double heatmap generation refined feature activation maps.
- **Benign Scans ($N=67$)**: Mean IoU = **8.10%** ($0.0810$), Median IoU = **0.0375**
- **Malignant Scans ($N=32$)**: Mean IoU = **13.32%** ($0.1332$), Median IoU = **0.0695**
- **Normal Scans ($N=21$)**: **N/A** (no ground-truth lesion expected)
- **Overall Lesion Scans ($N=99$)**: Mean IoU = **9.78%** ($0.0978$), Median IoU = **0.0519**

---

## 5. Test Suite Verification

- **Total Automated Tests**: 20
- **Passed**: 20/20 (100% pass rate)

---

## 6. Remaining Limitations & Clinical Disclaimer

- **Explainability vs. Clinical Diagnostic Accuracy**: Grad-CAM localization IoU measures visual feature alignment with BUSI lesion masks. It does **not** prove clinical diagnostic accuracy, safety, or causal reasoning.
- **Coarse Feature Receptive Field**: Standard CNN feature extractors downsample spatial resolution by $32\times$ ($224 \times 224 \rightarrow 7 \times 7$), producing diffuse saliency regions rather than crisp anatomical lesion boundaries.
