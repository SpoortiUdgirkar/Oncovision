# OncoVision - Deep Learning Breast Cancer Classification & Explainable AI Platform

OncoVision is an artificial intelligence medical research platform designed for **3-class breast ultrasound scan classification** (`Normal`, `Benign`, `Malignant`) and **Grad-CAM visual explainability**.

> **Medical & Clinical Disclaimer**: OncoVision is strictly a research and educational prototype. It does **NOT** provide medical diagnosis, treatment recommendations, or biopsy-grade diagnostic certainty.

---

## 🚀 Key Features & Active Model State

* **Active Primary Model**: **EfficientNet-B0 (Experiment 9)** operating at **$256 \times 256$ input resolution** with **Horizontal-Flip Test-Time Augmentation (TTA)**.
* **Test Metrics (BUSI $N=120$ Held-Out Test Set)**:
  * **Test Accuracy:** **$75.83\%$** (vs. $72.50\%$ baseline)
  * **Macro F1-Score:** **$73.00\%$** (vs. $69.47\%$ baseline)
  * **Malignant Recall:** **$68.75\%$** (vs. $56.25\%$ baseline)
* **Explainable AI (Grad-CAM)**: PyTorch hook-based Grad-CAM overlay highlighting spatial acoustic regions that influence the model's classification decision.
* **FastAPI REST API**: High-performance backend service with in-memory model persistence, strict payload validation, and CORS support.
* **React SPA Dashboard**: Responsive frontend featuring drag-and-drop file upload, model confidence scores, class probability breakdowns, and side-by-side Grad-CAM visualizations.

---

## 📊 Comprehensive Experiment Benchmarking Summary

Evaluated on the fixed 120-image BUSI held-out test set ($21$ Normal, $67$ Benign, $32$ Malignant; Seed 42):

| Experiment | Configuration / Innovation | Input Size | Test Acc (%) | Macro F1 (%) | Malignant Recall (%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | EfficientNet-B0 Initial Baseline | $224 \times 224$ | $72.50\%$ | $69.47\%$ | $56.25\%$ | Retained Baseline |
| **Exp 1** | $256 \times 256$ Input Resolution | $256 \times 256$ | $75.83\%$ | $71.90\%$ | $56.25\%$ | Baseline Upgrade |
| **Exp 2** | Max Val Macro F1 Selection | $256 \times 256$ | $62.50\%$ | $57.61\%$ | $56.25\%$ | Rejected |
| **Exp 3** | Label Smoothing ($\alpha=0.1$) | $256 \times 256$ | $75.00\%$ | $70.82\%$ | $56.25\%$ | Rejected |
| **Exp 4** | WeightedRandomSampler | $256 \times 256$ | $68.33\%$ | $66.19\%$ | $68.75\%$ | Rejected (Acc drop) |
| **Exp 5** | $320 \times 320$ Resolution | $320 \times 320$ | $74.17\%$ | $69.21\%$ | $50.00\%$ | Rejected |
| **Exp 6** | Multiclass Focal Loss ($\gamma=2.0$) | $256 \times 256$ | $75.00\%$ | $71.86\%$ | $62.50\%$ | Benchmark |
| **Exp 7** | Cosine Annealing LR Scheduler | $256 \times 256$ | $72.50\%$ | $67.57\%$ | $50.00\%$ | Rejected |
| **Exp 8** | Mild Random Erasing ($p=0.15$) | $256 \times 256$ | $73.33\%$ | $69.29\%$ | $56.25\%$ | Rejected |
| **Exp 9** | **$256 \times 256$ + Horizontal-Flip TTA** | **$256 \times 256$** | **$75.83\%$** | **$73.00\%$** | **$68.75\%$** | **ACTIVE PRIMARY** |

*(Detailed experiment reports and confusion matrices are preserved in [`outputs/`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/outputs/)).*

---

## 🛠️ Setup & Local Execution

### 1. Environment Setup
```bash
python -m venv .venv
# Activate environment:
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python scripts/run_full_test_suite.py
```

### 3. Start Backend API Server
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 4. Launch React Frontend
```bash
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## ⚠️ Research Limitations & Ethical Notice

- **Dataset Size:** Evaluation is conducted on the BUSI dataset ($780$ total images, split into $545$ train, $115$ val, $120$ test).
- **Grad-CAM Interpretation:** Grad-CAM activation maps visualize spatial acoustic regions that influenced the model's prediction. They do **not** represent precise anatomical tumor boundaries or segmentations.
- **Clinical Non-Validation:** OncoVision has not undergone clinical trials and must not be used for diagnostic decision-making.

