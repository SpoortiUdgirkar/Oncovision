# OncoVision Backend Render Deployment Audit Report

**Date:** October 9, 2026  
**Target Platform:** Render Free Web Service (Hobby Plan)  
**Target Service:** OncoVision FastAPI REST API Backend  
**Audit Status:** Audit Complete — No Files Modified  
**Overall Deployment Verdict:** **DEPLOYABLE WITH CHANGES**  

---

## 1. Backend Structure & Startup Command

- **FastAPI Entry Point:** [`backend/main.py`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/backend/main.py)
  - Application instance: `app = FastAPI(...)`
  - Lifecycle Manager: `@asynccontextmanager async def lifespan(app: FastAPI)` loads model checkpoint into memory at startup.
- **Dependencies Configuration File:** [`requirements.txt`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/requirements.txt)
- **Documented Local Start Command:** `uvicorn backend.main:app --host 127.0.0.1 --port 8000`
- **Render Production Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

---

## 2. Dependency Audit & Package Footprint

### 2.1 Current `requirements.txt` Inspection
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
grad-cam>=1.4.8
pillow>=10.0.0
fastapi>=0.100.0
uvicorn>=0.22.0
python-multipart>=0.0.6
pydantic>=2.0.0
tqdm>=4.65.0
```

### 2.2 Dependency Analysis & Issues
1. **PyTorch PyPI Default Wheel (CUDA Heavy):**  
   Standard `pip install torch torchvision` pulls PyPI wheels with bundled NVIDIA CUDA binaries (~800 MB download, ~2.8 GB on disk). On Render's CPU-only free tier, this unnecessarily bloats build time and disk usage.
   - **Fix:** Specify PyTorch CPU-only wheels using `--extra-index-url https://download.pytorch.org/whl/cpu`.
2. **OpenCV GUI System Library Requirement:**  
   `opencv-python>=4.8.0` depends on X11/GUI libraries (`libGL.so.1`, `libglib2.0-0`). On minimal Linux container environments (like Render), importing `cv2` will throw `ImportError: libGL.so.1: cannot open shared object file`.
   - **Fix:** Replace `opencv-python` with `opencv-python-headless`.

---

## 3. Active Model Checkpoint Audit

- **Active Model Architecture:** EfficientNet-B0 ($256 \times 256$ input resolution + horizontal-flip TTA)
- **Active Checkpoint File Path:** [`models/efficientnet_b0_exp1_256_best.pth`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp1_256_best.pth)
- **Exact File Size:** **16,347,299 bytes (15.59 MB)**
- **Assessment:** The active model checkpoint is remarkably lightweight (15.59 MB) and ideal for cloud web services. However, the repository contains unused model checkpoints (e.g., `resnet50_best.pth` at 90 MB, `densenet121_best.pth` at 27 MB, totaling ~380 MB across 21 files). Excluding unused `.pth` files from git tracking or `.dockerignore` will significantly accelerate build upload times.

---

## 4. Build & Deployment Bundle Size Estimate

| Layer / Component | Standard PyPI Install | Optimized CPU Install | Notes |
| :--- | :---: | :---: | :--- |
| **Active Model (`efficientnet_b0_exp1_256_best.pth`)** | 15.59 MB | 15.59 MB | Lightweight CNN weights |
| **Unused Checkpoints in Repo** | ~364 MB | 0 MB (Excluded) | Recommend `.slugignore` or excluding from Git |
| **PyTorch + Torchvision** | ~2,800 MB | ~215 MB | CPU-only PyTorch wheel |
| **OpenCV, Scipy, Pandas, Sklearn, Matplotlib** | ~140 MB | ~120 MB | `opencv-python-headless` |
| **FastAPI, Uvicorn, Pydantic, Pillow** | ~25 MB | ~25 MB | Web server stack |
| **Total Estimated Disk Footprint** | **~3.34 GB** | **~375.59 MB** | **Render Free Disk Limit: 16 GB** |

---

## 5. Memory Requirements & RAM Footprint Audit

### 5.1 Empirically Measured Memory Footprint (Windows CPU Environment)

Memory usage was measured using `psutil` RSS across application lifecycle stages:

| Stage / Operation | Measured RSS Memory (RAM) | Margin to Render 512 MB Limit | Status |
| :--- | :---: | :---: | :---: |
| **1. Baseline Python Interpreter** | 200.37 MB | 311.63 MB | PASS |
| **2. Post-Import (PyTorch, FastAPI, OpenCV, Matplotlib)** | 271.00 MB | 241.00 MB | PASS |
| **3. Post-Model Load (`efficientnet_b0_exp1_256_best.pth`)** | 309.18 MB | 202.82 MB | PASS |
| **4. Post-Inference (256x256 + Horizontal-Flip TTA)** | 321.96 MB | 190.04 MB | PASS |
| **5. Peak RAM During Grad-CAM Generation & Matplotlib Render** | **484.60 MB** | **27.40 MB (94.6% of limit)** | **WARNING** |

### 5.2 Estimated RAM Under Production & Multi-User Conditions
- **Single Request Peak:** **484.60 MB** is under the **512 MB RAM cap** by only **27.4 MB**.
- **Concurrent Load Risk:** If two requests arrive simultaneously or Python garbage collection is delayed during OpenCV colormapping and Matplotlib plot rendering, RAM will exceed 512 MB. This will trigger Render's Linux Out-Of-Memory (OOM) manager, sending a `SIGKILL` signal to terminate the application container.

---

## 6. Runtime & Environment Compatibility

- **OS / Platform:** Linux (Debian/Ubuntu x86_64 container on Render).
- **Python Version:** Compatible with Python 3.10+.
- **Headless Graphics Backend:** `matplotlib.use("Agg")` is explicitly configured in `backend/services/gradcam_service.py`, preventing X-server GUI display errors.
- **Headless OpenCV Dependency:** Must switch `opencv-python` to `opencv-python-headless` in `requirements.txt` to avoid `libGL.so.1` missing library crashes on Render.

---

## 7. Storage & File System Audit

- **Render Storage Model:** Ephemeral file system (no persistent disk on free tier; disk resets on spin-down).
- **Model Loading:** Checkpoint [`models/efficientnet_b0_exp1_256_best.pth`](file:///c:/Users/spoor/.gemini/antigravity-ide/scratch/OncoVision/models/efficientnet_b0_exp1_256_best.pth) is loaded read-only from the app directory.
- **Inference & Grad-CAM Outputs:** Images are processed in-memory (`io.BytesIO`) and returned directly to the client as Base64 PNG data URIs. No prediction files are saved to local disk.
- **Assessment:** **PASS**. The backend is completely stateless and requires zero persistent disk storage.

---

## 8. API & Network Requirements

1. **Health Check Endpoint:** `GET /health` returns HTTP 200 (`model_loaded: true`). Ready for Render's health check probe.
2. **CORS Configuration:** `main.py` uses `allow_origins=["*"]`. (PASS for testing; recommend restriction to frontend domain in production).
3. **Payload Limits:** Maximum image upload size is capped at 15 MB in `preprocessing_service.py`. Well within Render's HTTP body limits.
4. **Cold Starts & Spin-Down:** Render free web services spin down after 15 minutes of inactivity. Initial cold start (container boot + PyTorch import + model load) takes ~10–25 seconds.
5. **Frontend API URL:** React frontend currently targets `http://localhost:8000`. Must be updated via environment variable (`VITE_API_BASE_URL`) to point to the production Render URL (e.g., `https://oncovision-api.onrender.com`).

---

## 9. Render Free-Tier Compatibility Audit Matrix

| Requirement / Criterion | Render Free Limit | OncoVision Backend Metric | Compatibility Status |
| :--- | :--- | :--- | :---: |
| **Max RAM (Memory)** | 512 MB | 309 MB startup / **484.6 MB peak (Grad-CAM)** | **WARNING** |
| **Build Disk Space** | 16 GB | ~375 MB (CPU wheels) / ~3.3 GB (Standard PyPI) | **PASS** |
| **CPU Allocation** | 0.1 CPU | EfficientNet-B0 256x256 inference (~120ms CPU) | **PASS** |
| **Inactivity Spin-Down** | 15 minutes | Handled via `@asynccontextmanager` startup | **PASS** |
| **Persistent Storage** | None (Ephemeral) | 0 KB required (Base64 in-memory response) | **PASS** |
| **Graphics Library Dependencies** | Headless Linux | `opencv-python` missing `libGL.so.1` system lib | **FAIL** |
| **PyTorch Package Size** | 500 build mins | PyPI CUDA wheel default (~2.8 GB install) | **WARNING** |
| **Health Probe** | HTTP 200 | `GET /health` operational | **PASS** |

---

## 10. Deployment Verdict & Recommendations

### **OVERALL VERDICT:** **DEPLOYABLE WITH CHANGES**

The OncoVision backend **can** be deployed on Render's free web service tier, but **requires specific configuration changes** to prevent Linux container crashes (OOM kills and missing graphics libraries).

---

### Key Blockers & Recommended Fixes

#### 1. Blocker: `opencv-python` missing `libGL.so.1` on Render Linux Container
- **Issue:** `opencv-python` requires GUI system libraries not installed on Render containers.
- **Fix:** In `requirements.txt`, replace:
  ```text
  opencv-python>=4.8.0
  ```
  with:
  ```text
  opencv-python-headless>=4.8.0
  ```

#### 2. Blocker: PyTorch CUDA Binary Bloat & Build Slowness
- **Issue:** Default `pip install torch torchvision` installs CUDA drivers (~2.8 GB), slowing builds and risking disk/time limits.
- **Fix:** Add PyTorch CPU wheel index flag at the top of `requirements.txt`:
  ```text
  --extra-index-url https://download.pytorch.org/whl/cpu
  torch>=2.0.0
  torchvision>=0.15.0
  ```

#### 3. Warning: High Peak RAM During Grad-CAM (484.60 MB / 512 MB Cap)
- **Issue:** Grad-CAM backprop + Matplotlib rendering consumes 484.6 MB RAM (94.6% of limit). Concurrent requests will trigger SIGKILL (OOM).
- **Fixes:**
  - Add explicit garbage collection (`import gc; gc.collect()`) after generating Grad-CAM figures.
  - Close Matplotlib figures aggressively (`plt.close('all')`).
  - Set `PYTHONOPTIMIZE=1` in Render Environment Variables to disable debug assertions.

#### 4. Environment Configuration for Frontend
- **Issue:** React frontend calls `http://localhost:8000`.
- **Fix:** Update frontend API calls to use `import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"`.
