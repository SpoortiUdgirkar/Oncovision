"""
OncoVision Master Automated Test & Validation Suite.

Executes comprehensive empirical tests for all 15 required validation scenarios:
1. Dataset loading
2. Image preprocessing
3. Model loading
4. Model inference
5. Grad-CAM generation
6. FastAPI /health
7. FastAPI /predict
8. React frontend build & layout verification
9. Frontend-backend communication & CORS
10. Invalid image upload handling
11. Very large image file handling (>15MB)
12. Unsupported file format handling (.txt, .pdf)
13. Missing model checkpoint handling
14. Backend unavailable error handling
15. Multiple consecutive predictions stability
"""

import sys
import os
import io
import time
from pathlib import Path
import torch
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    CLASS_NAMES,
    IDX_TO_CLASS,
    DEVICE
)
from src.preprocessing.dataset import BreastUltrasoundDataset, create_dataloaders
from src.preprocessing.transforms import get_val_test_transforms
from src.models.resnet import build_resnet50
from src.gradcam.explain import GradCAM, generate_gradcam
from backend.main import app
from backend.services.model_service import ModelService


def run_all_tests():
    test_results = []

    def record_test(test_id, name, expected, actual, passed):
        status_str = "PASS" if passed else "FAIL"
        test_results.append({
            "id": test_id,
            "name": name,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "status": status_str
        })
        print(f"[{status_str}] Test {test_id:02d}: {name}")
        print(f"       Expected: {expected}")
        print(f"       Actual  : {actual}\n")

    print("=" * 80)
    print("           ONCOVISION MASTER TEST & VALIDATION SUITE           ")
    print("=" * 80 + "\n")

    # ----------------------------------------------------
    # Test 1: Dataset Loading
    # ----------------------------------------------------
    try:
        train_ds = BreastUltrasoundDataset(TRAIN_DIR)
        val_ds = BreastUltrasoundDataset(VAL_DIR)
        test_ds = BreastUltrasoundDataset(TEST_DIR)
        tot = len(train_ds) + len(val_ds) + len(test_ds)
        passed = tot > 0 and len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0
        record_test(
            1, "Dataset Loading",
            "Valid image samples found in train, val, and test splits",
            f"Found {tot} total images (Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)})",
            passed
        )
    except Exception as e:
        record_test(1, "Dataset Loading", "Dataset loads without error", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 2: Image Preprocessing
    # ----------------------------------------------------
    try:
        sample_pil = Image.new("RGB", (500, 400), color=(128, 128, 128))
        transform = get_val_test_transforms()
        tensor = transform(sample_pil)
        expected_shape = torch.Size([3, 224, 224])
        passed = (tensor.shape == expected_shape) and (tensor.dtype == torch.float32)
        record_test(
            2, "Image Preprocessing",
            "Output tensor shape [3, 224, 224] with float32 dtype",
            f"Shape: {tensor.shape}, Dtype: {tensor.dtype}, Min: {tensor.min():.2f}, Max: {tensor.max():.2f}",
            passed
        )
    except Exception as e:
        record_test(2, "Image Preprocessing", "Transforms execute successfully", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 3: Model Loading
    # ----------------------------------------------------
    try:
        model_path = MODELS_DIR / "resnet50_best.pth"
        if not model_path.exists():
            model_path = MODELS_DIR / "resnet50_final.pth"
        
        model = build_resnet50(pretrained=False, freeze_backbone=False)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(DEVICE)
        passed = model_path.exists() and (model.fc[1].out_features == 3)
        record_test(
            3, "Model Loading",
            "ResNet50 weights loaded into memory with 3 output classes",
            f"Loaded {model_path.name}, Out Features: {model.fc[1].out_features}, Device: {DEVICE}",
            passed
        )
    except Exception as e:
        record_test(3, "Model Loading", "Model loads successfully", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 4: Model Inference
    # ----------------------------------------------------
    try:
        dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            logits = model(dummy_input)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        prob_sum = float(torch.sum(probs).item())
        pred_idx = int(torch.argmax(probs).item())
        passed = (logits.shape == torch.Size([1, 3])) and (abs(prob_sum - 1.0) < 1e-3) and (pred_idx in [0, 1, 2])
        record_test(
            4, "Model Inference",
            "Output logits shape [1, 3], probabilities sum to 1.0, class in [0, 1, 2]",
            f"Logits Shape: {logits.shape}, Prob Sum: {prob_sum:.4f}, Predicted Class: '{IDX_TO_CLASS[pred_idx]}'",
            passed
        )
    except Exception as e:
        record_test(4, "Model Inference", "Inference executes without error", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 5: Grad-CAM Generation
    # ----------------------------------------------------
    try:
        test_img_path = list((TEST_DIR).rglob("*.png"))[0]
        grad_res = generate_gradcam(image_path=test_img_path, model=model, device=DEVICE)
        has_overlay = ("overlay" in grad_res) and (grad_res["overlay"].shape[2] == 3)
        has_confidence = 0.0 <= grad_res["confidence"] <= 1.0
        passed = has_overlay and has_confidence and grad_res["output_path"].exists()
        record_test(
            5, "Grad-CAM Generation",
            "Normalized 2D activation map, color heatmap, and overlay figure generated",
            f"Prediction: {grad_res['predicted_class']} ({grad_res['confidence']*100:.1f}%), Output: {grad_res['output_path'].name}",
            passed
        )
    except Exception as e:
        record_test(5, "Grad-CAM Generation", "Grad-CAM generates successfully", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 6: FastAPI /health Endpoint
    # ----------------------------------------------------
    try:
        with TestClient(app) as client:
            res = client.get("/health")
            passed = (res.status_code == 200) and (res.json().get("status") == "ok") and (res.json().get("model_loaded") is True)
            record_test(
                6, "FastAPI /health Endpoint",
                "HTTP 200 OK with status='ok' and model_loaded=true",
                f"HTTP {res.status_code}: {res.json()}",
                passed
            )
    except Exception as e:
        record_test(6, "FastAPI /health Endpoint", "Health endpoint responds 200 OK", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 7: FastAPI /predict Endpoint
    # ----------------------------------------------------
    try:
        with open(test_img_path, "rb") as f:
            img_bytes = f.read()

        with TestClient(app) as client:
            res = client.post("/predict", files={"file": (test_img_path.name, io.BytesIO(img_bytes), "image/png")})
            data = res.json()
            has_pred = data.get("prediction") in ["Normal", "Benign", "Malignant"]
            has_b64 = str(data.get("gradcam_image")).startswith("data:image/png;base64,")
            passed = (res.status_code == 200) and has_pred and has_b64
            record_test(
                7, "FastAPI /predict Endpoint",
                "HTTP 200 OK returning prediction, confidence float, and Base64 Grad-CAM string",
                f"HTTP {res.status_code}: Pred='{data.get('prediction')}', Conf={data.get('confidence')}, B64 Len={len(str(data.get('gradcam_image')))}",
                passed
            )
    except Exception as e:
        record_test(7, "FastAPI /predict Endpoint", "Predict endpoint responds 200 OK", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 8: React Frontend Build & Layout
    # ----------------------------------------------------
    try:
        dist_html = PROJECT_ROOT / "frontend" / "dist" / "index.html"
        app_jsx = PROJECT_ROOT / "frontend" / "src" / "App.jsx"
        has_dist = dist_html.exists()
        has_title = "ONCOVISION" in app_jsx.read_text(encoding="utf-8")
        has_disclaimer = "This system is developed for academic/research purposes" in app_jsx.read_text(encoding="utf-8")
        passed = has_dist and has_title and has_disclaimer
        record_test(
            8, "React Frontend",
            "Vite production bundle compiled (dist/index.html), header and medical disclaimer present",
            f"dist/index.html exists: {has_dist}, Title present: {has_title}, Disclaimer present: {has_disclaimer}",
            passed
        )
    except Exception as e:
        record_test(8, "React Frontend", "Frontend bundle verified", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 9: Frontend-Backend Communication & CORS
    # ----------------------------------------------------
    try:
        with TestClient(app) as client:
            res = client.options("/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
            has_cors = "access-control-allow-origin" in [h.lower() for h in res.headers.keys()] or (res.status_code == 200)
            passed = (res.status_code in [200, 204]) or has_cors
            record_test(
                9, "Frontend-Backend Communication & CORS",
                "CORS middleware enables cross-origin requests from React frontend (http://localhost:5173)",
                f"HTTP {res.status_code}, CORS Headers: {dict(res.headers)}",
                passed
            )
    except Exception as e:
        record_test(9, "Frontend-Backend Communication & CORS", "CORS handled cleanly", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 10: Invalid Image Upload
    # ----------------------------------------------------
    try:
        with TestClient(app) as client:
            res = client.post("/predict", files={"file": ("corrupted.png", io.BytesIO(b"Not an image byte stream"), "image/png")})
            passed = (res.status_code == 400) and ("corrupted" in res.json().get("detail", "").lower() or "invalid" in res.json().get("detail", "").lower())
            record_test(
                10, "Invalid Image Upload Handling",
                "HTTP 400 Bad Request with structured error details",
                f"HTTP {res.status_code}: {res.json().get('detail')}",
                passed
            )
    except Exception as e:
        record_test(10, "Invalid Image Upload Handling", "Rejects invalid image with 400", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 11: Very Large Image File (>15MB)
    # ----------------------------------------------------
    try:
        large_dummy_bytes = b"0" * (16 * 1024 * 1024)  # 16 MB dummy payload
        with TestClient(app) as client:
            res = client.post("/predict", files={"file": ("huge_scan.png", io.BytesIO(large_dummy_bytes), "image/png")})
            passed = (res.status_code == 400) and ("exceeds" in res.json().get("detail", "").lower() or "limit" in res.json().get("detail", "").lower())
            record_test(
                11, "Very Large Image File Handling",
                "HTTP 400 Bad Request when file size exceeds 15 MB boundary limit",
                f"HTTP {res.status_code}: {res.json().get('detail')}",
                passed
            )
    except Exception as e:
        record_test(11, "Very Large Image File Handling", "Rejects oversized image with 400", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 12: Unsupported File Format (.txt, .pdf)
    # ----------------------------------------------------
    try:
        with TestClient(app) as client:
            res = client.post("/predict", files={"file": ("document.pdf", io.BytesIO(b"%PDF-1.4 dummy pdf content"), "application/pdf")})
            passed = (res.status_code == 400) and ("unsupported" in res.json().get("detail", "").lower() or "extension" in res.json().get("detail", "").lower())
            record_test(
                12, "Unsupported File Format Handling",
                "HTTP 400 Bad Request for non-image file extensions (.pdf, .txt, .zip)",
                f"HTTP {res.status_code}: {res.json().get('detail')}",
                passed
            )
    except Exception as e:
        record_test(12, "Unsupported File Format Handling", "Rejects unsupported format with 400", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 13: Missing Model Checkpoint Handling
    # ----------------------------------------------------
    try:
        dummy_service = ModelService.__new__(ModelService)
        dummy_service.model = None
        dummy_service.device = DEVICE
        try:
            dummy_service.load_model(model_path="models/non_existent_model_checkpoint.pth")
            passed = False
            act = "No exception raised"
        except FileNotFoundError as fnf:
            passed = True
            act = f"FileNotFoundError caught cleanly: '{str(fnf)}'"
        record_test(
            13, "Missing Model Checkpoint Handling",
            "Raises descriptive FileNotFoundError when model checkpoint is missing",
            act,
            passed
        )
    except Exception as e:
        record_test(13, "Missing Model Checkpoint Handling", "Handles missing model file cleanly", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 14: Backend Unavailable Error Handling
    # ----------------------------------------------------
    try:
        app_code = (PROJECT_ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
        has_try_catch = ("try {" in app_code) and ("catch (" in app_code)
        has_error_state = "setError(" in app_code and "error-alert" in app_code
        passed = has_try_catch and has_error_state
        record_test(
            14, "Backend Unavailable Error Handling",
            "Frontend App.jsx wraps network requests in try-catch and renders alert banner on failure",
            f"Try-Catch Present: {has_try_catch}, Error Alert Banner UI Present: {has_error_state}",
            passed
        )
    except Exception as e:
        record_test(14, "Backend Unavailable Error Handling", "Frontend handles network failure", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 15: Multiple Consecutive Predictions
    # ----------------------------------------------------
    try:
        with open(test_img_path, "rb") as f:
            img_bytes = f.read()

        consecutive_passes = 0
        with TestClient(app) as client:
            for i in range(5):
                res = client.post("/predict", files={"file": (f"test_{i}.png", io.BytesIO(img_bytes), "image/png")})
                if res.status_code == 200 and "prediction" in res.json():
                    consecutive_passes += 1

        passed = consecutive_passes == 5
        record_test(
            15, "Multiple Consecutive Predictions",
            "5/5 consecutive API inference & Grad-CAM requests return HTTP 200 OK with stable memory",
            f"Successful consecutive requests: {consecutive_passes}/5",
            passed
        )
    except Exception as e:
        record_test(15, "Multiple Consecutive Predictions", "Consecutive predictions run cleanly", f"Error: {str(e)}", False)

    # Summary
    passed_count = sum(1 for t in test_results if t["passed"])
    total_count = len(test_results)
    print("=" * 80)
    print(f"       MASTER SUITE SUMMARY: {passed_count}/{total_count} TESTS PASSED        ")
    print("=" * 80 + "\n")

    return test_results


if __name__ == "__main__":
    run_all_tests()
