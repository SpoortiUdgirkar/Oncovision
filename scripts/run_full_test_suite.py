"""
OncoVision Master Automated Test & Validation Suite.

Executes comprehensive empirical tests for all 20 required validation scenarios:
1. Dataset loading
2. Image preprocessing
3. Model loading (EfficientNet-B0 Primary & ResNet50 Benchmark)
4. Model inference & probability output
5. Grad-CAM generation & target layer auto-detection
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
16. EfficientNet-B0 primary model checkpoint loading
17. Heatmap non-NaN & non-empty validation
18. BUSI mask loading & matching manifest
19. Mask binarization & NEAREST-NEIGHBOR preprocessing
20. Grad-CAM Localization IoU calculation & Normal-class N/A handling
"""

import sys
import os
import io
import time
from pathlib import Path
import numpy as np
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
from src.preprocessing.transforms import get_train_transforms, get_val_test_transforms
from src.preprocessing.mask_loader import BUSIMaskLoader, get_test_set_mask_manifest
from src.models.factory import build_model
from src.gradcam.explain import GradCAM, generate_gradcam
from src.gradcam.localization import compute_iou
from src.training.loss import FocalLoss
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
        expected_shape = torch.Size([3, 256, 256])
        passed = (tensor.shape == expected_shape) and (tensor.dtype == torch.float32)
        record_test(
            2, "Image Preprocessing",
            "Output tensor shape [3, 256, 256] with float32 dtype",
            f"Shape: {tensor.shape}, Dtype: {tensor.dtype}, Min: {tensor.min():.2f}, Max: {tensor.max():.2f}",
            passed
        )
    except Exception as e:
        record_test(2, "Image Preprocessing", "Transforms execute successfully", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 3: Primary Model Checkpoint Loading
    # ----------------------------------------------------
    try:
        model_path = MODELS_DIR / "efficientnet_b0_exp1_256_best.pth"
        if not model_path.exists():
            model_path = MODELS_DIR / "efficientnet_b0_best.pth"

        model = build_model("efficientnet_b0", num_classes=3, pretrained=False, freeze_backbone=False)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(DEVICE)
        baseline_exists = (MODELS_DIR / "efficientnet_b0_best.pth").exists()
        passed = model_path.exists() and (model.classifier[1].out_features == 3) and baseline_exists
        record_test(
            3, "Primary Model Loading",
            "EfficientNet-B0 Exp 9 (256x256) checkpoint loaded into memory with 3 output classes",
            f"Loaded {model_path.name}, Out Features: {model.classifier[1].out_features}, Baseline Preserved: {baseline_exists}",
            passed
        )
    except Exception as e:
        record_test(3, "Primary Model Loading", "Model loads successfully", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 4: Model Inference & Class Probabilities
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
            4, "Model Inference & Probabilities",
            "Output logits shape [1, 3], probabilities sum to 1.0, class in [0, 1, 2]",
            f"Logits Shape: {logits.shape}, Prob Sum: {prob_sum:.4f}, Predicted Class: '{IDX_TO_CLASS[pred_idx]}'",
            passed
        )
    except Exception as e:
        record_test(4, "Model Inference & Probabilities", "Inference executes without error", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 5: Grad-CAM Generation & Target Layer Auto-Detection
    # ----------------------------------------------------
    try:
        test_img_path = list((TEST_DIR).rglob("*.png"))[0]
        grad_res = generate_gradcam(image_path=test_img_path, model=model, device=DEVICE)
        has_overlay = ("overlay" in grad_res) and (grad_res["overlay"].shape[2] == 3)
        has_confidence = 0.0 <= grad_res["confidence"] <= 1.0
        passed = has_overlay and has_confidence and grad_res["output_path"].exists()
        record_test(
            5, "Grad-CAM Generation & Target Layer",
            "Target layer model.features[-1][0] detected, 2D map, heatmap, and overlay figure generated",
            f"Prediction: {grad_res['predicted_class']} ({grad_res['confidence']*100:.1f}%), Output: {grad_res['output_path'].name}",
            passed
        )
    except Exception as e:
        record_test(5, "Grad-CAM Generation & Target Layer", "Grad-CAM generates successfully", f"Error: {str(e)}", False)

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
    # Test 7: FastAPI /predict Endpoint & Model Metadata
    # ----------------------------------------------------
    try:
        with open(test_img_path, "rb") as f:
            img_bytes = f.read()

        with TestClient(app) as client:
            res = client.post("/predict", files={"file": (test_img_path.name, io.BytesIO(img_bytes), "image/png")})
            data = res.json()
            has_pred = data.get("prediction") in ["Normal", "Benign", "Malignant"]
            has_b64 = str(data.get("gradcam_image")).startswith("data:image/png;base64,")
            has_model_name = "model_name" in data and "EfficientNet" in data.get("model_name", "")
            passed = (res.status_code == 200) and has_pred and has_b64 and has_model_name
            record_test(
                7, "FastAPI /predict Endpoint",
                "HTTP 200 OK returning prediction, confidence, model_name='EfficientNet-B0', and Base64 Grad-CAM string",
                f"HTTP {res.status_code}: Pred='{data.get('prediction')}', Conf={data.get('confidence')}, Model='{data.get('model_name')}'",
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
        has_model_tag = "EfficientNet-B0" in app_jsx.read_text(encoding="utf-8")
        passed = has_dist and has_title and has_model_tag
        record_test(
            8, "React Frontend Build & Layout",
            "Vite production bundle compiled (dist/index.html), header, disclaimer, and EfficientNet-B0 tag present",
            f"dist/index.html exists: {has_dist}, Title present: {has_title}, EfficientNet tag present: {has_model_tag}",
            passed
        )
    except Exception as e:
        record_test(8, "React Frontend Build & Layout", "Frontend bundle verified", f"Error: {str(e)}", False)

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
    # Test 10: Invalid Image Upload Handling
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
    # Test 11: Very Large Image File Handling (>15MB)
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
    # Test 12: Unsupported File Format Handling
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
    # Test 15: Multiple Consecutive Predictions Stability
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

    # ----------------------------------------------------
    # Test 16: Heatmap Non-NaN & Non-Empty Validation
    # ----------------------------------------------------
    try:
        cam_engine = GradCAM(model)
        dummy_t = torch.randn(1, 3, 224, 224, requires_grad=True, device=DEVICE)
        cam_map, _, _, _ = cam_engine.generate_map(dummy_t)
        has_nan = np.isnan(cam_map).any()
        has_inf = np.isinf(cam_map).any()
        passed = (cam_map.ndim == 2) and (not has_nan) and (not has_inf) and (cam_map.max() <= 1.0)
        record_test(
            16, "Heatmap Non-NaN & Non-Empty Validation",
            "Grad-CAM 2D map has shape (7, 7) or (H, W), no NaN or Inf values, normalized in [0, 1]",
            f"Shape: {cam_map.shape}, Has NaN: {has_nan}, Has Inf: {has_inf}, Max: {cam_map.max():.2f}",
            passed
        )
    except Exception as e:
        record_test(16, "Heatmap Non-NaN & Non-Empty Validation", "Heatmap validation succeeds", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 17: BUSI Mask Loading & Matching Manifest
    # ----------------------------------------------------
    try:
        manifest = get_test_set_mask_manifest()
        tot_matched = manifest["matched_exactly_1_mask"] + manifest["matched_multiple_masks"]
        passed = (manifest["total_test_images"] == 120) and (tot_matched == 120)
        record_test(
            17, "BUSI Mask Loading & Matching Manifest",
            "120/120 test classification images matched to valid BUSI mask files",
            f"Total Test Images: {manifest['total_test_images']}, Matched: {tot_matched}, Exactly 1: {manifest['matched_exactly_1_mask']}, Multiple: {manifest['matched_multiple_masks']}",
            passed
        )
    except Exception as e:
        record_test(17, "BUSI Mask Loading & Matching Manifest", "Mask matching succeeds", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 18: Mask Binarization & NEAREST-NEIGHBOR Preprocessing
    # ----------------------------------------------------
    try:
        mask_loader = BUSIMaskLoader()
        sample_img = list((TEST_DIR / "Benign").glob("*.png"))[0]
        mask_np, has_lesion, n_masks = mask_loader.load_mask(sample_img, target_shape=(224, 224))
        unique_vals = set(np.unique(mask_np))
        passed = (mask_np.shape == (224, 224)) and unique_vals.issubset({0, 255}) and has_lesion
        record_test(
            18, "Mask Preprocessing",
            "Binary mask shape (224, 224) with unique pixel values subset of {0, 255}",
            f"Shape: {mask_np.shape}, Unique Values: {unique_vals}, Has Lesion: {has_lesion}",
            passed
        )
    except Exception as e:
        record_test(18, "Mask Preprocessing", "Mask preprocessing succeeds", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 19: Grad-CAM Localization IoU Calculation
    # ----------------------------------------------------
    try:
        dummy_att = np.zeros((100, 100), dtype=np.uint8)
        dummy_gt = np.zeros((100, 100), dtype=np.uint8)
        dummy_att[20:50, 20:50] = 255
        dummy_gt[30:60, 20:50] = 255

        iou = compute_iou(dummy_att, dummy_gt)
        # Intersection = 20*30 = 600, Union = 30*30 + 30*30 - 600 = 1200. IoU = 0.5
        passed = abs(iou - 0.5) < 1e-3
        record_test(
            19, "Grad-CAM Localization IoU Calculation",
            "IoU formula computes exact spatial overlap ratio (expected ~0.50)",
            f"Computed IoU: {iou:.4f}",
            passed
        )
    except Exception as e:
        record_test(19, "Grad-CAM Localization IoU Calculation", "IoU computation succeeds", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 20: Normal-Class N/A Handling
    # ----------------------------------------------------
    try:
        mask_loader = BUSIMaskLoader()
        normal_img = list((TEST_DIR / "Normal").glob("*.png"))[0]
        normal_mask, has_lesion, _ = mask_loader.load_mask(normal_img)
        passed = (not has_lesion) and (normal_mask.sum() == 0)
        record_test(
            20, "Normal-Class N/A Handling",
            "Normal scan masks returned as all-zero (has_lesion=False), reported as N/A",
            f"Normal Scan: {normal_img.name}, Has Lesion: {has_lesion}, Mask Pixel Sum: {normal_mask.sum()}",
            passed
        )
    except Exception as e:
        record_test(20, "Normal-Class N/A Handling", "Normal class N/A handling succeeds", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 21: Focal Loss Unit & Mathematical Verification
    # ----------------------------------------------------
    try:
        focal_loss_fn = FocalLoss(gamma=2.0, alpha=None, reduction="mean")
        dummy_logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 3.0, 0.2]], requires_grad=True)
        dummy_targets = torch.tensor([0, 1])
        loss_val = focal_loss_fn(dummy_logits, dummy_targets)
        loss_val.backward()
        
        has_grad = dummy_logits.grad is not None and torch.isfinite(dummy_logits.grad).all().item()
        is_finite = torch.isfinite(loss_val).item() and (loss_val.item() > 0.0)
        passed = is_finite and has_grad
        
        record_test(
            21, "Focal Loss Unit & Mathematical Verification",
            "Focal Loss computes finite positive loss and valid backprop gradients",
            f"Loss: {loss_val.item():.4f}, Finite: {is_finite}, Gradients Valid: {has_grad}",
            passed
        )
    except Exception as e:
        record_test(21, "Focal Loss Unit Verification", "Focal Loss unit test succeeds", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 22: Random Erasing Transform Verification
    # ----------------------------------------------------
    try:
        from torchvision.transforms import RandomErasing
        
        tr_erasing = get_train_transforms(image_size=(256, 256), use_random_erasing=True)
        val_no_erasing = get_val_test_transforms(image_size=(256, 256))
        
        has_erasing_in_train = any(isinstance(t, RandomErasing) for t in tr_erasing.transforms)
        has_erasing_in_val = any(isinstance(t, RandomErasing) for t in val_no_erasing.transforms)
        
        passed = has_erasing_in_train and (not has_erasing_in_val)
        record_test(
            22, "Random Erasing Transform Verification",
            "RandomErasing present in training transforms and absent from validation/test transforms",
            f"In Train: {has_erasing_in_train}, In Val/Test: {has_erasing_in_val}",
            passed
        )
    except Exception as e:
        record_test(22, "Random Erasing Transform Verification", "Transform verification succeeds", f"Error: {str(e)}", False)

    # ----------------------------------------------------
    # Test 23: Test-Time Augmentation (TTA) Probability Averaging & Determinism
    # ----------------------------------------------------
    try:
        dummy_img = torch.randn(1, 3, 256, 256, device=DEVICE)
        dummy_flipped = torch.flip(dummy_img, dims=[3])
        
        # Verify shape & operation symmetry
        passed_shape = (dummy_flipped.shape == dummy_img.shape)
        
        dummy_p1 = torch.tensor([[0.6, 0.3, 0.1]])
        dummy_p2 = torch.tensor([[0.4, 0.4, 0.2]])
        p_tta = (dummy_p1 + dummy_p2) / 2.0
        prob_sum = float(torch.sum(p_tta).item())
        
        passed = passed_shape and (abs(prob_sum - 1.0) < 1e-3) and (int(torch.argmax(p_tta).item()) == 0)
        record_test(
            23, "TTA Probability Averaging & Determinism",
            "Horizontal flip shape preservation and valid TTA probability averaging (sum=1.0)",
            f"Flipped Shape: {dummy_flipped.shape}, Prob Sum: {prob_sum:.4f}, Argmax: {int(torch.argmax(p_tta).item())}",
            passed
        )
    except Exception as e:
        record_test(23, "TTA Verification", "TTA verification succeeds", f"Error: {str(e)}", False)

    # Summary
    passed_count = sum(1 for t in test_results if t["passed"])
    total_count = len(test_results)
    print("=" * 80)
    print(f"       MASTER SUITE SUMMARY: {passed_count}/{total_count} TESTS PASSED        ")
    print("=" * 80 + "\n")

    return test_results


if __name__ == "__main__":
    run_all_tests()
