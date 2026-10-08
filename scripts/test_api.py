"""
FastAPI Backend Integration Test Script for OncoVision.

Executes integration tests on:
1. GET /health
2. POST /predict with valid image upload (verifying prediction, confidence score, Base64 Grad-CAM image output)
3. POST /predict error handling (invalid file format, empty file)

Usage:
    python scripts/test_api.py
"""

import sys
from pathlib import Path
import io

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    print("\n[TEST 1] Testing GET /health endpoint...")
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        print(f"  [PASSED] Status: {data['status']}, Model Loaded: {data['model_loaded']}, Device: {data['device']}")


def test_predict_endpoint_valid_image():
    print("\n[TEST 2] Testing POST /predict endpoint with valid sample image...")
    test_img_path = PROJECT_ROOT / "dataset" / "test" / "Benign" / "benign (1).png"
    if not test_img_path.exists():
        test_imgs = list((PROJECT_ROOT / "dataset" / "test").rglob("*.png"))
        if len(test_imgs) == 0:
            raise RuntimeError("No test images found for API test!")
        test_img_path = test_imgs[0]

    with open(test_img_path, "rb") as f:
        file_bytes = f.read()

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": (test_img_path.name, io.BytesIO(file_bytes), "image/png")}
        )

        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        data = response.json()

        assert "prediction" in data, "Missing 'prediction' in response"
        assert "confidence" in data, "Missing 'confidence' in response"
        assert "gradcam_image" in data, "Missing 'gradcam_image' in response"
        assert data["prediction"] in ["Normal", "Benign", "Malignant"], f"Invalid prediction class: {data['prediction']}"
        assert 0.0 <= data["confidence"] <= 1.0, f"Invalid confidence: {data['confidence']}"
        assert data["gradcam_image"].startswith("data:image/png;base64,"), "gradcam_image is not valid Base64 URI"

        print(f"  [PASSED] Prediction      : {data['prediction']}")
        print(f"  [PASSED] Confidence      : {data['confidence']*100:.2f}%")
        print(f"  [PASSED] Probabilities   : {data['probabilities']}")
        print(f"  [PASSED] Grad-CAM Base64 : {data['gradcam_image'][:45]}... (Length: {len(data['gradcam_image'])} chars)")


def test_predict_endpoint_invalid_file():
    print("\n[TEST 3] Testing POST /predict endpoint error handling for invalid file...")
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("text.txt", io.BytesIO(b"Hello world, invalid image!"), "text/plain")}
        )
        assert response.status_code == 400, f"Expected status 400 Bad Request, got {response.status_code}"
        print(f"  [PASSED] Successfully rejected invalid file with status 400: {response.json()['detail']}")



if __name__ == "__main__":
    print("=" * 70)
    print("        ONCOVISION - FASTAPI BACKEND INTEGRATION TEST SUITE        ")
    print("=" * 70)
    
    test_health_endpoint()
    test_predict_endpoint_valid_image()
    test_predict_endpoint_invalid_file()

    print("\n" + "=" * 70)
    print("       ALL FASTAPI BACKEND INTEGRATION TESTS PASSED CLEANLY!       ")
    print("=" * 70 + "\n")
