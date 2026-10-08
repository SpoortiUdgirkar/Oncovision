# OncoVision - FastAPI REST API Documentation

The **OncoVision Backend** is built using **FastAPI** and **Uvicorn**. It provides RESTful API endpoints for model health monitoring, image validation, breast ultrasound cancer classification, and Base64-encoded Grad-CAM explainability heatmap generation.

---

## 1. Server Configuration Overview

* **Base URL**: `http://localhost:8000`
* **Interactive OpenAPI Specs (Swagger UI)**: `http://localhost:8000/docs`
* **ReDoc Specification**: `http://localhost:8000/redoc`
* **CORS Policy**: Enabled for cross-origin requests (`allow_origins=["*"]`) from React single-page applications (`http://localhost:5173`).

---

## 2. API Endpoints Specification

### 2.1 Health Check Endpoint

#### `GET /health` or `GET /api/health`
Returns system status, device allocation, and model loading status.

* **Query Parameters**: None
* **Success Response (HTTP 200 OK)**:
```json
{
  "status": "ok",
  "message": "OncoVision Backend API is healthy and operational.",
  "model_loaded": true,
  "model_path": "C:\\Users\\spoor\\.gemini\\antigravity-ide\\scratch\\OncoVision\\models\\resnet50_best.pth",
  "device": "cpu",
  "target_classes": [
    "Normal",
    "Benign",
    "Malignant"
  ],
  "version": "1.0.0"
}
```

---

### 2.2 Prediction & Grad-CAM Endpoint

#### `POST /predict` or `POST /api/predict`
Accepts an uploaded breast ultrasound image, executes ResNet50 model inference, generates a Grad-CAM heatmap, and returns JSON classification results with a Base64 PNG visualization string.

* **Content-Type**: `multipart/form-data`
* **Request Body**:
  * `file` (binary file upload, required): Uploaded ultrasound scan (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`). Maximum size limit: **15 MB**.

* **Success Response (HTTP 200 OK)**:
```json
{
  "prediction": "Benign",
  "confidence": 0.3698,
  "gradcam_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "probabilities": {
    "Normal": 0.2832,
    "Benign": 0.3698,
    "Malignant": 0.3470
  },
  "message": "Successfully classified ultrasound scan as 'Benign' with 37.0% confidence.",
  "disclaimer": "Model Explanation Notice: This Grad-CAM activation map highlights regions that influenced the deep learning model's prediction. It serves as an AI attention visualization tool and is NOT medically definitive or diagnostic proof."
}
```

* **Error Responses**:
  * **HTTP 400 Bad Request** (Unsupported format or file size exceeded):
  ```json
  {
    "detail": "Unsupported file extension '.pdf'. Supported extensions: ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']"
  }
  ```
  * **HTTP 400 Bad Request** (Corrupted image file):
  ```json
  {
    "detail": "Uploaded file is corrupted or not a valid readable image format."
  }
  ```
  * **HTTP 500 Internal Server Error** (Uninitialized model or internal exception):
  ```json
  {
    "detail": "Model is not loaded into memory. Please verify server startup status."
  }
  ```

---

## 3. Client Code Integration Examples

### 3.1 cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@dataset/test/Benign/benign (1).png"
```

### 3.2 Python Client Example (`requests`)

```python
import requests

url = "http://localhost:8000/predict"
file_path = "dataset/test/Benign/benign (1).png"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/png")}
    response = requests.post(url, files=files)

data = response.json()
print("Prediction :", data["prediction"])
print("Confidence :", f"{data['confidence'] * 100:.2f}%")
print("Base64 Len :", len(data["gradcam_image"]))
```

### 3.3 React JavaScript Client Example (`fetch`)

```javascript
const formData = new FormData();
formData.append('file', fileObject);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData,
});

const data = await response.json();
console.log('Prediction:', data.prediction);
console.log('Grad-CAM Data URI:', data.gradcam_image);
```
