"""
Inference & Prediction Router for OncoVision FastAPI Backend.

Endpoint: POST /predict (and POST /api/predict)
Accepts uploaded ultrasound scan, executes model inference using primary EfficientNet-B0 model,
generates Grad-CAM explainability heatmap, and returns JSON response.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.services.preprocessing_service import validate_image_file, load_and_preprocess_image
from backend.services.model_service import model_service
from backend.services.gradcam_service import generate_gradcam_base64
from src.gradcam.explain import MEDICAL_DISCLAIMER_TEXT

router = APIRouter()


@router.post("/predict", tags=["Prediction"])
@router.post("/api/predict", tags=["Prediction"])
async def predict_ultrasound_scan(file: UploadFile = File(...)):
    """
    Accepts an uploaded breast ultrasound image and returns classification prediction,
    confidence score, class probabilities, primary model metadata, and Base64-encoded Grad-CAM heatmap.
    """
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded into memory. Please verify server startup status."
        )

    try:
        # Read file bytes
        file_bytes = await file.read()

        # Validate file
        validate_image_file(file, file_bytes)

        # Preprocess image
        pil_img, img_tensor = load_and_preprocess_image(file_bytes)

        # Run Model Inference with Horizontal-Flip TTA
        pred_class, confidence, probs_dict, pred_idx = model_service.predict(img_tensor, use_tta=True)

        # Generate Grad-CAM Base64 Overlay targeting TTA predicted class index
        base64_gradcam, _, _, _ = generate_gradcam_base64(
            model=model_service.model,
            orig_pil=pil_img,
            img_tensor=img_tensor,
            target_class=pred_idx
        )

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "model_name": model_service.model_name,
            "gradcam_image": base64_gradcam,
            "probabilities": {k: round(v, 4) for k, v in probs_dict.items()},
            "message": f"Successfully classified ultrasound scan as '{pred_class}' with {confidence*100:.1f}% confidence using {model_service.model_name}.",
            "disclaimer": MEDICAL_DISCLAIMER_TEXT
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during prediction: {str(ex)}"
        )
