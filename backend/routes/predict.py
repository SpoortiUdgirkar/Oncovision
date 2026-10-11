"""
Inference & Prediction Router for OncoVision FastAPI Backend.

Endpoint: POST /predict (and POST /api/predict)
Accepts uploaded ultrasound scan, executes model inference using primary EfficientNet-B0 model,
generates Grad-CAM explainability heatmap, and optionally persists scan images and database record if a patient is selected.
"""

import io
import gc
import base64
import uuid
import asyncio
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from backend.services.preprocessing_service import validate_image_file, load_and_preprocess_image
from backend.services.model_service import model_service
from backend.services.gradcam_service import generate_gradcam_base64
from backend.services.supabase_service import supabase_service
from backend.auth import get_current_user
from src.gradcam.explain import MEDICAL_DISCLAIMER_TEXT

router = APIRouter()


@router.post("/predict", tags=["Prediction"])
@router.post("/api/predict", tags=["Prediction"])
async def predict_ultrasound_scan(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    patient_code: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Accepts an uploaded breast ultrasound image and returns classification prediction,
    confidence score, class probabilities, primary model metadata, and Base64-encoded Grad-CAM heatmap.
    If patient_id or patient_code is provided, uploads original and Grad-CAM images to private storage
    and inserts a scan record linked to the patient's UUID in Supabase DB.
    """
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded into memory. Please verify server startup status."
        )

    # Validate patient existence if patient_id or patient_code is provided
    target_patient_id = patient_id or patient_code
    patient_record = None
    if target_patient_id:
        if supabase_service.is_configured():
            patient_record = supabase_service.get_patient_by_id(target_patient_id)
            if not patient_record:
                raise HTTPException(
                    status_code=404,
                    detail=f"Patient with ID or code '{target_patient_id}' not found."
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

        # Release PyTorch tensor and raw upload bytes immediately to free memory
        del img_tensor
        del file_bytes
        gc.collect()

        saved_scan = None
        # Save scan images and DB record if patient is selected and Supabase is configured
        if patient_record and supabase_service.is_configured():
            scan_uuid = str(uuid.uuid4())
            patient_uuid = patient_record["id"]

            orig_path = f"scans/{patient_uuid}/{scan_uuid}_original.png"
            gradcam_ext = "jpg" if "image/jpeg" in base64_gradcam else "png"
            gradcam_path = f"scans/{patient_uuid}/{scan_uuid}_gradcam.{gradcam_ext}"

            # Prepare original image byte buffer
            orig_buffer = io.BytesIO()
            pil_img.save(orig_buffer, format="PNG")
            orig_bytes = orig_buffer.getvalue()
            orig_buffer.close()
            del orig_buffer
            del pil_img

            uploaded_paths = []
            try:
                # Step A: Sequential upload of original image to avoid concurrent thread and buffer spikes
                await asyncio.to_thread(
                    supabase_service.upload_storage_file,
                    "oncovision-scans",
                    orig_path,
                    orig_bytes,
                    "image/png"
                )
                uploaded_paths.append(orig_path)
                del orig_bytes
                gc.collect()

                # Step B: Decode Grad-CAM bytes and upload sequentially
                gradcam_bytes = base64.b64decode(base64_gradcam.split(",")[-1])
                gradcam_content_type = "image/jpeg" if gradcam_ext == "jpg" else "image/png"
                await asyncio.to_thread(
                    supabase_service.upload_storage_file,
                    "oncovision-scans",
                    gradcam_path,
                    gradcam_bytes,
                    gradcam_content_type
                )
                uploaded_paths.append(gradcam_path)
                del gradcam_bytes
                gc.collect()

                # Step C: Prepare database record payload
                scan_payload = {
                    "id": scan_uuid,
                    "patient_id": patient_uuid,
                    "original_image_path": orig_path,
                    "gradcam_image_path": gradcam_path,
                    "predicted_class": pred_class,
                    "normal_probability": round(float(probs_dict.get("Normal", 0.0)), 4),
                    "benign_probability": round(float(probs_dict.get("Benign", 0.0)), 4),
                    "malignant_probability": round(float(probs_dict.get("Malignant", 0.0)), 4),
                    "model_version": model_service.model_name,
                }

                # Step D: Insert scan record in database
                saved_scan = supabase_service.create_scan_record(scan_payload)

            except Exception as storage_db_err:
                # Atomic Cleanup on partial failure
                print(f"[ERROR] Failed during scan storage/DB record creation: {str(storage_db_err)}. Cleaning up uploaded files...")
                for path in uploaded_paths:
                    try:
                        supabase_service.delete_storage_file("oncovision-scans", path)
                    except Exception as cleanup_err:
                        print(f"[WARNING] Failed to clean up storage file '{path}': {str(cleanup_err)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to persist scan record or storage images: {str(storage_db_err)}"
                )
        else:
            if "pil_img" in locals():
                del pil_img
            gc.collect()

        response_data = {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "model_name": model_service.model_name,
            "gradcam_image": base64_gradcam,
            "probabilities": {k: round(v, 4) for k, v in probs_dict.items()},
            "message": f"Successfully classified ultrasound scan as '{pred_class}' with {confidence*100:.1f}% confidence using {model_service.model_name}.",
            "disclaimer": MEDICAL_DISCLAIMER_TEXT
        }

        if saved_scan:
            response_data["saved_scan"] = saved_scan
            response_data["patient"] = {
                "id": patient_record["id"],
                "patient_code": patient_record.get("patient_code"),
                "full_name": patient_record.get("full_name")
            }

        return response_data

    except HTTPException as http_ex:
        raise http_ex
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during prediction: {str(ex)}"
        )

