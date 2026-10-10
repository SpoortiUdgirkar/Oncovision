"""
Health Check Router for OncoVision FastAPI Backend.
"""

from fastapi import APIRouter
from backend.services.model_service import model_service
from backend.services.supabase_service import supabase_service
from src.config import CLASS_NAMES, DEVICE

router = APIRouter()


@router.get("/health", tags=["Health"])
@router.get("/api/health", tags=["Health"])
def health_check():
    """
    Health check endpoint returning API status, loaded model details, target classes,
    and Supabase integration configuration status.
    """
    supa_status = supabase_service.get_config_status()
    return {
        "status": "ok",
        "message": "OncoVision Backend API is healthy and operational.",
        "model_loaded": model_service.is_loaded(),
        "model_path": str(model_service.model_path) if model_service.model_path else None,
        "device": str(DEVICE),
        "target_classes": CLASS_NAMES,
        "supabase_configured": supa_status["configured"],
        "supabase_status": {
            "configured": supa_status["configured"],
            "url": supa_status["supabase_url"],
            "secret_key_masked": supa_status["secret_key_masked"]
        },
        "version": "1.0.0"
    }
