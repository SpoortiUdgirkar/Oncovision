"""
Health Check Router for OncoVision FastAPI Backend.
"""

from fastapi import APIRouter
from backend.services.model_service import model_service
from src.config import CLASS_NAMES, DEVICE

router = APIRouter()


@router.get("/health", tags=["Health"])
@router.get("/api/health", tags=["Health"])
def health_check():
    """
    Health check endpoint returning API status, loaded model details, and target classes.
    """
    return {
        "status": "ok",
        "message": "OncoVision Backend API is healthy and operational.",
        "model_loaded": model_service.is_loaded(),
        "model_path": str(model_service.model_path) if model_service.model_path else None,
        "device": str(DEVICE),
        "target_classes": CLASS_NAMES,
        "version": "1.0.0"
    }
