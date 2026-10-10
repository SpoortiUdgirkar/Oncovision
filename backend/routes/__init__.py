"""
API Routes for OncoVision FastAPI Backend.
"""

from backend.routes.health import router as health_router
from backend.routes.predict import router as predict_router
from backend.routes.patients import router as patients_router
from backend.routes.scans import router as scans_router

__all__ = ["health_router", "predict_router", "patients_router", "scans_router"]
