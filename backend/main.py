"""
FastAPI REST API Server for OncoVision Breast Ultrasound Classification & Explainability.

Endpoints:
- GET /: Root status endpoint
- GET /health: Health check endpoint
- POST /predict: Ultrasound scan image prediction & Grad-CAM visual explainability endpoint

Usage:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services.model_service import model_service
from backend.routes.health import router as health_router
from backend.routes.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager: Loads trained model into memory at server startup
    to prevent reloading or retraining on individual API requests.
    """
    print("[SERVER STARTUP] Initializing OncoVision Backend Services...")
    try:
        model_service.load_model()
    except Exception as e:
        print(f"[WARNING] Model load failed on startup: {str(e)}")

    yield

    print("[SERVER SHUTDOWN] Shutting down OncoVision Backend Services...")


app = FastAPI(
    title="OncoVision API",
    description="Deep Learning REST API for Breast Ultrasound Cancer Classification and Grad-CAM Explainability",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS Middleware for React frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health_router)
app.include_router(predict_router)


@app.get("/", tags=["Root"])
def read_root():
    """Root status endpoint."""
    return {
        "service": "OncoVision AI Backend API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs"
    }
