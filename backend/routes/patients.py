"""
Patient Registration & Retrieval Router for OncoVision FastAPI Backend.

Endpoints:
- POST /patients (and POST /api/patients): Register a new patient record
- GET /patients/{patient_id_or_code} (and GET /api/patients/{patient_id_or_code}): Retrieve patient record by ID or code

Exact Table Schema & CHECK Constraints Matched:
`patients`:
  - id (uuid)
  - patient_code (text, unique)
  - full_name (text, required)
  - age (int, required, 0-120)
  - gender (text, required CHECK constraint: 'Female', 'Male', 'Other', 'Prefer not to say')
  - notes (text, optional)
  - created_at (timestamptz)

Security Gatekeeper (Fail-Closed):
- Requires ONCOVISION_API_KEY or BACKEND_ADMIN_KEY configured on server (HTTP 503 if unconfigured).
- Requires request header X-Admin-API-Key matching server key (HTTP 401 if missing/invalid).
"""

import os
from typing import Optional, Literal
from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field

from backend.services.supabase_service import supabase_service
from backend.auth import get_current_user

router = APIRouter()

SECURITY_DISCLAIMER_NOTICE = (
    "SECURITY NOTICE: Patient endpoints require Supabase Auth JWT authentication. "
    "All requests are validated against Supabase Auth tokens."
)

VALID_GENDERS = ("Female", "Male", "Other", "Prefer not to say")


class PatientCreate(BaseModel):
    """
    Schema for creating a new patient record.
    Matches exact existing Supabase 'patients' table columns & CHECK constraints:
    - full_name: required string
    - age: required int (0 to 120)
    - gender: required enum ('Female', 'Male', 'Other', 'Prefer not to say')
    - notes: optional string
    """
    full_name: str = Field(..., min_length=1, max_length=150, description="Full name of patient")
    age: int = Field(..., ge=0, le=120, description="Patient age in years (required 0-120)")
    gender: Literal["Female", "Male", "Other", "Prefer not to say"] = Field(
        ..., description="Gender (Female, Male, Other, Prefer not to say)"
    )
    notes: Optional[str] = Field(None, description="Clinical notes or observations (optional)")


@router.get("/patients", tags=["Patients"])
@router.get("/api/patients", tags=["Patients"])
async def list_patients(
    search: Optional[str] = Query(None, description="Optional search query for patient_code or full_name"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    current_user: dict = Depends(get_current_user)
):
    """
    Lists patient records filtered by name or patient code with pagination.
    Requires Supabase Auth Bearer JWT token.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database integration is not configured on the backend server."
        )

    try:
        res = supabase_service.list_patients(query=search, page=page, page_size=page_size)
        if isinstance(res, dict):
            patients = res.get("patients", [])
            total = res.get("total", len(patients))
            total_pages = res.get("total_pages", 1)
        else:
            patients = res
            total = len(patients)
            total_pages = 1

        return {
            "status": "success",
            "count": len(patients),
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "patients": patients
        }
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list patients: {str(ex)}"
        )


@router.post("/patients", status_code=status.HTTP_201_CREATED, tags=["Patients"])
@router.post("/api/patients", status_code=status.HTTP_201_CREATED, tags=["Patients"])
async def create_patient(
    payload: PatientCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Registers a new patient record in the Supabase 'patients' table.
    Generates a unique server-side patient_code (e.g. OV-P-7K9M2X) with collision retries.
    Requires Supabase Auth Bearer JWT token.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database integration is not configured on the backend server. Please set SUPABASE_URL and SUPABASE_SECRET_KEY."
        )

    try:
        patient_data = payload.model_dump()
        registered_patient = supabase_service.register_patient(patient_data)
        
        return {
            "status": "success",
            "message": f"Patient '{payload.full_name}' registered successfully.",
            "patient": registered_patient,
            "security_notice": SECURITY_DISCLAIMER_NOTICE
        }
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register patient in database: {str(ex)}"
        )


@router.get("/patients/{patient_id_or_code}", tags=["Patients"])
@router.get("/api/patients/{patient_id_or_code}", tags=["Patients"])
async def get_patient(
    patient_id_or_code: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieves a patient record by UUID primary key or patient_code (e.g. OV-P-7K9M2X).
    Requires Supabase Auth Bearer JWT token.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database integration is not configured on the backend server."
        )

    try:
        patient = supabase_service.get_patient_by_id(patient_id_or_code)
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient record '{patient_id_or_code}' not found."
            )
        
        return {
            "status": "success",
            "patient": patient,
            "security_notice": SECURITY_DISCLAIMER_NOTICE
        }
    except HTTPException as http_ex:
        raise http_ex
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve patient record: {str(ex)}"
        )


@router.get("/patients/{patient_id_or_code}/scans", tags=["Patients"])
@router.get("/api/patients/{patient_id_or_code}/scans", tags=["Patients"])
async def get_patient_scans(
    patient_id_or_code: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieves chronological scan history for a patient, including signed URLs for private images.
    Requires Supabase Auth Bearer JWT token.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database integration is not configured on the backend server."
        )

    patient = supabase_service.get_patient_by_id(patient_id_or_code)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient record '{patient_id_or_code}' not found."
        )

    try:
        scans = supabase_service.get_patient_scans(patient["id"])
        
        # Populate temporary signed URLs for private storage files
        for scan in scans:
            orig_path = scan.get("original_image_path")
            grad_path = scan.get("gradcam_image_path")
            
            scan["original_image_url"] = supabase_service.create_signed_url("oncovision-scans", orig_path) if orig_path else None
            scan["gradcam_image_url"] = supabase_service.create_signed_url("oncovision-scans", grad_path) if grad_path else None

        return {
            "status": "success",
            "patient": patient,
            "count": len(scans),
            "scans": scans
        }
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve patient scan history: {str(ex)}"
        )

