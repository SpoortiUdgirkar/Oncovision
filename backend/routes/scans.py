"""
Scans and Diagnostic PDF Report Routes for OncoVision API.

Provides authenticated PDF report generation endpoints for individual saved scans.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response
from typing import Dict, Any

from backend.auth import get_current_user
from backend.services.supabase_service import supabase_service
from backend.services.pdf_service import generate_scan_report_pdf

router = APIRouter()


@router.get("/scans/{scan_id}/report.pdf", tags=["Reports"])
@router.get("/api/scans/{scan_id}/report.pdf", tags=["Reports"])
async def download_scan_report(
    scan_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Generates and returns a downloadable, printable PDF diagnostic report for a saved scan.
    Requires Supabase Auth Bearer JWT token.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database integration is not configured on the backend server."
        )

    # 1. Retrieve scan record by scan UUID
    scan = supabase_service.get_scan_by_id(scan_id)
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan record '{scan_id}' not found."
        )

    # 2. Retrieve associated patient record
    patient_id = scan.get("patient_id")
    if not patient_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan record '{scan_id}' does not have a linked patient ID."
        )

    patient = supabase_service.get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Associated patient record '{patient_id}' for scan '{scan_id}' was not found."
        )

    # 3. Retrieve raw image bytes from private Supabase storage bucket
    orig_path = scan.get("original_image_path")
    grad_path = scan.get("gradcam_image_path")

    orig_bytes = supabase_service.download_storage_file("oncovision-scans", orig_path) if orig_path else None
    grad_bytes = supabase_service.download_storage_file("oncovision-scans", grad_path) if grad_path else None

    # 4. Generate PDF report buffer
    try:
        pdf_bytes = generate_scan_report_pdf(
            patient=patient,
            scan=scan,
            original_image_bytes=orig_bytes,
            gradcam_image_bytes=grad_bytes
        )
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF report: {str(ex)}"
        )

    # 5. Format downloadable filename
    patient_code = patient.get("patient_code", "Patient").replace(" ", "_")
    short_scan_id = scan_id[:8]
    filename = f"OncoVision_Report_{patient_code}_{short_scan_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )
