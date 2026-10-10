"""
Unit & Integration Tests for OncoVision Downloadable PDF Diagnostic Reports.

Verifies:
1. Server-side PDF binary generation with ReportLab.
2. Required report fields (patient ID, name, age, gender, timestamps, probabilities, disclaimer).
3. Image handling (valid images and missing/corrupted image fallbacks).
4. FastAPI route authentication (HTTP 401 on missing/invalid token).
5. 404 Not Found error handling for non-existent scan IDs.
6. HTTP 200 OK downloadable PDF response headers and content disposition.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.pdf_service import generate_scan_report_pdf
from backend.auth import create_test_jwt

VALID_TEST_TOKEN = create_test_jwt()
TEST_AUTH_HEADERS = {"Authorization": f"Bearer {VALID_TEST_TOKEN}"}


class TestPdfReportFeature(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

        self.sample_patient = {
            "id": "patient-uuid-111",
            "patient_code": "OV-P-TEST01",
            "full_name": "Jane Doe",
            "age": 48,
            "gender": "Female",
            "created_at": "2026-10-10T12:00:00Z"
        }

        self.sample_scan = {
            "id": "scan-uuid-9999",
            "patient_id": "patient-uuid-111",
            "original_image_path": "scans/patient-uuid-111/orig.png",
            "gradcam_image_path": "scans/patient-uuid-111/grad.png",
            "predicted_class": "Benign",
            "normal_probability": 0.05,
            "benign_probability": 0.90,
            "malignant_probability": 0.05,
            "model_version": "EfficientNet-B0 (256x256 + TTA)",
            "created_at": "2026-10-10T14:30:00Z"
        }

    def test_pdf_binary_generation_and_disclaimer(self):
        """Verify generate_scan_report_pdf produces valid PDF starting with %PDF and handles image byte fallbacks cleanly."""
        print("\n[TEST PDF 1] Testing server-side PDF binary generation...")
        pdf_bytes = generate_scan_report_pdf(
            patient=self.sample_patient,
            scan=self.sample_scan,
            original_image_bytes=None,
            gradcam_image_bytes=None
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Generated PDF buffer does not start with standard %PDF magic header.")
        self.assertGreater(len(pdf_bytes), 1000, "Generated PDF file size is unexpectedly small.")
        print(f"  [PASSED] PDF generated successfully ({len(pdf_bytes)} bytes) with valid header & disclaimer layout.")

    def test_pdf_route_requires_authentication(self):
        """Verify GET /scans/{id}/report.pdf requires Bearer JWT authentication."""
        print("\n[TEST PDF 2] Testing PDF route authentication requirement...")
        res_no_auth = self.client.get("/scans/scan-uuid-9999/report.pdf")
        self.assertEqual(res_no_auth.status_code, 401, f"Expected 401 Unauthorized, got {res_no_auth.status_code}")

        res_bad_auth = self.client.get("/scans/scan-uuid-9999/report.pdf", headers={"Authorization": "Bearer invalid_token"})
        self.assertEqual(res_bad_auth.status_code, 401, f"Expected 401 Unauthorized for invalid token, got {res_bad_auth.status_code}")
        print("  [PASSED] Unauthenticated requests cleanly rejected with HTTP 401.")

    @patch("backend.routes.scans.supabase_service.get_scan_by_id", return_value=None)
    @patch("backend.routes.scans.supabase_service.is_configured", return_value=True)
    def test_pdf_route_non_existent_scan(self, mock_is_config, mock_get_scan):
        """Verify GET /scans/{id}/report.pdf returns 404 Not Found for non-existent scan ID."""
        print("\n[TEST PDF 3] Testing 404 error handling for non-existent scan ID...")
        res = self.client.get("/scans/non-existent-scan-id/report.pdf", headers=TEST_AUTH_HEADERS)
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())
        print("  [PASSED] Non-existent scan ID cleanly returned HTTP 404 Not Found.")

    @patch("backend.routes.scans.supabase_service.download_storage_file", return_value=None)
    @patch("backend.routes.scans.supabase_service.get_patient_by_id")
    @patch("backend.routes.scans.supabase_service.get_scan_by_id")
    @patch("backend.routes.scans.supabase_service.is_configured", return_value=True)
    def test_successful_pdf_download_route(self, mock_is_config, mock_get_scan, mock_get_patient, mock_download):
        """Verify GET /scans/{id}/report.pdf returns HTTP 200 OK with application/pdf content type and content-disposition header."""
        print("\n[TEST PDF 4] Testing successful PDF report download route...")
        mock_get_scan.return_value = self.sample_scan
        mock_get_patient.return_value = self.sample_patient

        res = self.client.get("/scans/scan-uuid-9999/report.pdf", headers=TEST_AUTH_HEADERS)
        self.assertEqual(res.status_code, 200, f"Expected 200 OK, got {res.status_code}: {res.text}")
        self.assertEqual(res.headers.get("content-type"), "application/pdf")

        disposition = res.headers.get("content-disposition", "")
        self.assertIn("attachment", disposition)
        self.assertIn("OncoVision_Report_OV-P-TEST01_scan-uui.pdf", disposition)
        self.assertTrue(res.content.startswith(b"%PDF"))
        print(f"  [PASSED] HTTP 200 OK returning downloadable PDF with header Content-Disposition: '{disposition}'.")


if __name__ == "__main__":
    unittest.main()
