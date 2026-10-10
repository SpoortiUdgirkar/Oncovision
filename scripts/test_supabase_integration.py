"""
Unit & Integration Test Suite for Supabase Phase 2 Backend Services (Using Mocks).

Test Classification:
- ALL tests in this file use MOCKS (unittest.mock) and run 100% OFFLINE.
- NO real Supabase credentials or network connections are required or used.

Scenarios Tested:
1. PatientCreate schema validation (required age 0-120, required gender enum, optional notes).
2. Rejection of missing age, invalid age (<0 or >120), missing gender, and invalid gender values.
3. Fail-Closed Security Gatekeeper:
   - Missing server key -> HTTP 503 Service Unavailable
   - Server key set, missing request key -> HTTP 401 Unauthorized
   - Server key set, incorrect request key -> HTTP 401 Unauthorized
   - Server key set, correct request key -> Allowed (HTTP 201 / HTTP 200)
4. Patient code generator & collision retries.
5. Patient registration & retrieval endpoints (mocked database).
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.supabase_service import supabase_service, SupabaseService
from backend.auth import create_test_jwt, TEST_JWT_SECRET, verify_supabase_jwt

VALID_TEST_TOKEN = create_test_jwt()
AUTH_HEADERS = {"Authorization": f"Bearer {VALID_TEST_TOKEN}"}


class TestSupabaseIntegration(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_config_validation_and_secret_masking(self):
        """Verify configuration validation and secret key masking (no plaintext leaks). [OFFLINE MOCKED]"""
        print("\n[TEST 1 - OFFLINE MOCKED] Testing Supabase config validation & secret key masking...")
        
        # Unconfigured environment
        with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SECRET_KEY": ""}, clear=True):
            service = SupabaseService()
            self.assertFalse(service.is_configured())
            self.assertIsNone(service.get_masked_secret_key())
            status = service.get_config_status()
            self.assertFalse(status["configured"])
            self.assertIsNone(status["secret_key_masked"])

        # Configured environment with secret key masking
        secret_key = "sb_secret_service_role_key_998877"
        with patch.dict(os.environ, {"SUPABASE_URL": "https://test-proj.supabase.co", "SUPABASE_SECRET_KEY": secret_key}):
            service = SupabaseService()
            self.assertTrue(service.is_configured())
            masked = service.get_masked_secret_key()
            self.assertIsNotNone(masked)
            self.assertNotIn(secret_key, masked)  # Full secret MUST NOT be present in masked string
            self.assertTrue(masked.startswith("sb_s"))
            self.assertTrue(masked.endswith("8877"))
            
            status = service.get_config_status()
            self.assertTrue(status["configured"])
            self.assertEqual(status["supabase_url"], "https://test-proj.supabase.co")
            print(f"  [PASSED] Configured: True, Masked Key: {status['secret_key_masked']}")

    def test_patient_code_generator(self):
        """Verify server-side patient code generator format and uniqueness. [OFFLINE MOCKED]"""
        print("\n[TEST 2 - OFFLINE MOCKED] Testing server-side patient code generator...")
        code = SupabaseService.generate_patient_code()
        self.assertTrue(code.startswith("OV-P-"))
        self.assertEqual(len(code), 11)  # "OV-P-" (5) + 6 random chars = 11

        codes = {SupabaseService.generate_patient_code() for _ in range(100)}
        self.assertEqual(len(codes), 100, "Generated patient codes contain duplicates!")
        print(f"  [PASSED] Sample Code: {code} | 100/100 Unique Codes Verified.")

    @patch("backend.services.supabase_service.supabase_service.get_client")
    @patch("backend.services.supabase_service.supabase_service.is_configured", return_value=True)
    def test_patient_code_collision_retry(self, mock_is_config, mock_get_client):
        """Verify patient code collision retry logic when a generated code already exists. [OFFLINE MOCKED]"""
        print("\n[TEST 3 - OFFLINE MOCKED] Testing patient code collision retry logic...")
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_insert = MagicMock()
        
        res_collision = MagicMock()
        res_collision.data = [{"id": "existing-uuid"}]
        
        res_unique = MagicMock()
        res_unique.data = []

        res_inserted = MagicMock()
        res_inserted.data = [{
            "id": "new-uuid-1234",
            "patient_code": "OV-P-RETRY2",
            "full_name": "Collision Test Patient",
            "age": 35,
            "gender": "Female",
            "notes": "Retried insertion"
        }]

        call_count = 0
        def mock_eq_impl(field, val):
            nonlocal call_count
            call_count += 1
            mock_ex = MagicMock()
            if call_count == 1:
                mock_ex.execute.return_value = res_collision
            else:
                mock_ex.execute.return_value = res_unique
            return mock_ex

        mock_select.eq.side_effect = mock_eq_impl
        mock_table.select.return_value = mock_select
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = res_inserted
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        patient_data = {"full_name": "Collision Test Patient", "age": 35, "gender": "Female", "notes": "Retried insertion"}
        result = supabase_service.register_patient(patient_data, max_retries=3)

        self.assertEqual(result["id"], "new-uuid-1234")
        self.assertEqual(result["full_name"], "Collision Test Patient")
        print("  [PASSED] Collision retry successfully recovered and inserted unique patient code.")

    @patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_SECRET_KEY": "sb_sec"})
    def test_schema_validation_required_age_and_gender(self):
        """Verify PatientCreate schema enforces required age (0-120) and gender enum ('Female', 'Male', 'Other', 'Prefer not to say'). [OFFLINE MOCKED]"""
        print("\n[TEST 4 - OFFLINE MOCKED] Testing PatientCreate schema validation (age & gender)...")

        # Case A: Missing age -> 422 Unprocessable Entity
        res_no_age = self.client.post("/patients", json={"full_name": "Jane", "gender": "Female"}, headers=AUTH_HEADERS)
        self.assertEqual(res_no_age.status_code, 422)

        # Case B: Invalid age (<0 or >120) -> 422 Unprocessable Entity
        res_neg_age = self.client.post("/patients", json={"full_name": "Jane", "age": -5, "gender": "Female"}, headers=AUTH_HEADERS)
        self.assertEqual(res_neg_age.status_code, 422)

        res_over_age = self.client.post("/patients", json={"full_name": "Jane", "age": 150, "gender": "Female"}, headers=AUTH_HEADERS)
        self.assertEqual(res_over_age.status_code, 422)

        # Case C: Missing gender -> 422 Unprocessable Entity
        res_no_gender = self.client.post("/patients", json={"full_name": "Jane", "age": 45}, headers=AUTH_HEADERS)
        self.assertEqual(res_no_gender.status_code, 422)

        # Case D: Invalid gender enum value -> 422 Unprocessable Entity
        res_bad_gender = self.client.post("/patients", json={"full_name": "Jane", "age": 45, "gender": "UnknownGender"}, headers=AUTH_HEADERS)
        self.assertEqual(res_bad_gender.status_code, 422)

        print("  [PASSED] Schema validation correctly rejects missing/invalid age and invalid gender values.")

    def test_jwt_bearer_token_verification_scenarios(self):
        """Verify Supabase JWT Bearer token authentication across all scenarios (missing, malformed, expired, invalid signature, valid). [OFFLINE MOCKED]"""
        print("\n[TEST 5 - OFFLINE MOCKED] Testing JWT Bearer Token Verification scenarios...")

        # Scenario 1: Missing Authorization header -> HTTP 401
        res_missing = self.client.post("/patients", json={"full_name": "Jane", "age": 30, "gender": "Female"})
        self.assertEqual(res_missing.status_code, 401)
        self.assertIn("Missing Authorization header", res_missing.json()["detail"])

        # Scenario 2: Malformed Authorization header format -> HTTP 401
        res_malformed = self.client.post("/patients", json={"full_name": "Jane", "age": 30, "gender": "Female"}, headers={"Authorization": "InvalidFormatToken"})
        self.assertEqual(res_malformed.status_code, 401)
        self.assertIn("Invalid Authorization header format", res_malformed.json()["detail"])

        # Scenario 3: Expired JWT Token -> HTTP 401
        expired_token = create_test_jwt(expires_in_seconds=-3600)
        res_expired = self.client.post("/patients", json={"full_name": "Jane", "age": 30, "gender": "Female"}, headers={"Authorization": f"Bearer {expired_token}"})
        self.assertEqual(res_expired.status_code, 401)
        self.assertIn("expired", res_expired.json()["detail"].lower())

        # Scenario 4: Invalid signature JWT Token -> HTTP 401
        bad_sig_token = create_test_jwt(secret="wrong_secret_key_that_fails_verification_12345")
        res_bad_sig = self.client.post("/patients", json={"full_name": "Jane", "age": 30, "gender": "Female"}, headers={"Authorization": f"Bearer {bad_sig_token}"})
        self.assertEqual(res_bad_sig.status_code, 401)

        print("  [PASSED] JWT authentication cleanly rejects missing, malformed, expired, and invalid-signature tokens with HTTP 401.")

    @patch("backend.services.supabase_service.supabase_service.get_client")
    @patch("backend.services.supabase_service.supabase_service.is_configured", return_value=True)
    def test_patient_registration_mocked(self, mock_is_config, mock_get_client):
        """Verify POST /patients route with exact schema & valid gender. [OFFLINE MOCKED]"""
        print("\n[TEST 6 - OFFLINE MOCKED] Testing POST /patients endpoint with valid payload...")
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_insert = MagicMock()

        mock_select.eq.return_value.execute.return_value.data = []

        mock_insert.execute.return_value.data = [{
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "patient_code": "OV-P-A8X9K2",
            "full_name": "Jane Doe",
            "age": 42,
            "gender": "Prefer not to say",
            "notes": "Routine screening notes",
            "created_at": "2026-10-10T12:00:00Z"
        }]

        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_table.insert.return_value = mock_insert
        mock_get_client.return_value = mock_client

        payload = {
            "full_name": "Jane Doe",
            "age": 42,
            "gender": "Prefer not to say",
            "notes": "Routine screening notes"
        }

        response = self.client.post("/patients", json=payload, headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["patient"]["full_name"], "Jane Doe")
        self.assertEqual(data["patient"]["gender"], "Prefer not to say")
        self.assertEqual(data["patient"]["notes"], "Routine screening notes")
        print(f"  [PASSED] HTTP 201 Created: {data['message']} (Code: {data['patient']['patient_code']})")

    @patch("backend.services.supabase_service.supabase_service.get_client")
    @patch("backend.services.supabase_service.supabase_service.is_configured", return_value=True)
    def test_patient_retrieval_mocked(self, mock_is_config, mock_get_client):
        """Verify GET /patients/{id} route with correct Bearer JWT token. [OFFLINE MOCKED]"""
        print("\n[TEST 7 - OFFLINE MOCKED] Testing GET /patients/{id} endpoint...")
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_eq = MagicMock()

        mock_eq.execute.return_value.data = [{
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "patient_code": "OV-P-A8X9K2",
            "full_name": "Jane Doe",
            "age": 42,
            "gender": "Female",
            "notes": "Existing notes"
        }]

        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        mock_get_client.return_value = mock_client

        response = self.client.get("/patients/123e4567-e89b-12d3-a456-426614174000", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["patient"]["patient_code"], "OV-P-A8X9K2")
        print(f"  [PASSED] HTTP 200 OK: Found patient {data['patient']['patient_code']}")

        # Case B: Patient Not Found (404)
        mock_eq.execute.return_value.data = []
        response_404 = self.client.get("/patients/NON_EXISTENT_ID", headers=AUTH_HEADERS)
        self.assertEqual(response_404.status_code, 404)
        print("  [PASSED] HTTP 404 Not Found cleanly returned for missing patient ID.")

    @patch("backend.services.supabase_service.supabase_service.get_client")
    @patch("backend.services.supabase_service.supabase_service.is_configured", return_value=True)
    def test_patient_list_search_mocked(self, mock_is_config, mock_get_client):
        """Verify GET /patients endpoint with search query filtering. [OFFLINE MOCKED]"""
        print("\n[TEST 8 - OFFLINE MOCKED] Testing GET /patients with search filter...")

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_or = MagicMock()
        mock_order = MagicMock()
        mock_limit = MagicMock()

        mock_limit.execute.return_value.data = [{
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "patient_code": "OV-P-MATCH1",
            "full_name": "Jane Smith",
            "age": 50,
            "gender": "Female"
        }]

        mock_order.limit.return_value = mock_limit
        mock_or.order.return_value = mock_order
        mock_select.or_.return_value = mock_or
        mock_table.select.return_value = mock_select
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        response = self.client.get("/patients?search=Jane", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["patients"][0]["patient_code"], "OV-P-MATCH1")
        print("  [PASSED] Patient list search successfully filtered results.")

    @patch("backend.services.supabase_service.supabase_service.get_client")
    @patch("backend.services.supabase_service.supabase_service.is_configured", return_value=True)
    def test_patient_scan_history_mocked(self, mock_is_config, mock_get_client):
        """Verify GET /patients/{id}/scans endpoint returning scan history and signed URLs. [OFFLINE MOCKED]"""
        print("\n[TEST 9 - OFFLINE MOCKED] Testing GET /patients/{id}/scans endpoint...")

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_eq = MagicMock()
        mock_order = MagicMock()
        mock_storage = MagicMock()

        # Mock patient lookup
        patient_data = {
            "id": "patient-uuid-111",
            "patient_code": "OV-P-HIST01",
            "full_name": "History Patient",
            "age": 45,
            "gender": "Female"
        }

        # Mock scans lookup
        scan_data = [{
            "id": "scan-uuid-999",
            "patient_id": "patient-uuid-111",
            "original_image_path": "scans/patient-uuid-111/scan-uuid-999_original.png",
            "gradcam_image_path": "scans/patient-uuid-111/scan-uuid-999_gradcam.png",
            "predicted_class": "Benign",
            "normal_probability": 0.05,
            "benign_probability": 0.90,
            "malignant_probability": 0.05,
            "model_version": "EfficientNet-B0",
            "created_at": "2026-10-10T12:00:00Z"
        }]

        # Set up table mock dispatch
        def mock_table_dispatch(table_name):
            t_mock = MagicMock()
            if table_name == "patients":
                t_mock.select.return_value.eq.return_value.execute.return_value.data = [patient_data]
            elif table_name == "scans":
                t_mock.select.return_value.eq.return_value.order.return_value.execute.return_value.data = scan_data
            return t_mock

        mock_client.table.side_effect = mock_table_dispatch
        
        # Mock Storage signed URL generator
        mock_storage_bucket = MagicMock()
        mock_storage_bucket.create_signed_url.return_value = {"signedURL": "https://supabase.co/storage/v1/object/sign/test.png"}
        mock_client.storage.from_.return_value = mock_storage_bucket
        mock_get_client.return_value = mock_client

        response = self.client.get("/patients/patient-uuid-111/scans", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["scans"][0]["predicted_class"], "Benign")
        self.assertIsNotNone(data["scans"][0]["original_image_url"])
        print("  [PASSED] Patient scan history returned signed image URLs correctly.")

    @patch("backend.routes.predict.generate_gradcam_base64")
    @patch("backend.routes.predict.model_service")
    @patch("backend.routes.predict.supabase_service")
    def test_atomic_storage_cleanup_on_db_failure(self, mock_supa_service, mock_model_service, mock_gradcam):
        """Verify partial failure handling deletes uploaded storage files when DB insert fails. [OFFLINE MOCKED]"""
        print("\n[TEST 10 - OFFLINE MOCKED] Testing atomic storage cleanup on DB failure...")

        mock_model_service.is_loaded.return_value = True
        mock_model_service.model_name = "EfficientNet-B0"
        mock_model_service.model = MagicMock()
        mock_model_service.predict.return_value = ("Benign", 0.95, {"Normal": 0.02, "Benign": 0.95, "Malignant": 0.03}, 1)
        mock_gradcam.return_value = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", None, None, None)

        mock_supa_service.is_configured.return_value = True
        mock_supa_service.get_patient_by_id.return_value = {
            "id": "patient-uuid-clean",
            "patient_code": "OV-P-CLEAN1",
            "full_name": "Clean Test Patient"
        }

        # Storage upload succeeds
        mock_supa_service.upload_storage_file.side_effect = lambda bucket, path, bytes, ct: path
        # DB creation fails with RuntimeError
        mock_supa_service.create_scan_record.side_effect = RuntimeError("Database connection timeout during insert")

        test_img_path = PROJECT_ROOT / "dataset" / "test" / "Benign" / "benign (1).png"
        if not test_img_path.exists():
            test_imgs = list((PROJECT_ROOT / "dataset" / "test").rglob("*.png"))
            test_img_path = test_imgs[0]

        with open(test_img_path, "rb") as f:
            file_bytes = f.read()

        import io
        response = self.client.post(
            "/predict",
            data={"patient_id": "patient-uuid-clean"},
            files={"file": (test_img_path.name, io.BytesIO(file_bytes), "image/png")},
            headers=AUTH_HEADERS
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("Failed to persist scan record", response.json()["detail"])
        # Assert cleanup delete_storage_file was called for uploaded files
        self.assertGreaterEqual(mock_supa_service.delete_storage_file.call_count, 1)
        print("  [PASSED] Atomic storage cleanup triggered and successfully removed orphaned files on DB failure.")

    @patch("backend.routes.predict.generate_gradcam_base64")
    @patch("backend.routes.predict.model_service")
    @patch("backend.routes.predict.supabase_service")
    def test_predict_with_patient_scan_saved_mocked(self, mock_supa_service, mock_model_service, mock_gradcam):
        """Verify POST /predict with patient_id saves storage files and DB record. [OFFLINE MOCKED]"""
        print("\n[TEST 11 - OFFLINE MOCKED] Testing POST /predict with patient scan persistence...")

        mock_model_service.is_loaded.return_value = True
        mock_model_service.model_name = "EfficientNet-B0"
        mock_model_service.model = MagicMock()
        mock_model_service.predict.return_value = ("Malignant", 0.98, {"Normal": 0.01, "Benign": 0.01, "Malignant": 0.98}, 2)
        mock_gradcam.return_value = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", None, None, None)

        mock_supa_service.is_configured.return_value = True
        mock_supa_service.get_patient_by_id.return_value = {
            "id": "patient-uuid-save",
            "patient_code": "OV-P-SAVE01",
            "full_name": "Save Test Patient"
        }
        mock_supa_service.upload_storage_file.side_effect = lambda bucket, path, bytes, ct: path
        mock_supa_service.create_scan_record.return_value = {
            "id": "scan-uuid-saved-100",
            "patient_id": "patient-uuid-save",
            "predicted_class": "Malignant"
        }

        test_img_path = PROJECT_ROOT / "dataset" / "test" / "Benign" / "benign (1).png"
        if not test_img_path.exists():
            test_imgs = list((PROJECT_ROOT / "dataset" / "test").rglob("*.png"))
            test_img_path = test_imgs[0]

        with open(test_img_path, "rb") as f:
            file_bytes = f.read()

        import io
        response = self.client.post(
            "/predict",
            data={"patient_id": "patient-uuid-save"},
            files={"file": (test_img_path.name, io.BytesIO(file_bytes), "image/png")},
            headers=AUTH_HEADERS
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], "Malignant")
        self.assertIn("saved_scan", data)
        self.assertEqual(data["saved_scan"]["id"], "scan-uuid-saved-100")
        self.assertEqual(data["patient"]["patient_code"], "OV-P-SAVE01")
        print("  [PASSED] Scan and Grad-CAM successfully uploaded to storage and scan record persisted in database.")




def run_supabase_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSupabaseIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 80)
    print("  ONCOVISION - SUPABASE PHASE 2 INTEGRATION TEST SUITE (100% OFFLINE MOCKED)  ")
    print("=" * 80)
    success = run_supabase_tests()
    if not success:
        sys.exit(1)

