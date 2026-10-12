"""Tests for HTTP access to the Modules 1 through 7 workflow."""

import io
import unittest

from PIL import Image

import app.api as acquisition_api
import app.api.analysis as analysis_api
from app import create_app
from app.modules.acquisition import ImageAcquisition
from app.modules.results import ResultService


class StubPipeline:
    """A deterministic pipeline used to verify HTTP wiring without ML loading."""

    def analyze_image(self, submission_id, image_bytes):
        assert image_bytes
        return ResultService().create_result(submission_id, [0.2, 0.8])


def png_bytes(colour: str = "blue") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), colour).save(buffer, format="PNG")
    return buffer.getvalue()


class TestAnalysisApi(unittest.TestCase):
    def setUp(self):
        self.original_acquisition = acquisition_api.acquisition
        self.original_pipeline = analysis_api.pipeline
        acquisition_api.acquisition = ImageAcquisition(persist=False)
        analysis_api.pipeline = StubPipeline()

        app = create_app()
        app.config["TESTING"] = True
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session["user"] = {"id": "test-user", "email": "test@example.com"}

    def tearDown(self):
        acquisition_api.acquisition = self.original_acquisition
        analysis_api.pipeline = self.original_pipeline

    def test_analyze_returns_submission_and_dashboard_ready_result(self):
        response = self.client.post(
            "/api/analysis/analyze",
            data={"image": (io.BytesIO(png_bytes()), "face.png")},
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertTrue(payload["submission"]["valid"])
        self.assertEqual(payload["result"]["predicted_label"], "Morphed")
        self.assertEqual(
            payload["result"]["submission_id"], payload["submission"]["submission_id"]
        )

    def test_analyze_rejects_invalid_uploads_before_pipeline_execution(self):
        response = self.client.post(
            "/api/analysis/analyze",
            data={"image": (io.BytesIO(b"not an image"), "face.png")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["submission"]["valid"])

    def test_batch_returns_individual_results_without_a_pdf_when_requested(self):
        response = self.client.post(
            "/api/analysis/batch",
            data={
                "images": [
                    (io.BytesIO(png_bytes("blue")), "first.png"),
                    (io.BytesIO(png_bytes("red")), "second.png"),
                ],
                "generate_report": "false",
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload["successful_count"], 2)
        self.assertEqual(payload["failed_count"], 0)
        self.assertIsNone(payload["report_path"])

    def test_submission_audit_endpoint_returns_submission_events(self):
        analysis_response = self.client.post(
            "/api/analysis/analyze",
            data={"image": (io.BytesIO(png_bytes()), "face.png")},
        )
        submission_id = analysis_response.get_json()["submission"]["submission_id"]

        response = self.client.get(f"/api/analysis/audit/submission/{submission_id}")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["events"])
        self.assertEqual(response.get_json()["events"][0]["submission_id"], submission_id)
