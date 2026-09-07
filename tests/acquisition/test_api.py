"""HTTP-level tests for image-acquisition endpoints."""

import io
import unittest

from PIL import Image

import app.api as acquisition_api
from app import create_app
from app.modules.acquisition import ImageAcquisition


class TestAcquisitionApi(unittest.TestCase):
    def setUp(self):
        self.original_acquisition = acquisition_api.acquisition
        acquisition_api.acquisition = ImageAcquisition(persist=False)
        app = create_app()
        app.config["TESTING"] = True
        self.client = app.test_client()

        image_bytes = io.BytesIO()
        Image.new("RGB", (20, 20), "red").save(image_bytes, format="PNG")
        self.image_data = image_bytes.getvalue()

    def tearDown(self):
        acquisition_api.acquisition = self.original_acquisition

    def test_upload_creates_retrievable_submission(self):
        response = self.client.post(
            "/api/acquisition/upload",
            data={"image": (io.BytesIO(self.image_data), "face.png")},
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["metadata"]["format"], "PNG")

        status_response = self.client.get(
            f"/api/acquisition/submission/{payload['submission_id']}"
        )
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.get_json()["status"], "valid")

    def test_upload_rejects_non_image_data(self):
        response = self.client.post(
            "/api/acquisition/upload",
            data={"image": (io.BytesIO(b"not an image"), "face.png")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["valid"])

    def test_upload_requires_image_field(self):
        response = self.client.post("/api/acquisition/upload")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "No image file provided")

    def test_upload_returns_json_when_request_is_too_large(self):
        response = self.client.post(
            "/api/acquisition/upload",
            data={"image": (io.BytesIO(b"x" * (11 * 1024 * 1024)), "large.png")},
        )

        self.assertEqual(response.status_code, 413)
        self.assertFalse(response.get_json()["valid"])
