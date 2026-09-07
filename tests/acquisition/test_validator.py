"""Unit tests for image acquisition and validation."""

import io
import unittest
from unittest.mock import patch

from PIL import Image

from app.modules.acquisition import (
    ImageAcquisition,
    ImageValidator,
    MAX_FILE_SIZE,
    SubmissionManager,
)


def png_bytes() -> bytes:
    """Produce a valid in-memory PNG for tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


class TestImageValidator(unittest.TestCase):
    def setUp(self):
        self.validator = ImageValidator()
        self.image_data = png_bytes()

    def test_accepts_supported_image(self):
        valid, error = self.validator.validate_image(self.image_data, "face.png")

        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_rejects_empty_and_oversized_files(self):
        valid, error = self.validator.validate_size(b"")
        self.assertFalse(valid)
        self.assertEqual(error, "File is empty")

        valid, error = self.validator.validate_size(b"x" * (MAX_FILE_SIZE + 1))
        self.assertFalse(valid)
        self.assertIn("exceeds maximum limit", error)

    def test_rejects_non_image_and_unsupported_image(self):
        valid, error = self.validator.validate_format(b"not an image", "face.png")
        self.assertFalse(valid)
        self.assertIn("Invalid image file", error)

        buffer = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buffer, format="GIF")
        valid, error = self.validator.validate_format(buffer.getvalue(), "face.gif")
        self.assertFalse(valid)
        self.assertIn("Unsupported format", error)

    def test_rejects_truncated_image(self):
        valid, error = self.validator.validate_format(self.image_data[:-10], "face.png")

        self.assertFalse(valid)
        self.assertIn("Invalid image file", error)


class TestSubmissionManager(unittest.TestCase):
    def setUp(self):
        self.manager = SubmissionManager()
        self.image_data = png_bytes()

    def test_creates_retrieves_and_updates_metadata(self):
        metadata = self.manager.create_submission(self.image_data, "face.png")

        self.assertTrue(metadata.submission_id.startswith("SUB_"))
        self.assertEqual(metadata.format, "PNG")
        self.assertEqual(metadata.size, len(self.image_data))
        self.assertEqual(self.manager.get_submission(metadata.submission_id), metadata)

        self.assertTrue(
            self.manager.update_validation_status(
                metadata.submission_id, "invalid", "test error"
            )
        )
        self.assertEqual(
            self.manager.get_submission(metadata.submission_id).error_message,
            "test error",
        )

    def test_persist_is_called_when_storage_is_configured(self):
        manager = SubmissionManager("data/metadata/submissions.json")
        with patch.object(manager, "_persist_submissions") as persist:
            manager.create_submission(self.image_data, "face.png")

        persist.assert_called_once()


class TestImageAcquisition(unittest.TestCase):
    def setUp(self):
        self.acquisition = ImageAcquisition(persist=False)
        self.image_data = png_bytes()

    def test_records_valid_and_invalid_submissions(self):
        success = self.acquisition.process_image(self.image_data, "face.png")
        failure = self.acquisition.process_image(b"invalid", "face.png")

        self.assertTrue(success["valid"])
        self.assertFalse(failure["valid"])
        self.assertEqual(failure["metadata"]["validation_status"], "invalid")
        self.assertEqual(
            self.acquisition.get_submission_status(success["submission_id"])["status"],
            "valid",
        )
        self.assertIsNone(self.acquisition.get_submission_status("not-found"))
