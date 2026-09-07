"""Image acquisition, validation, and submission-metadata management."""

import hashlib
import io
import json
import os
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image

SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB


@dataclass
class SubmissionMetadata:
    """Metadata retained for a submitted image."""

    submission_id: str
    filename: str
    format: str
    size: int
    timestamp: str
    file_hash: str
    validation_status: str
    error_message: Optional[str] = None


class ImageValidator:
    """Validates uploaded images for size, format, and complete decodability."""

    def __init__(self, max_size: int = MAX_FILE_SIZE) -> None:
        self.max_size = max_size
        self.supported_formats = SUPPORTED_FORMATS

    def validate_size(self, file_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """Check that a file is present and does not exceed the configured limit."""
        file_size = len(file_bytes)
        if file_size == 0:
            return False, "File is empty"
        if file_size > self.max_size:
            return False, (
                f"File size {file_size} exceeds maximum limit of {self.max_size} bytes"
            )
        return True, None

    def get_image_format(self, file_bytes: bytes) -> str:
        """Return the format of an already validated image."""
        with Image.open(io.BytesIO(file_bytes)) as image:
            return image.format or "Unknown"

    def validate_format(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[bool, Optional[str]]:
        """Confirm the image is supported and can be completely decoded.

        Content, rather than a filename extension, is authoritative.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(file_bytes)) as image:
                    format_name = image.format
                    if format_name not in self.supported_formats:
                        supported = ", ".join(sorted(self.supported_formats))
                        return False, (
                            f"Unsupported format: {format_name}. Supported: {supported}"
                        )
                    image.verify()

                # ``verify`` checks the file structure. Reopen and load it so a
                # truncated image cannot pass validation.
                with Image.open(io.BytesIO(file_bytes)) as image:
                    image.load()
            return True, None
        except (
            Image.DecompressionBombError,
            Image.UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            return False, f"Invalid image file: {exc}"

    def validate_image(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[bool, Optional[str]]:
        """Run all validation checks for an uploaded file."""
        size_valid, size_error = self.validate_size(file_bytes)
        if not size_valid:
            return False, size_error
        return self.validate_format(file_bytes, filename)


class SubmissionManager:
    """Creates and retrieves submission metadata, optionally backed by JSON."""

    def __init__(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self.submissions: Dict[str, SubmissionMetadata] = {}
        self._load_submissions()

    def _load_submissions(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            with self.storage_path.open(encoding="utf-8") as file:
                records = json.load(file)
            self.submissions = {
                submission_id: SubmissionMetadata(**metadata)
                for submission_id, metadata in records.items()
            }
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            raise RuntimeError(
                f"Could not load submission metadata from {self.storage_path}"
            ) from exc

    def _persist_submissions(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.storage_path.with_suffix(".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        submission_id: asdict(metadata)
                        for submission_id, metadata in self.submissions.items()
                    },
                    file,
                    indent=2,
                    sort_keys=True,
                )
            os.replace(temporary_path, self.storage_path)
        except OSError as exc:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError("Could not persist submission metadata") from exc

    def generate_submission_id(self) -> str:
        """Generate a collision-resistant submission identifier."""
        return f"SUB_{uuid.uuid4().hex.upper()}"

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        """Compute a SHA-256 checksum for integrity and audit correlation."""
        return hashlib.sha256(file_bytes).hexdigest()

    def record_submission(self, metadata: SubmissionMetadata) -> SubmissionMetadata:
        """Store metadata and persist it when durable storage is configured."""
        self.submissions[metadata.submission_id] = metadata
        self._persist_submissions()
        return metadata

    def create_submission(self, file_bytes: bytes, filename: str) -> SubmissionMetadata:
        """Create metadata for a validated image submission."""
        metadata = SubmissionMetadata(
            submission_id=self.generate_submission_id(),
            filename=filename,
            format=ImageValidator().get_image_format(file_bytes),
            size=len(file_bytes),
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_hash=self.compute_file_hash(file_bytes),
            validation_status="valid",
        )
        return self.record_submission(metadata)

    def get_submission(self, submission_id: str) -> Optional[SubmissionMetadata]:
        """Retrieve metadata for a submission, if it exists."""
        return self.submissions.get(submission_id)

    def get_all_submissions(self) -> Dict[str, SubmissionMetadata]:
        """Return a shallow copy of all submissions."""
        return self.submissions.copy()

    def update_validation_status(
        self, submission_id: str, status: str, error_msg: Optional[str] = None
    ) -> bool:
        """Update and persist a submission's validation state."""
        metadata = self.get_submission(submission_id)
        if metadata is None:
            return False
        metadata.validation_status = status
        metadata.error_message = error_msg
        self._persist_submissions()
        return True


class ImageAcquisition:
    """Public entry point for processing a single image submission."""

    def __init__(
        self, storage_path: Optional[Union[str, Path]] = None, persist: bool = True
    ) -> None:
        default_path = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "metadata"
            / "submissions.json"
        )
        self.validator = ImageValidator()
        self.submission_manager = SubmissionManager(
            (storage_path or default_path) if persist else None
        )

    def process_image(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Validate an image and create an auditable submission record."""
        is_valid, error_message = self.validator.validate_image(file_bytes, filename)
        if not is_valid:
            metadata = SubmissionMetadata(
                submission_id=self.submission_manager.generate_submission_id(),
                filename=filename,
                format="Unknown",
                size=len(file_bytes),
                timestamp=datetime.now(timezone.utc).isoformat(),
                file_hash=self.submission_manager.compute_file_hash(file_bytes),
                validation_status="invalid",
                error_message=error_message,
            )
            self.submission_manager.record_submission(metadata)
            return {
                "submission_id": metadata.submission_id,
                "valid": False,
                "error": error_message,
                "metadata": asdict(metadata),
            }

        metadata = self.submission_manager.create_submission(file_bytes, filename)
        return {
            "submission_id": metadata.submission_id,
            "valid": True,
            "error": None,
            "metadata": asdict(metadata),
        }

    def get_submission_status(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """Return the status and metadata of a submission."""
        metadata = self.submission_manager.get_submission(submission_id)
        if metadata is None:
            return None
        return {
            "submission_id": metadata.submission_id,
            "status": metadata.validation_status,
            "metadata": asdict(metadata),
        }
