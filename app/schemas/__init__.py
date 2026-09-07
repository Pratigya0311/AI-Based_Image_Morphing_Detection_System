"""Shared data-contract schemas."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ImageSubmission:
    """Schema for an image submitted to the pipeline."""

    file_bytes: bytes
    filename: str
    format: Optional[str] = None
    size: Optional[int] = None


@dataclass
class SubmissionResponse:
    """Schema returned after validating an image submission."""

    submission_id: str
    valid: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ValidationResult:
    """Schema for validation without creating a submission."""

    valid: bool
    error_message: Optional[str] = None
    format: Optional[str] = None
    size: Optional[int] = None


@dataclass
class BatchSubmission:
    """Schema for a future batch-processing request."""

    images: List[ImageSubmission]
    batch_id: Optional[str] = None
