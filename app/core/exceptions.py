"""Structured exceptions shared by application modules."""

from __future__ import annotations


class ApplicationError(Exception):
    """Base error that can be safely translated into an API response."""

    error_code = "APPLICATION_ERROR"

    def __init__(self, message: str, *, submission_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.submission_id = submission_id


class PreprocessingError(ApplicationError):
    """Base error raised while preparing a face for model inference."""

    error_code = "PREPROCESSING_FAILED"


class InvalidImageDataError(PreprocessingError):
    error_code = "INVALID_IMAGE_DATA"


class NoFaceDetectedError(PreprocessingError):
    error_code = "NO_FACE_DETECTED"


class MultipleFacesDetectedError(PreprocessingError):
    error_code = "MULTIPLE_FACES"


class LandmarkExtractionError(PreprocessingError):
    error_code = "LANDMARK_EXTRACTION_FAILED"

