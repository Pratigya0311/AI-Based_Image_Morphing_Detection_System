"""Unit tests for Module 2 face preprocessing."""

import cv2
import numpy as np
import pytest
import torch

from app.modules.preprocessing import (
    FaceBoundingBox,
    FacePreprocessor,
    InvalidImageError,
    LandmarkDetectionError,
    LowImageQualityError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)


class FixedFaceDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return self.faces


class FixedEyeDetector:
    def detect(self, image, face):
        return (40.0, 50.0), (80.0, 55.0)


class FailingEyeDetector:
    def detect(self, image, face):
        raise LandmarkDetectionError("Eyes unavailable")


def image_bytes(width=120, height=120, *, blurred=False):
    """Create a textured image which represents a sufficiently sharp face crop."""
    y, x = np.indices((height, width))
    pattern = ((x * 17 + y * 31) % 256).astype(np.uint8)
    image = np.dstack((pattern, np.roll(pattern, 3, axis=0), np.roll(pattern, 5, axis=1)))
    if blurred:
        image = cv2.GaussianBlur(image, (21, 21), 0)
    encoded, values = cv2.imencode(".png", image)
    assert encoded
    return values.tobytes()


def test_process_returns_module_three_compatible_tensor():
    preprocessor = FacePreprocessor(
        FixedFaceDetector([FaceBoundingBox(30, 30, 60, 60)]), landmark_detector=FixedEyeDetector()
    )

    result = preprocessor.process("SUB_1", image_bytes())

    assert result.tensor.shape == (3, 224, 224)
    assert result.tensor.dtype == torch.float32
    assert result.source_dimensions == (120, 120)
    assert result.face_bounding_box == FaceBoundingBox(30, 30, 60, 60)


def test_rejects_invalid_image_and_missing_face():
    preprocessor = FacePreprocessor(FixedFaceDetector([]), landmark_detector=FixedEyeDetector())
    with pytest.raises(InvalidImageError):
        preprocessor.process("SUB_1", b"not an image")
    with pytest.raises(NoFaceDetectedError):
        preprocessor.process("SUB_1", image_bytes())


def test_rejects_multiple_faces_and_missing_eye_landmarks():
    multiple_faces = FacePreprocessor(
        FixedFaceDetector([FaceBoundingBox(1, 1, 50, 50), FaceBoundingBox(60, 60, 50, 50)]),
        landmark_detector=FixedEyeDetector(),
    )
    with pytest.raises(MultipleFacesDetectedError):
        multiple_faces.process("SUB_1", image_bytes())

    missing_eyes = FacePreprocessor(
        FixedFaceDetector([FaceBoundingBox(30, 30, 60, 60)]), landmark_detector=FailingEyeDetector()
    )
    with pytest.raises(LandmarkDetectionError):
        missing_eyes.process("SUB_1", image_bytes())


def test_rejects_face_outside_source_bounds():
    preprocessor = FacePreprocessor(
        FixedFaceDetector([FaceBoundingBox(100, 100, 30, 30)]), landmark_detector=FixedEyeDetector()
    )
    with pytest.raises(ValueError, match="outside image bounds"):
        preprocessor.process("SUB_1", image_bytes())


def test_rejects_blurry_face_crops_before_feature_extraction():
    preprocessor = FacePreprocessor(
        FixedFaceDetector([FaceBoundingBox(30, 30, 60, 60)]),
        landmark_detector=FixedEyeDetector(),
        min_sharpness=20.0,
    )

    with pytest.raises(LowImageQualityError, match="too low"):
        preprocessor.process("SUB_BLURRY", image_bytes(blurred=True))

    assert FacePreprocessor.sharpness_score(
        cv2.imdecode(np.frombuffer(image_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    ) > 20.0
