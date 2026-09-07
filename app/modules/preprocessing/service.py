"""Face detection, alignment, normalization, and tensor conversion."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from app.core.exceptions import (
    InvalidImageDataError,
    LandmarkExtractionError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    PreprocessingError,
)
from app.schemas.contracts import FaceBoundingBox, PreprocessedImage, ValidatedImage


Landmarks = dict[str, tuple[float, float]]


class FaceDetector(Protocol):
    """Detect faces from a BGR OpenCV image."""

    def detect(self, image: np.ndarray) -> Sequence[FaceBoundingBox]: ...


class LandmarkExtractor(Protocol):
    """Extract named facial landmarks for one detected face."""

    def extract(self, image: np.ndarray, face: FaceBoundingBox) -> Landmarks: ...


class TensorFactory(Protocol):
    """Convert a CHW float32 NumPy array into the runtime tensor type."""

    def create(self, values: np.ndarray) -> object: ...


class PytorchTensorFactory:
    """Production tensor factory; imported lazily to keep errors actionable."""

    def create(self, values: np.ndarray) -> object:
        try:
            import torch
        except ImportError as error:
            raise PreprocessingError(
                "PyTorch is required to create a model-ready tensor. Install the project dependencies."
            ) from error
        return torch.from_numpy(np.ascontiguousarray(values))


class OpenCVHaarFaceDetector:
    """CPU-friendly OpenCV Haar-cascade face detector for the initial baseline."""

    def __init__(self, cascade_path: str | Path | None = None) -> None:
        path = Path(cascade_path) if cascade_path else Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(str(path))
        if self._classifier.empty():
            raise ValueError(f"could not load face cascade: {path}")

    def detect(self, image: np.ndarray) -> Sequence[FaceBoundingBox]:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._classifier.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5)
        return [FaceBoundingBox(int(x), int(y), int(width), int(height)) for x, y, width, height in faces]


class OpenCVLBFLandmarkExtractor:
    """68-point landmark adapter backed by OpenCV's LBF facemark model."""

    def __init__(self, model_path: str | Path) -> None:
        if not hasattr(cv2, "face"):
            raise ValueError("OpenCV facemark support requires opencv-contrib-python.")
        self._facemark = cv2.face.createFacemarkLBF()
        self._facemark.loadModel(str(model_path))

    def extract(self, image: np.ndarray, face: FaceBoundingBox) -> Landmarks:
        faces = np.asarray([[face.x, face.y, face.width, face.height]], dtype=np.int32)
        success, landmarks = self._facemark.fit(image, faces)
        if not success or len(landmarks) != 1:
            raise LandmarkExtractionError("Facial landmarks could not be extracted.")

        points = np.asarray(landmarks[0], dtype=np.float32).reshape(-1, 2)
        if len(points) < 48:
            raise LandmarkExtractionError("Facial landmarks could not be extracted.")

        left_eye_values = np.mean(points[36:42], axis=0)
        right_eye_values = np.mean(points[42:48], axis=0)
        left_eye = (float(left_eye_values[0]), float(left_eye_values[1]))
        right_eye = (float(right_eye_values[0]), float(right_eye_values[1]))
        return {"left_eye": left_eye, "right_eye": right_eye}


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    output_size: tuple[int, int] = (224, 224)
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    crop_margin: float = 0.2
    reject_multiple_faces: bool = True

    def __post_init__(self) -> None:
        if len(self.output_size) != 2 or min(self.output_size) <= 0:
            raise ValueError("output_size values must be positive")
        if not 0 <= self.crop_margin < 1:
            raise ValueError("crop_margin must be in [0, 1)")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("normalization statistics must contain RGB values")
        if any(value <= 0 for value in self.normalization_std):
            raise ValueError("normalization standard deviations must be positive")


class FacePreprocessor:
    """Coordinates Module 2 without knowing about feature extraction or inference."""

    def __init__(
        self,
        detector: FaceDetector,
        landmark_extractor: LandmarkExtractor,
        *,
        config: PreprocessingConfig | None = None,
        tensor_factory: TensorFactory | None = None,
    ) -> None:
        self._detector = detector
        self._landmark_extractor = landmark_extractor
        self._config = config or PreprocessingConfig()
        self._tensor_factory = tensor_factory or PytorchTensorFactory()

    def process(self, validated_image: ValidatedImage) -> PreprocessedImage:
        """Return one aligned, normalized face tensor for a validated image."""
        image = self._decode_image(validated_image)
        faces = list(self._detector.detect(image))
        if not faces:
            raise NoFaceDetectedError(
                "No detectable face was found in the uploaded image.",
                submission_id=validated_image.submission_id,
            )
        if len(faces) > 1 and self._config.reject_multiple_faces:
            raise MultipleFacesDetectedError(
                "Multiple faces were detected; submit an image containing one face.",
                submission_id=validated_image.submission_id,
            )

        face = self._validate_face(faces[0], image.shape, validated_image.submission_id)
        landmarks = self._landmark_extractor.extract(image, face)
        left_eye, right_eye = self._validate_eye_landmarks(landmarks, validated_image.submission_id)
        aligned_face = self._align_and_crop(image, face, left_eye, right_eye)
        normalized_chw = self._normalize(aligned_face)

        return PreprocessedImage(
            submission_id=validated_image.submission_id,
            tensor=self._tensor_factory.create(normalized_chw),
            face_bounding_box=face,
            source_dimensions=(image.shape[1], image.shape[0]),
            preprocessing_metadata={
                "output_size": self._config.output_size,
                "color_space": "RGB",
                "normalization": {"mean": self._config.normalization_mean, "std": self._config.normalization_std},
                "alignment_landmarks": ("left_eye", "right_eye"),
            },
        )

    @staticmethod
    def _decode_image(validated_image: ValidatedImage) -> np.ndarray:
        try:
            raw = validated_image.image_bytes
            if raw is None:
                raw = validated_image.image_path.read_bytes()  # type: ignore[union-attr]
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, TypeError) as error:
            raise InvalidImageDataError("The validated image could not be read.", submission_id=validated_image.submission_id) from error
        if image is None:
            raise InvalidImageDataError("The validated image could not be decoded.", submission_id=validated_image.submission_id)
        return image

    @staticmethod
    def _validate_face(face: FaceBoundingBox, shape: tuple[int, ...], submission_id: str) -> FaceBoundingBox:
        image_height, image_width = shape[:2]
        if face.x < 0 or face.y < 0 or face.x + face.width > image_width or face.y + face.height > image_height:
            raise PreprocessingError("Detected face is outside image bounds.", submission_id=submission_id)
        return face

    @staticmethod
    def _validate_eye_landmarks(landmarks: Landmarks, submission_id: str) -> tuple[tuple[float, float], tuple[float, float]]:
        try:
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
        except KeyError as error:
            raise LandmarkExtractionError("Required eye landmarks could not be extracted.", submission_id=submission_id) from error
        if left_eye == right_eye:
            raise LandmarkExtractionError("Eye landmarks must be distinct.", submission_id=submission_id)
        return left_eye, right_eye

    def _align_and_crop(
        self,
        image: np.ndarray,
        face: FaceBoundingBox,
        left_eye: tuple[float, float],
        right_eye: tuple[float, float],
    ) -> np.ndarray:
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        angle = degrees(atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        transform = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(image, transform, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

        margin_x = round(face.width * self._config.crop_margin)
        margin_y = round(face.height * self._config.crop_margin)
        left, top = max(face.x - margin_x, 0), max(face.y - margin_y, 0)
        right, bottom = min(face.x + face.width + margin_x, image.shape[1]), min(face.y + face.height + margin_y, image.shape[0])
        crop = aligned[top:bottom, left:right]
        if crop.size == 0:
            raise PreprocessingError("Face crop is empty after alignment.")
        return cv2.resize(crop, self._config.output_size, interpolation=cv2.INTER_AREA)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray(self._config.normalization_mean, dtype=np.float32)
        std = np.asarray(self._config.normalization_std, dtype=np.float32)
        return np.transpose((rgb - mean) / std, (2, 0, 1)).astype(np.float32)
