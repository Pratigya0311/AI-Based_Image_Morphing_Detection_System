"""Face detection, eye alignment, and ResNet-50-ready preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from app.modules.feature_extraction.extractor import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_HEIGHT,
    INPUT_WIDTH,
)


class PreprocessingError(ValueError):
    """Base error for a failed Module 2 preprocessing operation."""


class InvalidImageError(PreprocessingError):
    """The supplied bytes cannot be decoded as an image."""


class NoFaceDetectedError(PreprocessingError):
    """No usable face was detected."""


class MultipleFacesDetectedError(PreprocessingError):
    """More than one face was detected in an image requiring one face."""


class LandmarkDetectionError(PreprocessingError):
    """Eye landmarks required for alignment could not be detected."""


class LowImageQualityError(PreprocessingError):
    """The detected face is too blurry for reliable morph analysis."""


@dataclass(frozen=True)
class FaceBoundingBox:
    """A face rectangle in source-image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Face bounding-box dimensions must be positive")


@dataclass(frozen=True)
class PreprocessedFace:
    """Module 2 output that is directly accepted by Module 3."""

    submission_id: str
    tensor: Tensor
    face_bounding_box: FaceBoundingBox
    eye_landmarks: tuple[tuple[float, float], tuple[float, float]]
    source_dimensions: tuple[int, int]


class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> Sequence[FaceBoundingBox]: ...


class EyeLandmarkDetector(Protocol):
    def detect(
        self, image: np.ndarray, face: FaceBoundingBox
    ) -> tuple[tuple[float, float], tuple[float, float]]: ...


class HaarFaceDetector:
    """OpenCV Haar-cascade detector used as the CPU baseline."""

    def __init__(self, cascade_path: Optional[str | Path] = None) -> None:
        path = Path(cascade_path) if cascade_path else Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(str(path))
        if self.classifier.empty():
            raise RuntimeError(f"Could not load face detector cascade: {path}")

    def detect(self, image: np.ndarray) -> Sequence[FaceBoundingBox]:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.classifier.detectMultiScale(
            grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return [FaceBoundingBox(*map(int, detection)) for detection in detections]


class LBFLandmarkDetector:
    """Uses OpenCV's 68-point LBF facemark model for eye-based alignment."""

    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "LBF landmarks require opencv-contrib-python; install project dependencies."
            )
        path = (
            Path(model_path)
            if model_path
            else Path(__file__).resolve().parents[3] / "models" / "landmarks" / "lbfmodel.yaml"
        )
        if not path.is_file():
            raise RuntimeError(f"LBF landmark model was not found: {path}")
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(path))

    def detect(
        self, image: np.ndarray, face: FaceBoundingBox
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        faces = np.asarray([[face.x, face.y, face.width, face.height]], dtype=np.int32)
        success, landmarks = self.facemark.fit(image, faces)
        if not success or len(landmarks) != 1:
            raise LandmarkDetectionError("68-point facial landmarks could not be detected")
        points = np.asarray(landmarks[0], dtype=np.float32).reshape(-1, 2)
        if len(points) < 48:
            raise LandmarkDetectionError("The LBF model did not return eye landmarks")
        left_values = np.mean(points[36:42], axis=0)
        right_values = np.mean(points[42:48], axis=0)
        left_eye = (float(left_values[0]), float(left_values[1]))
        right_eye = (float(right_values[0]), float(right_values[1]))
        if left_eye == right_eye:
            raise LandmarkDetectionError("Detected eye landmarks are not distinct")
        return left_eye, right_eye


class FacePreprocessor:
    """Converts one validated image into a Module 3-ready face tensor."""

    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        landmark_detector: Optional[EyeLandmarkDetector] = None,
        *,
        output_size: tuple[int, int] = (INPUT_WIDTH, INPUT_HEIGHT),
        crop_margin: float = 0.2,
        reject_multiple_faces: bool = True,
        min_sharpness: float = 20.0,
    ) -> None:
        if output_size != (INPUT_WIDTH, INPUT_HEIGHT):
            raise ValueError("Module 3 requires preprocessing output size (224, 224)")
        if not 0 <= crop_margin < 1:
            raise ValueError("crop_margin must be in [0, 1)")
        if min_sharpness < 0:
            raise ValueError("min_sharpness must be non-negative")
        self.face_detector = face_detector or HaarFaceDetector()
        self.landmark_detector = landmark_detector or LBFLandmarkDetector()
        self.output_size = output_size
        self.crop_margin = crop_margin
        self.reject_multiple_faces = reject_multiple_faces
        self.min_sharpness = min_sharpness

    def process(self, submission_id: str, image_bytes: bytes) -> PreprocessedFace:
        """Detect, align, crop, resize, and normalize a single facial image."""
        if not isinstance(submission_id, str) or not submission_id.strip():
            raise PreprocessingError("A non-empty submission ID is required")
        image = self._decode(image_bytes)
        faces = list(self.face_detector.detect(image))
        if not faces:
            raise NoFaceDetectedError("No detectable face was found in the image")
        if len(faces) > 1 and self.reject_multiple_faces:
            raise MultipleFacesDetectedError(
                "Multiple faces were detected; submit an image containing one face"
            )
        face = self._validate_face(faces[0], image)
        eyes = self.landmark_detector.detect(image, face)
        aligned_face = self._align_crop_and_resize(image, face, eyes)
        sharpness = self.sharpness_score(aligned_face)
        if sharpness < self.min_sharpness:
            raise LowImageQualityError(
                "Image quality is too low for reliable morph analysis. "
                "Please upload a sharper, well-lit, front-facing image."
            )
        return PreprocessedFace(
            submission_id=submission_id,
            tensor=self._to_tensor(aligned_face),
            face_bounding_box=face,
            eye_landmarks=eyes,
            source_dimensions=(image.shape[1], image.shape[0]),
        )

    @staticmethod
    def _decode(image_bytes: bytes) -> np.ndarray:
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise InvalidImageError("Image bytes are required")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("The image could not be decoded")
        return image

    @staticmethod
    def _validate_face(face: FaceBoundingBox, image: np.ndarray) -> FaceBoundingBox:
        image_height, image_width = image.shape[:2]
        if face.x < 0 or face.y < 0 or face.x + face.width > image_width or face.y + face.height > image_height:
            raise PreprocessingError("Detected face is outside image bounds")
        return face

    def _align_crop_and_resize(
        self,
        image: np.ndarray,
        face: FaceBoundingBox,
        eyes: tuple[tuple[float, float], tuple[float, float]],
    ) -> np.ndarray:
        left_eye, right_eye = eyes
        centre = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        angle = degrees(atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        transform = cv2.getRotationMatrix2D(centre, angle, 1.0)
        aligned = cv2.warpAffine(image, transform, (image.shape[1], image.shape[0]))

        margin_x = round(face.width * self.crop_margin)
        margin_y = round(face.height * self.crop_margin)
        left = max(face.x - margin_x, 0)
        top = max(face.y - margin_y, 0)
        right = min(face.x + face.width + margin_x, image.shape[1])
        bottom = min(face.y + face.height + margin_y, image.shape[0])
        crop = aligned[top:bottom, left:right]
        if crop.size == 0:
            raise PreprocessingError("Face crop is empty after alignment")
        return cv2.resize(crop, self.output_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _to_tensor(image: np.ndarray) -> Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
            IMAGENET_STD, dtype=np.float32
        )
        return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))

    @staticmethod
    def sharpness_score(image: np.ndarray) -> float:
        """Return variance-of-Laplacian sharpness for an aligned face crop.

        Low values indicate that edge detail has been lost through blur.  The
        score is deliberately measured after crop/alignment, so background
        detail cannot make a blurry face pass the quality gate.
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
