from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.exceptions import (
    InvalidImageDataError,
    LandmarkExtractionError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    PreprocessingError,
)
from app.modules.preprocessing.service import FacePreprocessor, OpenCVLBFLandmarkExtractor, PreprocessingConfig
from app.schemas.contracts import FaceBoundingBox, ValidatedImage


class FixedDetector:
    def __init__(self, faces: Sequence[FaceBoundingBox]) -> None:
        self.faces = faces

    def detect(self, image: np.ndarray) -> Sequence[FaceBoundingBox]:
        return self.faces


class FixedLandmarkExtractor:
    def extract(self, image: np.ndarray, face: FaceBoundingBox) -> dict[str, tuple[float, float]]:
        return {"left_eye": (45.0, 48.0), "right_eye": (75.0, 50.0)}


class NumpyTensorFactory:
    def create(self, values: np.ndarray) -> np.ndarray:
        return values


class MissingEyeLandmarks:
    def extract(self, image: np.ndarray, face: FaceBoundingBox) -> dict[str, tuple[float, float]]:
        return {"nose": (60.0, 60.0)}


class SameEyeLandmarks:
    def extract(self, image: np.ndarray, face: FaceBoundingBox) -> dict[str, tuple[float, float]]:
        return {"left_eye": (60.0, 60.0), "right_eye": (60.0, 60.0)}


class FailingTensorFactory:
    def create(self, values: np.ndarray) -> np.ndarray:
        raise RuntimeError("tensor conversion failed")


class FakeFacemark:
    def fit(self, image: np.ndarray, faces: np.ndarray):
        points = np.zeros((68, 1, 2), dtype=np.float32)
        points[36:42, 0, :] = (40.0, 50.0)
        points[42:48, 0, :] = (80.0, 50.0)
        return True, [points]


def make_validated_image(*, width: int = 120, height: int = 120) -> ValidatedImage:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    encoded, image_bytes = cv2.imencode(".png", image)
    assert encoded
    return ValidatedImage(
        submission_id="SUB-001",
        filename="face.png",
        image_format="PNG",
        width=width,
        height=height,
        image_bytes=image_bytes.tobytes(),
    )


def test_process_returns_normalized_chw_face_tensor() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(30, 30, 60, 60)]),
        FixedLandmarkExtractor(),
        config=PreprocessingConfig(output_size=(64, 64)),
        tensor_factory=NumpyTensorFactory(),
    )

    result = preprocessor.process(make_validated_image())

    assert result.submission_id == "SUB-001"
    assert result.tensor.shape == (3, 64, 64)
    assert result.tensor.dtype == np.float32
    assert result.face_bounding_box == FaceBoundingBox(30, 30, 60, 60)
    assert result.preprocessing_metadata["color_space"] == "RGB"


def test_process_rejects_image_with_no_face() -> None:
    preprocessor = FacePreprocessor(FixedDetector([]), FixedLandmarkExtractor(), tensor_factory=NumpyTensorFactory())

    with pytest.raises(NoFaceDetectedError) as error:
        preprocessor.process(make_validated_image())

    assert error.value.error_code == "NO_FACE_DETECTED"
    assert error.value.submission_id == "SUB-001"


def test_process_rejects_multiple_faces_by_default() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(10, 10, 30, 30), FaceBoundingBox(60, 60, 30, 30)]),
        FixedLandmarkExtractor(),
        tensor_factory=NumpyTensorFactory(),
    )

    with pytest.raises(MultipleFacesDetectedError) as error:
        preprocessor.process(make_validated_image())

    assert error.value.error_code == "MULTIPLE_FACES"


def test_process_rejects_missing_required_landmarks() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(30, 30, 60, 60)]),
        MissingEyeLandmarks(),
        tensor_factory=NumpyTensorFactory(),
    )

    with pytest.raises(LandmarkExtractionError) as error:
        preprocessor.process(make_validated_image())

    assert error.value.error_code == "LANDMARK_EXTRACTION_FAILED"


def test_process_rejects_undecodable_image_data() -> None:
    invalid_image = ValidatedImage(
        submission_id="SUB-INVALID",
        filename="broken.png",
        image_format="PNG",
        width=10,
        height=10,
        image_bytes=b"not an image",
    )
    preprocessor = FacePreprocessor(FixedDetector([]), FixedLandmarkExtractor(), tensor_factory=NumpyTensorFactory())

    with pytest.raises(InvalidImageDataError) as error:
        preprocessor.process(invalid_image)

    assert error.value.error_code == "INVALID_IMAGE_DATA"


def test_process_rejects_face_bounding_box_outside_image_bounds() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(90, 90, 40, 40)]),
        FixedLandmarkExtractor(),
        tensor_factory=NumpyTensorFactory(),
    )

    with pytest.raises(PreprocessingError) as error:
        preprocessor.process(make_validated_image())

    assert error.value.error_code == "PREPROCESSING_FAILED"


def test_process_handles_face_crop_near_image_border() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(0, 0, 40, 40)]),
        FixedLandmarkExtractor(),
        config=PreprocessingConfig(output_size=(32, 32), crop_margin=0.5),
        tensor_factory=NumpyTensorFactory(),
    )

    result = preprocessor.process(make_validated_image())

    assert result.tensor.shape == (3, 32, 32)
    assert result.face_bounding_box == FaceBoundingBox(0, 0, 40, 40)


def test_process_handles_very_small_face() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(10, 10, 8, 8)]),
        FixedLandmarkExtractor(),
        config=PreprocessingConfig(output_size=(16, 16)),
        tensor_factory=NumpyTensorFactory(),
    )

    result = preprocessor.process(make_validated_image())

    assert result.tensor.shape == (3, 16, 16)


def test_process_handles_unusual_image_dimensions() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(80, 20, 40, 40)]),
        FixedLandmarkExtractor(),
        config=PreprocessingConfig(output_size=(48, 48)),
        tensor_factory=NumpyTensorFactory(),
    )

    result = preprocessor.process(make_validated_image(width=240, height=80))

    assert result.source_dimensions == (240, 80)
    assert result.tensor.shape == (3, 48, 48)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"output_size": (0, 224)},
        {"crop_margin": 1.0},
        {"normalization_mean": (0.5, 0.5)},
        {"normalization_std": (0.2, 0.0, 0.2)},
    ],
)
def test_config_rejects_invalid_values(config_kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(**config_kwargs)


def test_process_rejects_same_eye_landmark_coordinates() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(30, 30, 60, 60)]),
        SameEyeLandmarks(),
        tensor_factory=NumpyTensorFactory(),
    )

    with pytest.raises(LandmarkExtractionError) as error:
        preprocessor.process(make_validated_image())

    assert error.value.error_code == "LANDMARK_EXTRACTION_FAILED"


def test_process_uses_first_face_when_multiple_faces_are_allowed() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(10, 10, 30, 30), FaceBoundingBox(70, 70, 30, 30)]),
        FixedLandmarkExtractor(),
        config=PreprocessingConfig(reject_multiple_faces=False),
        tensor_factory=NumpyTensorFactory(),
    )

    result = preprocessor.process(make_validated_image())

    assert result.face_bounding_box == FaceBoundingBox(10, 10, 30, 30)


def test_process_propagates_tensor_factory_failure() -> None:
    preprocessor = FacePreprocessor(
        FixedDetector([FaceBoundingBox(30, 30, 60, 60)]),
        FixedLandmarkExtractor(),
        tensor_factory=FailingTensorFactory(),
    )

    with pytest.raises(RuntimeError, match="tensor conversion failed"):
        preprocessor.process(make_validated_image())


def test_process_reads_image_from_path() -> None:
    image = make_validated_image()
    local_tmp_dir = Path("tests/preprocessing/.tmp")
    local_tmp_dir.mkdir(exist_ok=True)
    image_path = local_tmp_dir / "face.png"
    try:
        image_path.write_bytes(image.image_bytes)
        path_image = ValidatedImage(
            submission_id="SUB-PATH",
            filename="face.png",
            image_format="PNG",
            width=image.width,
            height=image.height,
            image_path=image_path,
        )
        preprocessor = FacePreprocessor(
            FixedDetector([FaceBoundingBox(30, 30, 60, 60)]),
            FixedLandmarkExtractor(),
            tensor_factory=NumpyTensorFactory(),
        )

        result = preprocessor.process(path_image)

        assert result.submission_id == "SUB-PATH"
        assert result.tensor.shape == (3, 224, 224)
    finally:
        image_path.unlink(missing_ok=True)
        local_tmp_dir.rmdir()


def test_lbf_landmark_extractor_returns_plain_float_eye_tuples() -> None:
    extractor = OpenCVLBFLandmarkExtractor.__new__(OpenCVLBFLandmarkExtractor)
    extractor._facemark = FakeFacemark()

    landmarks = extractor.extract(np.zeros((120, 120, 3), dtype=np.uint8), FaceBoundingBox(30, 30, 60, 60))

    assert landmarks == {"left_eye": (40.0, 50.0), "right_eye": (80.0, 50.0)}
