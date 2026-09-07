from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.contracts import ValidatedImage
from app.modules.preprocessing import (
    FacePreprocessor,
    OpenCVHaarFaceDetector,
    OpenCVLBFLandmarkExtractor,
)

image_path = Path("data/sample/test_face.jpg")
landmark_model_path = Path("models/landmarks/lbfmodel.yaml")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

if not landmark_model_path.exists():
    raise FileNotFoundError(f"Could not find landmark model: {landmark_model_path}")

height, width = image.shape[:2]

validated = ValidatedImage(
    submission_id="REAL-001",
    filename=image_path.name,
    image_format=image_path.suffix.lstrip(".").upper(),
    width=width,
    height=height,
    image_path=image_path,
)

preprocessor = FacePreprocessor(
    detector=OpenCVHaarFaceDetector(),
    landmark_extractor=OpenCVLBFLandmarkExtractor(landmark_model_path),
)

result = preprocessor.process(validated)

print("Submission:", result.submission_id)
print("Tensor shape:", result.tensor.shape)
print("Face box:", result.face_bounding_box)
print("Metadata:", result.preprocessing_metadata)
