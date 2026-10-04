"""Tests for the Module 3 → 4 → 5 analysis pipeline."""

from pathlib import Path
import uuid

import torch

from app.modules.feature_extraction import EMBEDDING_DIMENSION
from app.modules.inference import MorphAnalysisPipeline, MorphInferenceService
from app.modules.preprocessing import FaceBoundingBox, PreprocessedFace
from app.modules.audit import AuditLogger

from .test_service import FixedLogitClassifier


class FixedFeatureExtractor:
    def extract_one(self, image):
        assert image.shape == (3, 224, 224)
        return torch.zeros(EMBEDDING_DIMENSION, dtype=torch.float32)


class FixedPreprocessor:
    def process(self, submission_id, image_bytes):
        assert image_bytes == b"image"
        return PreprocessedFace(
            submission_id=submission_id,
            tensor=torch.zeros((3, 224, 224), dtype=torch.float32),
            face_bounding_box=FaceBoundingBox(0, 0, 100, 100),
            eye_landmarks=((20.0, 30.0), (70.0, 30.0)),
            source_dimensions=(100, 100),
        )


def test_connects_feature_extraction_inference_and_result_preparation():
    pipeline = MorphAnalysisPipeline(
        feature_extractor=FixedFeatureExtractor(),
        inference_service=MorphInferenceService(
            model=FixedLogitClassifier(), device="cpu"
        ),
    )

    result = pipeline.analyze(
        "SUB_PIPELINE", torch.zeros((3, 224, 224), dtype=torch.float32)
    )

    assert result.submission_id == "SUB_PIPELINE"
    assert result.predicted_label == "Morphed"
    assert result.confidence_percentage > 50


def test_connects_image_preprocessing_through_to_result_preparation():
    pipeline = MorphAnalysisPipeline(
        preprocessor=FixedPreprocessor(),
        feature_extractor=FixedFeatureExtractor(),
        inference_service=MorphInferenceService(
            model=FixedLogitClassifier(), device="cpu"
        ),
    )

    result = pipeline.analyze_image("SUB_IMAGE", b"image")

    assert result.submission_id == "SUB_IMAGE"
    assert result.predicted_label == "Morphed"


def test_records_successful_analysis_in_the_audit_log():
    audit_path = Path(f"tests/inference/.tmp_pipeline_audit_{uuid.uuid4().hex}.db")
    try:
        audit_logger = AuditLogger(audit_path)
        pipeline = MorphAnalysisPipeline(
            feature_extractor=FixedFeatureExtractor(),
            inference_service=MorphInferenceService(model=FixedLogitClassifier(), device="cpu"),
            audit_logger=audit_logger,
        )

        pipeline.analyze("SUB_AUDIT", torch.zeros((3, 224, 224), dtype=torch.float32))

        event = audit_logger.list_events(submission_id="SUB_AUDIT")[0]
        assert event.event_type == "analysis_completed"
        assert event.outcome == "success"
        assert event.prediction == "Morphed"
    finally:
        audit_path.unlink(missing_ok=True)
