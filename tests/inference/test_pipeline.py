"""Tests for the Module 3 → 4 → 5 analysis pipeline."""

import torch

from app.modules.feature_extraction import EMBEDDING_DIMENSION
from app.modules.inference import MorphAnalysisPipeline, MorphInferenceService

from .test_service import FixedLogitClassifier


class FixedFeatureExtractor:
    def extract_one(self, image):
        assert image.shape == (3, 224, 224)
        return torch.zeros(EMBEDDING_DIMENSION, dtype=torch.float32)


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
