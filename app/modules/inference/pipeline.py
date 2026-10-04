"""The Module 3 → Module 4 → Module 5 morph-analysis handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

from torch import Tensor

from app.modules.feature_extraction import ResNet50FeatureExtractor
from app.modules.results import AnalysisResult, ResultService

from .service import DEFAULT_CHECKPOINT_PATH, MorphInferenceService


class FeatureExtractor(Protocol):
    """The single-image interface supplied by Module 3."""

    def extract_one(self, image: Tensor) -> Tensor: ...


class MorphAnalysisPipeline:
    """Connect preprocessed faces to a dashboard-ready morph-analysis result.

    The pipeline expects the exact Module 2 output contract accepted by Module 3:
    a normalized float32 RGB tensor of shape ``[3, 224, 224]``.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = DEFAULT_CHECKPOINT_PATH,
        *,
        feature_extractor: Optional[FeatureExtractor] = None,
        inference_service: Optional[MorphInferenceService] = None,
        result_service: Optional[ResultService] = None,
    ) -> None:
        self.feature_extractor = feature_extractor or ResNet50FeatureExtractor()
        self.inference_service = inference_service or MorphInferenceService(
            checkpoint_path=checkpoint_path
        )
        self.result_service = result_service or ResultService()

    def analyze(
        self,
        submission_id: str,
        preprocessed_image: Tensor,
        *,
        image_url: Optional[str] = None,
    ) -> AnalysisResult:
        """Extract features, classify them, and prepare a Module 5 result."""
        embedding = self.feature_extractor.extract_one(preprocessed_image)
        prediction = self.inference_service.predict_one(embedding)
        return self.result_service.create_result(
            submission_id, prediction.probabilities, image_url=image_url
        )
