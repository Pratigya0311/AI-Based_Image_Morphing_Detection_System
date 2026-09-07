"""Confidence scoring and dashboard-ready presentation of morph predictions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence, Union

GENUINE_LABEL = "Genuine"
MORPHED_LABEL = "Morphed"
CLASS_LABELS = (GENUINE_LABEL, MORPHED_LABEL)
ProbabilityInput = Union[Mapping[str, float], Sequence[float], Any]


class ResultPreparationError(ValueError):
    """Raised when classifier probabilities cannot produce a trustworthy result."""


@dataclass(frozen=True)
class AnalysisResult:
    """A complete, frontend-safe presentation of one morph-analysis result."""

    submission_id: str
    predicted_label: str
    confidence: float
    class_probabilities: dict[str, float]
    generated_at: str
    image_url: Optional[str] = None

    @property
    def confidence_percentage(self) -> float:
        """Return confidence as a percentage rounded for display."""
        return round(self.confidence * 100, 2)

    @property
    def presentation_status(self) -> str:
        """Return a semantic UI state for the results dashboard."""
        return "success" if self.predicted_label == GENUINE_LABEL else "warning"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON payload consumed by a results dashboard."""
        payload = asdict(self)
        payload["confidence_percentage"] = self.confidence_percentage
        payload["presentation_status"] = self.presentation_status
        return payload


class ConfidenceScorer:
    """Validates two-class model probabilities and selects the predicted class."""

    @staticmethod
    def _to_probability_mapping(probabilities: ProbabilityInput) -> dict[str, float]:
        if isinstance(probabilities, Mapping):
            if set(probabilities) != set(CLASS_LABELS):
                raise ResultPreparationError(
                    "Probability mappings must contain exactly 'Genuine' and 'Morphed'"
                )
            values = {label: probabilities[label] for label in CLASS_LABELS}
        else:
            # PyTorch and NumPy vectors expose tolist(), while ordinary Python
            # sequences remain valid without either dependency being imported.
            values_source = probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities
            if not isinstance(values_source, Sequence) or isinstance(values_source, (str, bytes)):
                raise ResultPreparationError(
                    "Probabilities must be a two-value sequence or a class-label mapping"
                )
            if len(values_source) != len(CLASS_LABELS):
                raise ResultPreparationError(
                    "Expected exactly two probabilities: [Genuine, Morphed]"
                )
            values = dict(zip(CLASS_LABELS, values_source))

        try:
            normalized = {label: float(value) for label, value in values.items()}
        except (TypeError, ValueError) as exc:
            raise ResultPreparationError("Probabilities must be numeric values") from exc

        if any(not math.isfinite(value) for value in normalized.values()):
            raise ResultPreparationError("Probabilities must be finite values")
        if any(value < 0 or value > 1 for value in normalized.values()):
            raise ResultPreparationError("Probabilities must be between 0 and 1")
        if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-6):
            raise ResultPreparationError("Class probabilities must sum to 1.0")
        return normalized

    def score(self, probabilities: ProbabilityInput) -> tuple[str, float, dict[str, float]]:
        """Return label, confidence, and validated class probabilities.

        Sequence inputs use the stable order ``[Genuine, Morphed]``.
        """
        normalized = self._to_probability_mapping(probabilities)
        predicted_label = max(CLASS_LABELS, key=normalized.__getitem__)
        return predicted_label, normalized[predicted_label], normalized


class ResultService:
    """Creates results ready for API serialization and dashboard rendering."""

    def __init__(self, scorer: Optional[ConfidenceScorer] = None) -> None:
        self.scorer = scorer or ConfidenceScorer()

    def create_result(
        self,
        submission_id: str,
        probabilities: ProbabilityInput,
        *,
        image_url: Optional[str] = None,
    ) -> AnalysisResult:
        """Convert Module 4 class probabilities into a dashboard result.

        Args:
            submission_id: Identifier assigned by Module 1.
            probabilities: Mapping by class label or [Genuine, Morphed] values
                from Module 4. Values must be normalized probabilities.
            image_url: Optional safe URL for displaying the submitted image.
        """
        if not isinstance(submission_id, str) or not submission_id.strip():
            raise ResultPreparationError("A non-empty submission ID is required")
        predicted_label, confidence, normalized = self.scorer.score(probabilities)
        return AnalysisResult(
            submission_id=submission_id,
            predicted_label=predicted_label,
            confidence=confidence,
            class_probabilities=normalized,
            generated_at=datetime.now(timezone.utc).isoformat(),
            image_url=image_url,
        )
