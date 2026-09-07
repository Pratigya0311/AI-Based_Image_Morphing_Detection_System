"""Unit tests for confidence scoring and dashboard result preparation."""

import pytest

from app.modules.results import (
    GENUINE_LABEL,
    MORPHED_LABEL,
    ConfidenceScorer,
    ResultPreparationError,
    ResultService,
)


def test_scores_sequence_probabilities_in_documented_class_order():
    label, confidence, probabilities = ConfidenceScorer().score([0.15, 0.85])

    assert label == MORPHED_LABEL
    assert confidence == 0.85
    assert probabilities == {GENUINE_LABEL: 0.15, MORPHED_LABEL: 0.85}


def test_scores_labelled_probability_mapping():
    label, confidence, probabilities = ConfidenceScorer().score(
        {MORPHED_LABEL: 0.1, GENUINE_LABEL: 0.9}
    )

    assert label == GENUINE_LABEL
    assert confidence == 0.9
    assert probabilities[GENUINE_LABEL] == 0.9


def test_creates_dashboard_ready_result_payload():
    result = ResultService().create_result(
        "SUB_123",
        [0.925, 0.075],
        image_url="/api/images/SUB_123",
    )

    payload = result.to_dict()
    assert payload["submission_id"] == "SUB_123"
    assert payload["predicted_label"] == GENUINE_LABEL
    assert payload["confidence_percentage"] == 92.5
    assert payload["presentation_status"] == "success"
    assert payload["image_url"] == "/api/images/SUB_123"
    assert payload["class_probabilities"] == {GENUINE_LABEL: 0.925, MORPHED_LABEL: 0.075}


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5],
        [0.7, 0.4],
        [-0.1, 1.1],
        [float("nan"), 1.0],
        {GENUINE_LABEL: 1.0},
        {GENUINE_LABEL: 0.5, MORPHED_LABEL: 0.4, "Other": 0.1},
    ],
)
def test_rejects_invalid_probability_inputs(probabilities):
    with pytest.raises(ResultPreparationError):
        ConfidenceScorer().score(probabilities)


def test_rejects_blank_submission_id():
    with pytest.raises(ResultPreparationError, match="submission ID"):
        ResultService().create_result(" ", [0.5, 0.5])


def test_marks_morphed_result_as_warning():
    result = ResultService().create_result("SUB_456", [0.01, 0.99])

    assert result.predicted_label == MORPHED_LABEL
    assert result.presentation_status == "warning"
