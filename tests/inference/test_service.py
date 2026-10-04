"""Unit tests for morph-classification model loading and inference."""

import pytest
import torch
from pathlib import Path
from torch import nn

from app.modules.feature_extraction import EMBEDDING_DIMENSION
from app.modules.inference import (
    GENUINE_LABEL,
    MORPHED_LABEL,
    ModelInferenceError,
    MorphClassifier,
    MorphInferenceService,
)


class FixedLogitClassifier(nn.Module):
    """Test double with deterministic logits in [Genuine, Morphed] order."""

    def forward(self, embeddings):
        return torch.tensor([[1.0, 3.0]], device=embeddings.device).repeat(
            embeddings.shape[0], 1
        )


@pytest.fixture
def service():
    return MorphInferenceService(model=FixedLogitClassifier(), device="cpu")


def test_returns_normalized_probabilities_for_batch(service):
    probabilities = service.predict_probabilities(
        torch.zeros((2, EMBEDDING_DIMENSION), dtype=torch.float32)
    )

    assert probabilities.shape == (2, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert probabilities[:, 1].gt(probabilities[:, 0]).all()


def test_returns_labelled_single_prediction_ready_for_module_five(service):
    result = service.predict_one(torch.zeros(EMBEDDING_DIMENSION, dtype=torch.float32))

    assert set(result.probabilities) == {GENUINE_LABEL, MORPHED_LABEL}
    assert result.predicted_label == MORPHED_LABEL
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "embeddings",
    [
        torch.zeros((2, 10), dtype=torch.float32),
        torch.zeros((1, EMBEDDING_DIMENSION), dtype=torch.int64),
        torch.full((1, EMBEDDING_DIMENSION), float("nan")),
    ],
)
def test_rejects_invalid_embedding_contract(service, embeddings):
    with pytest.raises(ModelInferenceError):
        service.predict_probabilities(embeddings)


def test_rejects_multiple_embeddings_for_predict_one(service):
    with pytest.raises(ModelInferenceError, match="exactly one"):
        service.predict_one(torch.zeros((2, EMBEDDING_DIMENSION), dtype=torch.float32))


def test_loads_state_dict_checkpoint():
    source_model = MorphClassifier()
    checkpoint_path = Path("tests/inference/.tmp_morph_classifier.pt")
    try:
        torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)
        service = MorphInferenceService(checkpoint_path=checkpoint_path, device="cpu")
        probabilities = service.predict_probabilities(
            torch.zeros((1, EMBEDDING_DIMENSION), dtype=torch.float32)
        )

        assert probabilities.shape == (1, 2)
        assert not service.model.training
    finally:
        checkpoint_path.unlink(missing_ok=True)


def test_rejects_missing_checkpoint():
    with pytest.raises(ModelInferenceError, match="not found"):
        MorphInferenceService(checkpoint_path="models/checkpoints/missing.pt", device="cpu")


def test_rejects_inference_without_trained_weights():
    service = MorphInferenceService(checkpoint_path=None, device="cpu")

    with pytest.raises(ModelInferenceError, match="No trained model checkpoint"):
        service.predict_probabilities(torch.zeros((1, EMBEDDING_DIMENSION)))
