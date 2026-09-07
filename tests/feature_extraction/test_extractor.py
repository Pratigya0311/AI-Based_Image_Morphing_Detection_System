"""Tests for the ResNet-50 feature-extraction module."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from app.modules.feature_extraction import (
    EMBEDDING_DIMENSION,
    FeatureExtractionError,
    ResNet50FeatureExtractor,
)


class DeterministicBackbone(nn.Module):
    """Small test double that preserves the ResNet-50 output contract."""

    def forward(self, images):
        return torch.ones((images.shape[0], EMBEDDING_DIMENSION), device=images.device)


@pytest.fixture
def extractor():
    return ResNet50FeatureExtractor(model=DeterministicBackbone(), device="cpu")


def test_extracts_embedding_for_one_image(extractor):
    embedding = extractor.extract_one(torch.zeros((3, 224, 224), dtype=torch.float32))

    assert embedding.shape == (EMBEDDING_DIMENSION,)
    assert embedding.device.type == "cpu"
    assert not embedding.requires_grad


def test_extracts_embedding_for_a_batch(extractor):
    embeddings = extractor.extract(torch.zeros((2, 3, 224, 224), dtype=torch.float32))

    assert embeddings.shape == (2, EMBEDDING_DIMENSION)


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros((3, 100, 100), dtype=torch.float32),
        torch.zeros((1, 1, 224, 224), dtype=torch.float32),
        torch.zeros((1, 3, 224, 224), dtype=torch.int64),
    ],
)
def test_rejects_invalid_input_contract(extractor, images):
    with pytest.raises(FeatureExtractionError):
        extractor.extract(images)
