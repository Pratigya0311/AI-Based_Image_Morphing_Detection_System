"""Deep feature extraction with a pretrained ResNet-50 backbone."""

from .extractor import (
    EMBEDDING_DIMENSION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    FeatureExtractionError,
    ResNet50FeatureExtractor,
)

__all__ = [
    "EMBEDDING_DIMENSION",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "FeatureExtractionError",
    "ResNet50FeatureExtractor",
]
