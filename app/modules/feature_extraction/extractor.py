"""ResNet-50 feature extraction for preprocessed facial images.

The module contract with preprocessing is a float32 RGB tensor of shape
``[batch, 3, 224, 224]``, normalized using ImageNet mean and standard
deviation.  It returns one 2048-dimensional embedding for each image.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

INPUT_HEIGHT = 224
INPUT_WIDTH = 224
INPUT_CHANNELS = 3
EMBEDDING_DIMENSION = 2048
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FeatureExtractionError(ValueError):
    """Raised when a tensor does not satisfy the feature-extraction contract."""


class ResNet50FeatureExtractor:
    """Extract frozen 2048-dimensional ResNet-50 embeddings.

    Args:
        pretrained: Load ImageNet-pretrained ResNet-50 weights. This is enabled
            by default for the project pipeline. Set to ``False`` only in tests
            or when a separate trained checkpoint will be loaded later.
        device: PyTorch device selection. ``None`` uses CUDA when available,
            otherwise CPU.
        model: Optional injected ResNet-50 model, intended for controlled tests.
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pretrained = pretrained
        self.model = model or self._build_model(pretrained)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _build_model(pretrained: bool) -> nn.Module:
        try:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
        except Exception as exc:  # Download/cache errors depend on the runtime.
            if pretrained:
                raise RuntimeError(
                    "Could not load pretrained ResNet-50 weights. Ensure internet access "
                    "for the initial download or place the weights in the PyTorch cache."
                ) from exc
            raise

        # ResNet's final fully connected layer produces classification logits;
        # replacing it with Identity exposes the pooled 2048-value embedding.
        backbone.fc = nn.Identity()
        return backbone

    @staticmethod
    def validate_input(images: Tensor) -> Tensor:
        """Validate and batch the Module 2 -> Module 3 tensor contract."""
        if not isinstance(images, Tensor):
            raise FeatureExtractionError("Expected a torch.Tensor as input")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise FeatureExtractionError(
                "Expected an image tensor with shape [batch, 3, 224, 224]"
            )
        if images.shape[1:] != (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH):
            raise FeatureExtractionError(
                "Expected image shape [batch, 3, 224, 224]; "
                f"received {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise FeatureExtractionError(
                "Expected a floating-point, ImageNet-normalized image tensor"
            )
        if not torch.isfinite(images).all():
            raise FeatureExtractionError("Image tensor contains NaN or infinite values")
        return images

    def extract(self, images: Tensor) -> Tensor:
        """Return CPU embeddings shaped ``[batch, 2048]`` without gradients."""
        validated_images = self.validate_input(images)
        with torch.inference_mode():
            embeddings = self.model(validated_images.to(self.device, dtype=torch.float32))
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise RuntimeError(
                "ResNet-50 returned an unexpected embedding shape: "
                f"{tuple(embeddings.shape)}"
            )
        return embeddings.detach().cpu()

    def extract_one(self, image: Tensor) -> Tensor:
        """Return one embedding shaped ``[2048]`` for a single image tensor."""
        return self.extract(image).squeeze(0)
