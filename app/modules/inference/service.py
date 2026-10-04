"""Checkpoint-backed morph classification and inference over face embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
from torch import Tensor, nn

from app.modules.feature_extraction import EMBEDDING_DIMENSION

GENUINE_LABEL = "Genuine"
MORPHED_LABEL = "Morphed"
CLASS_LABELS = (GENUINE_LABEL, MORPHED_LABEL)
NUMBER_OF_CLASSES = len(CLASS_LABELS)
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "checkpoints" / "morph_classifier.pt"
)


class ModelInferenceError(RuntimeError):
    """Raised when a model checkpoint or inference input is invalid."""


class MorphClassifier(nn.Module):
    """Binary classifier head trained on Module 3 ResNet-50 embeddings.

    The model returns logits in the fixed order ``[Genuine, Morphed]``. Softmax
    conversion belongs in :class:`MorphInferenceService` so training can use
    the raw logits with ``CrossEntropyLoss``.
    """

    def __init__(
        self,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        hidden_dimension: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if embedding_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        # The ``classifier`` attribute matches the trained project checkpoint.
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, NUMBER_OF_CLASSES),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        """Return logits shaped ``[batch, 2]``."""
        return self.classifier(embeddings)


@dataclass(frozen=True)
class InferenceResult:
    """One binary classifier result that can be passed directly to Module 5."""

    probabilities: dict[str, float]

    @property
    def predicted_label(self) -> str:
        return max(CLASS_LABELS, key=self.probabilities.__getitem__)


class MorphInferenceService:
    """Loads a trained morph classifier and converts embeddings into probabilities.

    Args:
        checkpoint_path: Optional path to a trusted, locally stored checkpoint.
            The checkpoint may be a raw state dictionary or a dictionary with a
            ``model_state_dict`` key. If supplied, it is loaded during startup.
        device: PyTorch device; ``None`` selects CUDA when available.
        model: Optional injected classifier, primarily for tests or controlled
            deployments with a custom trained architecture.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = DEFAULT_CHECKPOINT_PATH,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model or MorphClassifier()
        self._model_is_loaded = model is not None
        self.model.to(self.device)
        # An explicitly injected model is already the caller's loaded model;
        # do not overwrite it with the default production checkpoint.
        if checkpoint_path is not None and model is None:
            self.load_checkpoint(checkpoint_path)
        self._set_inference_mode()

    def _set_inference_mode(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _validate_embeddings(embeddings: Tensor) -> Tensor:
        if not isinstance(embeddings, Tensor):
            raise ModelInferenceError("Expected a torch.Tensor of feature embeddings")
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ModelInferenceError(
                f"Expected embedding shape [batch, {EMBEDDING_DIMENSION}]; "
                f"received {tuple(embeddings.shape)}"
            )
        if not embeddings.is_floating_point():
            raise ModelInferenceError("Feature embeddings must be floating-point tensors")
        if not torch.isfinite(embeddings).all():
            raise ModelInferenceError("Feature embeddings contain NaN or infinite values")
        return embeddings

    @staticmethod
    def _state_dict_from_checkpoint(checkpoint: Any) -> Mapping[str, Tensor]:
        if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, Mapping) or not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in state_dict.items()
        ):
            raise ModelInferenceError(
                "Checkpoint must contain a PyTorch model state dictionary"
            )
        return state_dict

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load trusted classifier weights from disk and reset inference mode."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise ModelInferenceError(f"Model checkpoint was not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(self._state_dict_from_checkpoint(checkpoint))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelInferenceError(f"Could not load model checkpoint: {path}") from exc
        self._model_is_loaded = True
        self._set_inference_mode()

    def predict_probabilities(self, embeddings: Tensor) -> Tensor:
        """Return CPU class probabilities shaped ``[batch, 2]``.

        Column order is fixed as ``[Genuine, Morphed]`` for Module 5.
        """
        if not self._model_is_loaded:
            raise ModelInferenceError(
                "No trained model checkpoint has been loaded for inference"
            )
        validated_embeddings = self._validate_embeddings(embeddings)
        with torch.inference_mode():
            logits = self.model(validated_embeddings.to(self.device, dtype=torch.float32))
            if logits.ndim != 2 or logits.shape != (
                validated_embeddings.shape[0],
                NUMBER_OF_CLASSES,
            ):
                raise ModelInferenceError(
                    "Classifier returned invalid logits shape: " f"{tuple(logits.shape)}"
                )
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu()

    def predict_one(self, embedding: Tensor) -> InferenceResult:
        """Return labelled probabilities for one embedding, ready for Module 5."""
        probabilities = self.predict_probabilities(embedding)
        if probabilities.shape[0] != 1:
            raise ModelInferenceError("predict_one accepts exactly one feature embedding")
        return InferenceResult(
            probabilities={
                label: float(probability)
                for label, probability in zip(CLASS_LABELS, probabilities[0].tolist())
            }
        )
