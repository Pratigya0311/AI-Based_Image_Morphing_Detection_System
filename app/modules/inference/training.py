"""Training support for the Module 4 morph-classifier head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from app.modules.inference.service import CLASS_LABELS

DATASET_CLASS_DIRECTORIES = {"genuine", "morphed"}


class TrainingConfigurationError(ValueError):
    """Raised when a training dataset or configuration is invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for training the classifier head on frozen ResNet-50 features."""

    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise TrainingConfigurationError("Epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise TrainingConfigurationError(
                "Learning rate must be positive and weight decay cannot be negative"
            )
        if self.num_workers < 0:
            raise TrainingConfigurationError("Number of workers cannot be negative")


def validate_dataset_layout(dataset_root: Path) -> Mapping[str, Path]:
    """Validate the expected ImageFolder layout and return split locations.

    Expected structure::

        dataset_root/
          train/{genuine,morphed}/
          val/{genuine,morphed}/
          test/{genuine,morphed}/
    """
    dataset_root = Path(dataset_root)
    split_paths = {split: dataset_root / split for split in ("train", "val", "test")}
    for split, split_path in split_paths.items():
        if not split_path.is_dir():
            raise TrainingConfigurationError(
                f"Missing '{split}' dataset directory: {split_path}"
            )
        classes = {path.name.lower() for path in split_path.iterdir() if path.is_dir()}
        if classes != DATASET_CLASS_DIRECTORIES:
            raise TrainingConfigurationError(
                f"'{split}' must contain exactly these class folders: "
                f"{sorted(DATASET_CLASS_DIRECTORIES)}"
            )
    return split_paths


def class_index_to_label(class_to_index: Mapping[str, int]) -> dict[int, str]:
    """Validate ImageFolder labels and convert indices to Module 4 labels."""
    normalized = {name.lower(): index for name, index in class_to_index.items()}
    if set(normalized) != DATASET_CLASS_DIRECTORIES:
        raise TrainingConfigurationError(
            "Dataset class folders must be named 'genuine' and 'morphed'"
        )
    return {
        normalized["genuine"]: CLASS_LABELS[0],
        normalized["morphed"]: CLASS_LABELS[1],
    }
