"""Tests for model-training configuration and dataset validation."""

import pytest
from pathlib import Path

from app.modules.inference import CLASS_LABELS
from app.modules.inference.training import (
    TrainingConfig,
    TrainingConfigurationError,
    class_index_to_label,
    validate_dataset_layout,
)


def _create_dataset_layout(root):
    for split in ("train", "val", "test"):
        for class_name in ("genuine", "morphed"):
            (root / split / class_name).mkdir(parents=True, exist_ok=True)


def test_validates_complete_dataset_layout():
    root = Path("tests/inference/.tmp_complete_dataset")
    _create_dataset_layout(root)

    splits = validate_dataset_layout(root)

    assert set(splits) == {"train", "val", "test"}


def test_rejects_incomplete_dataset_layout():
    root = Path("tests/inference/.tmp_incomplete_dataset")
    (root / "train" / "genuine").mkdir(parents=True, exist_ok=True)

    with pytest.raises(TrainingConfigurationError):
        validate_dataset_layout(root)


def test_validates_expected_class_index_mapping():
    mapping = class_index_to_label({"genuine": 0, "morphed": 1})

    assert mapping == {0: CLASS_LABELS[0], 1: CLASS_LABELS[1]}


@pytest.mark.parametrize(
    "config_kwargs",
    [{"epochs": 0}, {"batch_size": 0}, {"learning_rate": 0}, {"num_workers": -1}],
)
def test_rejects_invalid_training_configuration(config_kwargs):
    with pytest.raises(TrainingConfigurationError):
        TrainingConfig(**config_kwargs)
