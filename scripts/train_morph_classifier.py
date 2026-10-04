"""Train the Module 4 morph-classifier head on a labelled image dataset.

Usage:
    python scripts/train_morph_classifier.py --dataset data/raw/morph_dataset
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Permit direct execution with ``python scripts/train_morph_classifier.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.feature_extraction import IMAGENET_MEAN, IMAGENET_STD, ResNet50FeatureExtractor
from app.modules.inference import CLASS_LABELS, MorphClassifier
from app.modules.inference.training import (
    TrainingConfig,
    class_index_to_label,
    validate_dataset_layout,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the morph-classifier head")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root directory")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/morph_classifier.pt"),
        help="Output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _transforms(training: bool) -> transforms.Compose:
    operations: list[Any] = [transforms.Resize((224, 224))]
    if training:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transforms.Compose(operations)


def _build_loaders(dataset_root: Path, config: TrainingConfig) -> dict[str, DataLoader]:
    split_paths = validate_dataset_layout(dataset_root)
    image_datasets = {
        split: datasets.ImageFolder(path, transform=_transforms(split == "train"))
        for split, path in split_paths.items()
    }
    class_index_to_label(image_datasets["train"].class_to_idx)
    for split, image_dataset in image_datasets.items():
        if image_dataset.class_to_idx != image_datasets["train"].class_to_idx:
            raise ValueError(f"Class-to-index mapping differs in the '{split}' split")
        if not image_dataset:
            raise ValueError(f"The '{split}' split contains no images")
    return {
        split: DataLoader(
            image_dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, image_dataset in image_datasets.items()
    }


def _run_epoch(
    loader: DataLoader,
    extractor: ResNet50FeatureExtractor,
    classifier: MorphClassifier,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    classifier.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        embeddings = extractor.extract(images).to(device)
        labels = labels.to(device)
        if is_training:
            optimizer.zero_grad()
        logits = classifier(embeddings)
        loss = criterion(logits, labels)
        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.shape[0]
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.shape[0]

    return {"loss": total_loss / total, "accuracy": correct / total}


def main() -> None:
    arguments = parse_arguments()
    config = TrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
    )
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = _build_loaders(arguments.dataset, config)

    extractor = ResNet50FeatureExtractor(device=device)
    classifier = MorphClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        classifier.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_validation_accuracy = -1.0
    history = []
    for epoch in range(1, config.epochs + 1):
        training_metrics = _run_epoch(
            loaders["train"], extractor, classifier, criterion, device, optimizer
        )
        with torch.inference_mode():
            validation_metrics = _run_epoch(
                loaders["val"], extractor, classifier, criterion, device
            )
        entry = {"epoch": epoch, "train": training_metrics, "validation": validation_metrics}
        history.append(entry)
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train accuracy: {training_metrics['accuracy']:.4f} | "
            f"validation accuracy: {validation_metrics['accuracy']:.4f}"
        )

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            torch.save({"model_state_dict": classifier.state_dict()}, arguments.checkpoint)

    classifier.load_state_dict(torch.load(arguments.checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    with torch.inference_mode():
        test_metrics = _run_epoch(loaders["test"], extractor, classifier, criterion, device)
    summary = {
        "class_labels": CLASS_LABELS,
        "embedding_dimension": 2048,
        "device": str(device),
        "best_validation_accuracy": best_validation_accuracy,
        "test": test_metrics,
        "history": history,
    }
    metadata_path = arguments.checkpoint.with_suffix(".json")
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {arguments.checkpoint}")
    print(f"Saved metrics: {metadata_path}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
