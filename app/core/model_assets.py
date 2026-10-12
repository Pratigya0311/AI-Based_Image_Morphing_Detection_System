"""Resolve model assets from local development files or the Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "morph_classifier.pt"
LANDMARK_MODEL = PROJECT_ROOT / "models" / "landmarks" / "lbfmodel.yaml"


class ModelAssetError(RuntimeError):
    """Raised when a required deployment model asset cannot be resolved."""


def _download_from_hub(filename: str) -> Path:
    """Download one pinned private/public model asset into Hugging Face's cache."""
    repository = os.getenv("HF_MODEL_REPO", "").strip()
    if not repository:
        raise ModelAssetError(
            "Model asset is not available locally. Set HF_MODEL_REPO and HF_TOKEN "
            "for a deployed private model repository."
        )
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repository,
            filename=filename,
            revision=os.getenv("HF_MODEL_REVISION", "main").strip() or "main",
            token=os.getenv("HF_TOKEN") or None,
        )
    except Exception as exc:
        raise ModelAssetError(
            f"Could not download required model asset '{filename}' from the configured model repository"
        ) from exc
    return Path(downloaded)


def classifier_checkpoint_path() -> Path:
    """Return the local checkpoint in development or its Hub-backed cached path."""
    configured = os.getenv("MODEL_CHECKPOINT_PATH", "").strip()
    path = Path(configured) if configured else CLASSIFIER_CHECKPOINT
    return path if path.is_file() else _download_from_hub("checkpoints/morph_classifier.pt")


def landmark_model_path() -> Path:
    """Return the local landmark model in development or its Hub-backed cached path."""
    configured = os.getenv("LANDMARK_MODEL_PATH", "").strip()
    path = Path(configured) if configured else LANDMARK_MODEL
    return path if path.is_file() else _download_from_hub("landmarks/lbfmodel.yaml")
