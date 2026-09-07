"""Stable data contracts used between application modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    """Image accepted by Module 1 and ready for downstream processing."""

    submission_id: str
    filename: str
    image_format: str
    width: int
    height: int
    image_bytes: bytes | None = None
    image_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.submission_id:
            raise ValueError("submission_id is required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if (self.image_bytes is None) == (self.image_path is None):
            raise ValueError("provide exactly one of image_bytes or image_path")


@dataclass(frozen=True, slots=True)
class FaceBoundingBox:
    """Face position in source-image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("face bounding-box dimensions must be positive")


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """Model-ready face tensor returned by Module 2."""

    submission_id: str
    tensor: Any
    face_bounding_box: FaceBoundingBox
    source_dimensions: tuple[int, int]
    preprocessing_metadata: dict[str, Any]

