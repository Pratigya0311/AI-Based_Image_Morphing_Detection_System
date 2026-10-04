"""Batch orchestration and PDF reporting for completed morph analyses."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol, Sequence

from app.modules.results import AnalysisResult
from app.modules.audit import AuditLogger


class BatchProcessingError(ValueError):
    """Raised when a batch request itself is invalid."""


@dataclass(frozen=True)
class BatchImage:
    """One image item submitted for batch processing."""

    filename: str
    image_bytes: bytes
    submission_id: Optional[str] = None


@dataclass(frozen=True)
class BatchItemResult:
    """The success result or descriptive failure for one submitted image."""

    submission_id: str
    filename: str
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.result is not None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.result is not None:
            payload["result"] = self.result.to_dict()
        payload["succeeded"] = self.succeeded
        return payload


@dataclass(frozen=True)
class BatchResult:
    """Aggregate outcome and report location for a batch request."""

    batch_id: str
    created_at: str
    items: tuple[BatchItemResult, ...]
    report_path: Optional[str] = None

    @property
    def successful_count(self) -> int:
        return sum(item.succeeded for item in self.items)

    @property
    def failed_count(self) -> int:
        return len(self.items) - self.successful_count

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "report_path": self.report_path,
            "items": [item.to_dict() for item in self.items],
        }


class ImageAnalysisPipeline(Protocol):
    """The image-bytes analysis interface supplied by Modules 2 through 5."""

    def analyze_image(self, submission_id: str, image_bytes: bytes) -> AnalysisResult: ...


class PdfReportGenerator:
    """Creates a compact PDF report containing all individual batch outcomes."""

    def generate(self, batch_result: BatchResult, output_path: Path) -> Path:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
            from reportlab.lib import colors
        except ImportError as exc:
            raise RuntimeError("PDF reports require reportlab; install project dependencies.") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        styles = getSampleStyleSheet()
        document = SimpleDocTemplate(
            str(output_path), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm
        )
        story = [
            Paragraph("AI-Based Image Morphing Detection Report", styles["Title"]),
            Spacer(1, 6 * mm),
            Paragraph(f"Batch ID: {batch_result.batch_id}", styles["Normal"]),
            Paragraph(f"Generated: {batch_result.created_at}", styles["Normal"]),
            Paragraph(
                f"Successful: {batch_result.successful_count} | Failed: {batch_result.failed_count}",
                styles["Normal"],
            ),
            Spacer(1, 6 * mm),
        ]
        rows = [["Submission", "Filename", "Prediction", "Confidence", "Status/Error"]]
        for item in batch_result.items:
            if item.result:
                rows.append(
                    [
                        item.submission_id,
                        item.filename,
                        item.result.predicted_label,
                        f"{item.result.confidence_percentage:.2f}%",
                        "Processed",
                    ]
                )
            else:
                rows.append([item.submission_id, item.filename, "—", "—", item.error or "Failed"])
        table = Table(rows, repeatRows=1, colWidths=[32 * mm, 40 * mm, 27 * mm, 25 * mm, 48 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
                ]
            )
        )
        story.append(table)
        document.build(story)
        return output_path


class BatchProcessor:
    """Processes each image independently and optionally creates a PDF report."""

    def __init__(
        self,
        pipeline: ImageAnalysisPipeline,
        report_generator: Optional[PdfReportGenerator] = None,
        report_directory: Optional[str | Path] = None,
        max_batch_size: int = 50,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.pipeline = pipeline
        self.report_generator = report_generator or PdfReportGenerator()
        self.report_directory = Path(report_directory or Path("data") / "reports")
        self.max_batch_size = max_batch_size
        self.audit_logger = audit_logger or AuditLogger()

    def process(self, images: Sequence[BatchImage], *, generate_report: bool = True) -> BatchResult:
        """Process every image without allowing one failure to stop the batch."""
        if not images:
            raise BatchProcessingError("A batch must contain at least one image")
        if len(images) > self.max_batch_size:
            raise BatchProcessingError(
                f"Batch contains {len(images)} images; maximum is {self.max_batch_size}"
            )

        batch_id = f"BATCH_{uuid.uuid4().hex.upper()}"
        created_at = datetime.now(timezone.utc).isoformat()
        item_results = []
        for index, image in enumerate(images, start=1):
            submission_id = image.submission_id or f"{batch_id}_{index:03d}"
            try:
                result = self.pipeline.analyze_image(submission_id, image.image_bytes)
                item_results.append(BatchItemResult(submission_id, image.filename, result=result))
            except Exception as exc:
                item_results.append(
                    BatchItemResult(submission_id, image.filename, error=str(exc))
                )

        batch_result = BatchResult(batch_id, created_at, tuple(item_results))
        if generate_report:
            report_path = self.report_directory / f"{batch_id}.pdf"
            generated_path = self.report_generator.generate(batch_result, report_path)
            batch_result = BatchResult(batch_id, created_at, tuple(item_results), str(generated_path))
        self.audit_logger.record(
            "batch_completed", "success", batch_id=batch_id,
            metadata={
                "successful_count": batch_result.successful_count,
                "failed_count": batch_result.failed_count,
                "report_path": batch_result.report_path,
            },
        )
        return batch_result
