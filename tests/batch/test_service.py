"""Tests for Module 6 batch orchestration and PDF reporting."""

from pathlib import Path
import uuid

import pytest

from app.modules.batch import BatchImage, BatchProcessingError, BatchProcessor
from app.modules.results import ResultService
from app.modules.audit import AuditLogger


class StubPipeline:
    def analyze_image(self, submission_id, image_bytes):
        if image_bytes == b"bad":
            raise ValueError("No detectable face was found")
        return ResultService().create_result(submission_id, [0.1, 0.9])


class RecordingReportGenerator:
    def __init__(self):
        self.received = None
        self.output_path = None

    def generate(self, batch_result, output_path):
        self.received = batch_result
        self.output_path = output_path
        return output_path


def test_processes_each_image_and_preserves_individual_failures():
    reporter = RecordingReportGenerator()
    processor = BatchProcessor(StubPipeline(), reporter, report_directory="data/reports")

    result = processor.process(
        [BatchImage("valid.png", b"valid"), BatchImage("bad.png", b"bad")]
    )

    assert result.successful_count == 1
    assert result.failed_count == 1
    assert result.items[0].result.predicted_label == "Morphed"
    assert "No detectable face" in result.items[1].error
    assert result.report_path.endswith(".pdf")
    assert reporter.received.batch_id == result.batch_id


def test_can_process_without_generating_a_report():
    processor = BatchProcessor(StubPipeline(), report_directory="data/reports")

    result = processor.process([BatchImage("valid.png", b"valid")], generate_report=False)

    assert result.successful_count == 1
    assert result.report_path is None


@pytest.mark.parametrize("images", [[], [BatchImage(str(index), b"valid") for index in range(2)]])
def test_rejects_invalid_batch_sizes(images):
    processor = BatchProcessor(StubPipeline(), max_batch_size=1)

    with pytest.raises(BatchProcessingError):
        processor.process(images, generate_report=False)


def test_serializes_batch_result_for_an_api_response():
    processor = BatchProcessor(StubPipeline())
    result = processor.process([BatchImage("valid.png", b"valid")], generate_report=False)

    payload = result.to_dict()
    assert payload["successful_count"] == 1
    assert payload["items"][0]["result"]["predicted_label"] == "Morphed"


def test_writes_a_pdf_report():
    report_directory = Path("tests/batch/.tmp_reports")
    processor = BatchProcessor(StubPipeline(), report_directory=report_directory)
    try:
        result = processor.process([BatchImage("valid.png", b"valid")])

        report_path = Path(result.report_path)
        assert report_path.exists()
        assert report_path.read_bytes().startswith(b"%PDF")
    finally:
        for report_path in report_directory.glob("*.pdf"):
            report_path.unlink()
        report_directory.rmdir()


def test_records_batch_summary_in_audit_log():
    audit_path = Path(f"tests/batch/.tmp_batch_audit_{uuid.uuid4().hex}.db")
    try:
        audit_logger = AuditLogger(audit_path)
        processor = BatchProcessor(StubPipeline(), audit_logger=audit_logger)

        result = processor.process([BatchImage("valid.png", b"valid")], generate_report=False)

        event = audit_logger.list_events(batch_id=result.batch_id)[0]
        assert event.event_type == "batch_completed"
        assert event.metadata["successful_count"] == 1
    finally:
        audit_path.unlink(missing_ok=True)
