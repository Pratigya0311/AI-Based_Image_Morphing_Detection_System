"""HTTP endpoints that expose the complete Modules 1 through 7 workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, request, send_file, session
from werkzeug.utils import secure_filename

from app.modules.audit import AuditLogger
from app.core.database import database
from app.modules.batch import BatchImage, BatchProcessingError, BatchProcessor
from app.modules.inference import ModelInferenceError, MorphAnalysisPipeline
from app.modules.preprocessing import PreprocessingError
from app.api.auth import login_required

analysis_bp = Blueprint("analysis", __name__, url_prefix="/api/analysis")
logger = logging.getLogger(__name__)

# Created only when an analysis endpoint is used. This keeps upload and health
# endpoints available when ML assets are still being installed on a deployment.
pipeline: Optional[MorphAnalysisPipeline] = None


def _get_pipeline() -> MorphAnalysisPipeline:
    global pipeline
    if pipeline is None:
        pipeline = MorphAnalysisPipeline()
    return pipeline


def _error_response(message: str, status_code: int, **details: object):
    return jsonify({"error": message, **details}), status_code


def _acquire(file_storage):
    """Run Module 1 and return its durable submission record plus file bytes."""
    # Import at request time so this module always uses the package's shared
    # acquisition service, including in application tests where it is replaced.
    from app.api import acquisition

    if not file_storage.filename:
        return None, _error_response("No file selected", 400)

    image_bytes = file_storage.read()
    try:
        submission = acquisition.process_image(
            image_bytes, secure_filename(file_storage.filename)
        )
    except RuntimeError:
        logger.exception("Failed to store image submission metadata")
        return None, _error_response("Unable to store submission metadata", 500)

    if not submission["valid"]:
        return None, _error_response(
            submission["error"] or "Image validation failed",
            400,
            submission=submission,
        )
    return (submission, image_bytes), None


def _run_analysis(submission: dict, image_bytes: bytes):
    """Run Modules 2–5 and return a frontend-safe analysis response."""
    try:
        result = _get_pipeline().analyze_image(
            submission["submission_id"], image_bytes
        )
    except PreprocessingError as exc:
        return _error_response(
            str(exc), 422, submission=submission, stage="preprocessing"
        )
    except (ModelInferenceError, RuntimeError) as exc:
        logger.exception("Analysis pipeline is unavailable")
        return _error_response(
            "Analysis service is unavailable", 503,
            submission=submission,
            stage="inference",
            detail=str(exc),
        )
    except Exception:
        logger.exception("Analysis pipeline failed")
        return _error_response(
            "Analysis could not be completed", 500,
            submission=submission,
            stage="analysis",
        )

    AuditLogger().record(
        "user_analysis_completed",
        "success",
        submission_id=submission["submission_id"],
        prediction=result.predicted_label,
        confidence=result.confidence,
        actor_email=session["user"]["email"],
    )
    database.record_analysis(
        session["user"]["email"], submission["submission_id"],
        result.predicted_label, result.confidence,
    )
    return jsonify({"submission": submission, "result": result.to_dict()}), 201


@analysis_bp.route("/analyze", methods=["POST"])
@login_required
def analyze_image():
    """Validate one uploaded image and return its final morph-analysis result."""
    if "image" not in request.files:
        return _error_response("No image file provided", 400)
    acquired, error = _acquire(request.files["image"])
    if error is not None:
        return error
    submission, image_bytes = acquired
    return _run_analysis(submission, image_bytes)


@analysis_bp.route("/batch", methods=["POST"])
@login_required
def analyze_batch():
    """Validate and analyse a batch of images, with an optional PDF report."""
    files = request.files.getlist("images")
    if not files:
        return _error_response("At least one image file is required", 400)
    if len(files) > 50:
        return _error_response("Batch contains more than the maximum of 50 images", 400)

    acquired_images = []
    for file_storage in files:
        acquired, error = _acquire(file_storage)
        if error is not None:
            return error
        submission, image_bytes = acquired
        acquired_images.append(
            BatchImage(
                filename=submission["metadata"]["filename"],
                image_bytes=image_bytes,
                submission_id=submission["submission_id"],
            )
        )

    generate_report = request.form.get("generate_report", "true").lower() != "false"
    try:
        batch_result = BatchProcessor(_get_pipeline()).process(
            acquired_images, generate_report=generate_report
        )
    except BatchProcessingError as exc:
        return _error_response(str(exc), 400)
    except RuntimeError as exc:
        logger.exception("Batch analysis is unavailable")
        return _error_response("Batch analysis service is unavailable", 503, detail=str(exc))

    AuditLogger().record(
        "user_batch_completed",
        "success",
        batch_id=batch_result.batch_id,
        actor_email=session["user"]["email"],
        metadata={
            "successful_count": batch_result.successful_count,
            "failed_count": batch_result.failed_count,
        },
    )
    database.record_batch(
        session["user"]["email"], batch_result.batch_id,
        batch_result.successful_count, batch_result.failed_count, batch_result.report_path,
    )
    return jsonify(batch_result.to_dict()), 201


@analysis_bp.route("/report/<batch_id>", methods=["GET"])
@login_required
def download_report(batch_id: str):
    """Download the PDF report recorded for a completed batch."""
    events = AuditLogger().list_events(batch_id=batch_id)
    completed_event = next(
        (event for event in reversed(events) if event.event_type == "batch_completed"),
        None,
    )
    report_path = (completed_event.metadata or {}).get("report_path") if completed_event else None
    reports_directory = (Path("data") / "reports").resolve()
    candidate = Path(report_path).resolve() if report_path else None
    if (
        candidate is None
        or reports_directory not in candidate.parents
        or not candidate.is_file()
    ):
        return _error_response("Batch report not found", 404)
    return send_file(candidate, mimetype="application/pdf", as_attachment=True)


@analysis_bp.route("/audit/submission/<submission_id>", methods=["GET"])
@login_required
def submission_audit_events(submission_id: str):
    """Return operational audit events for a single submission."""
    events = AuditLogger().list_events(submission_id=submission_id)
    return jsonify({"submission_id": submission_id, "events": [event.to_dict() for event in events]})


@analysis_bp.route("/audit/batch/<batch_id>", methods=["GET"])
@login_required
def batch_audit_events(batch_id: str):
    """Return operational audit events for a single batch."""
    events = AuditLogger().list_events(batch_id=batch_id)
    return jsonify({"batch_id": batch_id, "events": [event.to_dict() for event in events]})


@analysis_bp.route("/health", methods=["GET"])
def analysis_health_check():
    """Report API readiness without eagerly loading ML dependencies."""
    return jsonify(
        {
            "status": "healthy",
            "module": "analysis",
            "pipeline_initialized": pipeline is not None,
        }
    )
