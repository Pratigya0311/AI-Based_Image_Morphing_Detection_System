"""REST API routes for the image-acquisition module."""

import logging
from dataclasses import asdict

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from app.modules.acquisition import ImageAcquisition, MAX_FILE_SIZE
from app.schemas import SubmissionResponse

acquisition_bp = Blueprint("acquisition", __name__, url_prefix="/api/acquisition")
acquisition = ImageAcquisition()
logger = logging.getLogger(__name__)


def _error_response(message: str, status_code: int):
    return jsonify({"error": message, "valid": False}), status_code


@acquisition_bp.app_errorhandler(RequestEntityTooLarge)
def handle_oversized_request(error):
    """Return JSON when Flask rejects an oversized multipart request."""
    return _error_response("Image upload exceeds the maximum permitted size", 413)


@acquisition_bp.route("/upload", methods=["POST"])
def upload_image():
    """Create a validated submission from a multipart ``image`` field."""
    if "image" not in request.files:
        return _error_response("No image file provided", 400)
    file = request.files["image"]
    if not file.filename:
        return _error_response("No file selected", 400)

    try:
        result = acquisition.process_image(file.read(), secure_filename(file.filename))
    except RuntimeError:
        logger.exception("Failed to store image submission metadata")
        return _error_response("Unable to store submission metadata", 500)

    response = SubmissionResponse(
        submission_id=result["submission_id"],
        valid=result["valid"],
        error=result["error"],
        metadata=result["metadata"],
    )
    return jsonify(asdict(response)), 201 if result["valid"] else 400


@acquisition_bp.route("/submission/<submission_id>", methods=["GET"])
def get_submission(submission_id: str):
    """Return one submission's current validation status."""
    result = acquisition.get_submission_status(submission_id)
    if result is None:
        return _error_response("Submission not found", 404)
    return jsonify(result), 200


@acquisition_bp.route("/submissions", methods=["GET"])
def get_all_submissions():
    """Return submission metadata for operational inspection."""
    submissions = acquisition.submission_manager.get_all_submissions()
    return jsonify(
        {
            "count": len(submissions),
            "submissions": {
                submission_id: asdict(metadata)
                for submission_id, metadata in submissions.items()
            },
        }
    ), 200


@acquisition_bp.route("/validate", methods=["POST"])
def validate_image():
    """Validate an image without creating a submission record."""
    if "image" not in request.files:
        return _error_response("No image file provided", 400)
    file = request.files["image"]
    file_bytes = file.read()
    valid, error_message = acquisition.validator.validate_image(
        file_bytes, secure_filename(file.filename)
    )
    return jsonify(
        {
            "valid": valid,
            "error_message": error_message,
            "format": acquisition.validator.get_image_format(file_bytes) if valid else None,
            "size": len(file_bytes),
        }
    ), 200 if valid else 400


@acquisition_bp.route("/health", methods=["GET"])
def health_check():
    """Return acquisition module health and supported upload limits."""
    return jsonify(
        {
            "status": "healthy",
            "module": "acquisition",
            "supported_formats": sorted(acquisition.validator.supported_formats),
            "max_file_size": MAX_FILE_SIZE,
        }
    ), 200
