"""
API layer for exposing module functionality
"""

from flask import Flask, request, jsonify, Blueprint
from werkzeug.utils import secure_filename
import os
import sys
from dataclasses import asdict

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from acquisition module
from app.modules.acquisition import ImageAcquisition, SubmissionManager
from app.schemas import SubmissionResponse

# Create blueprint for acquisition routes
acquisition_bp = Blueprint('acquisition', __name__, url_prefix='/api/acquisition')

# Initialize the acquisition module
acquisition = ImageAcquisition()


@acquisition_bp.route('/upload', methods=['POST'])
def upload_image():
    """
    Endpoint for uploading a single image
    Expects multipart/form-data with 'image' file
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'valid': False
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'valid': False
            }), 400
        
        # Read file bytes - THIS IS THE KEY FIX
        # Use .read() which returns bytes, not text
        file_bytes = file.read()
        filename = secure_filename(file.filename)
        
        # Debug: Print file info
        print(f"Received file: {filename}")
        print(f"File size: {len(file_bytes)} bytes")
        print(f"First 10 bytes: {file_bytes[:10]}")
        
        # Process the image
        result = acquisition.process_image(file_bytes, filename)
        
        response = SubmissionResponse(
            submission_id=result['submission_id'],
            valid=result['valid'],
            error=result.get('error'),
            metadata=result.get('metadata')
        )
        
        status_code = 200 if result['valid'] else 400
        return jsonify(asdict(response)), status_code
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Server error: {str(e)}',
            'valid': False
        }), 500


@acquisition_bp.route('/submission/<submission_id>', methods=['GET'])
def get_submission(submission_id):
    """
    Get the status and metadata of a specific submission
    """
    try:
        result = acquisition.get_submission_status(submission_id)
        if result is None:
            return jsonify({
                'error': 'Submission not found',
                'valid': False
            }), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'valid': False
        }), 500


@acquisition_bp.route('/submissions', methods=['GET'])
def get_all_submissions():
    """
    Get all submissions (for admin/debug purposes)
    """
    try:
        submissions = acquisition.submission_manager.get_all_submissions()
        result = {
            'count': len(submissions),
            'submissions': {
                sid: asdict(metadata) 
                for sid, metadata in submissions.items()
            }
        }
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'valid': False
        }), 500


@acquisition_bp.route('/validate', methods=['POST'])
def validate_image():
    """
    Validate an image without creating a submission
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'valid': False
            }), 400
        
        file = request.files['image']
        # Read as bytes - FIXED
        file_bytes = file.read()
        filename = secure_filename(file.filename)
        
        # Use the validator directly
        validator = acquisition.validator
        is_valid, error_message = validator.validate_image(file_bytes, filename)
        
        # Get image format if valid
        format_name = None
        if is_valid:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(file_bytes))
            format_name = img.format
        
        result = {
            'valid': is_valid,
            'error_message': error_message,
            'format': format_name,
            'size': len(file_bytes)
        }
        
        status_code = 200 if is_valid else 400
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'valid': False
        }), 500


# Health check endpoint
@acquisition_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the acquisition module"""
    return jsonify({
        'status': 'healthy',
        'module': 'acquisition',
        'supported_formats': ['JPEG', 'PNG', 'BMP'],
        'max_file_size': '10MB'
    }), 200