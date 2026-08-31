"""
Image Acquisition and Validation Module
Handles image upload, format validation, and submission tracking
"""

import os
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import mimetypes
from PIL import Image
import io

# Supported image formats
SUPPORTED_FORMATS = {'JPEG', 'PNG', 'BMP'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@dataclass
class SubmissionMetadata:
    """Data class for submission metadata"""
    submission_id: str
    filename: str
    format: str
    size: int
    timestamp: str
    file_hash: str
    validation_status: str
    error_message: Optional[str] = None


class ImageValidator:
    """
    Validates uploaded images for format, size, and integrity
    """
    
    def __init__(self, max_size: int = MAX_FILE_SIZE):
        self.max_size = max_size
        self.supported_formats = SUPPORTED_FORMATS
    
    def validate_format(self, file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if the image format is supported
        
        Args:
            file_bytes: Raw image file bytes
            filename: Name of the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if it's a valid image using PIL
            # IMPORTANT: Use BytesIO to handle binary data
            img = Image.open(io.BytesIO(file_bytes))
            format_name = img.format
            
            if format_name not in self.supported_formats:
                return False, f"Unsupported format: {format_name}. Supported: {', '.join(self.supported_formats)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def validate_size(self, file_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validate file size within limits
        
        Args:
            file_bytes: Raw image file bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_size = len(file_bytes)
        if file_size > self.max_size:
            return False, f"File size {file_size} exceeds maximum limit of {self.max_size} bytes"
        if file_size == 0:
            return False, "File is empty"
        return True, None
    
    def validate_image(self, file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Complete validation of the image
        
        Args:
            file_bytes: Raw image file bytes
            filename: Name of the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check size first
        size_valid, size_error = self.validate_size(file_bytes)
        if not size_valid:
            return False, size_error
        
        # Check format
        format_valid, format_error = self.validate_format(file_bytes, filename)
        if not format_valid:
            return False, format_error
        
        return True, None


class SubmissionManager:
    """
    Manages image submissions, generates IDs, and stores metadata
    """
    
    def __init__(self):
        self.submissions: Dict[str, SubmissionMetadata] = {}
    
    def generate_submission_id(self) -> str:
        """
        Generate a unique submission identifier
        
        Returns:
            Unique submission ID string
        """
        return f"SUB_{uuid.uuid4().hex[:8].upper()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def compute_file_hash(self, file_bytes: bytes) -> str:
        """
        Compute SHA-256 hash of the file for deduplication/integrity
        
        Args:
            file_bytes: Raw file bytes
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(file_bytes).hexdigest()
    
    def create_submission(self, file_bytes: bytes, filename: str) -> SubmissionMetadata:
        """
        Create a new submission with metadata
        
        Args:
            file_bytes: Raw image file bytes
            filename: Original filename
            
        Returns:
            SubmissionMetadata object
        """
        # Get image format
        img = Image.open(io.BytesIO(file_bytes))
        format_name = img.format
        
        submission_id = self.generate_submission_id()
        file_hash = self.compute_file_hash(file_bytes)
        
        metadata = SubmissionMetadata(
            submission_id=submission_id,
            filename=filename,
            format=format_name,
            size=len(file_bytes),
            timestamp=datetime.now().isoformat(),
            file_hash=file_hash,
            validation_status="valid"
        )
        
        self.submissions[submission_id] = metadata
        return metadata
    
    def get_submission(self, submission_id: str) -> Optional[SubmissionMetadata]:
        """
        Retrieve submission metadata by ID
        
        Args:
            submission_id: The submission ID
            
        Returns:
            SubmissionMetadata if found, None otherwise
        """
        return self.submissions.get(submission_id)
    
    def get_all_submissions(self) -> Dict[str, SubmissionMetadata]:
        """
        Get all submissions
        
        Returns:
            Dictionary of submission_id -> SubmissionMetadata
        """
        return self.submissions.copy()
    
    def update_validation_status(self, submission_id: str, status: str, error_msg: Optional[str] = None) -> bool:
        """
        Update the validation status of a submission
        
        Args:
            submission_id: The submission ID
            status: New validation status
            error_msg: Optional error message
            
        Returns:
            True if updated, False if submission not found
        """
        if submission_id not in self.submissions:
            return False
        
        self.submissions[submission_id].validation_status = status
        self.submissions[submission_id].error_message = error_msg
        return True


class ImageAcquisition:
    """
    Main entry point for Image Acquisition and Validation module
    Orchestrates validation and submission creation
    """
    
    def __init__(self):
        self.validator = ImageValidator()
        self.submission_manager = SubmissionManager()
    
    def process_image(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a single image submission
        
        Args:
            file_bytes: Raw image file bytes
            filename: Original filename
            
        Returns:
            Dictionary containing submission result with metadata
        """
        # Validate the image
        is_valid, error_message = self.validator.validate_image(file_bytes, filename)
        
        if not is_valid:
            # Create a minimal metadata for invalid submission
            submission_id = self.submission_manager.generate_submission_id()
            file_hash = self.submission_manager.compute_file_hash(file_bytes)
            
            metadata = SubmissionMetadata(
                submission_id=submission_id,
                filename=filename,
                format="Unknown",
                size=len(file_bytes),
                timestamp=datetime.now().isoformat(),
                file_hash=file_hash,
                validation_status="invalid",
                error_message=error_message
            )
            self.submission_manager.submissions[submission_id] = metadata
            
            return {
                'submission_id': submission_id,
                'valid': False,
                'error': error_message,
                'metadata': asdict(metadata)
            }
        
        # Create submission for valid image
        metadata = self.submission_manager.create_submission(file_bytes, filename)
        
        return {
            'submission_id': metadata.submission_id,
            'valid': True,
            'error': None,
            'metadata': asdict(metadata)
        }
    
    def get_submission_status(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific submission
        
        Args:
            submission_id: The submission ID
            
        Returns:
            Dictionary with submission status and metadata
        """
        metadata = self.submission_manager.get_submission(submission_id)
        if not metadata:
            return None
        
        return {
            'submission_id': metadata.submission_id,
            'status': metadata.validation_status,
            'metadata': asdict(metadata)
        }