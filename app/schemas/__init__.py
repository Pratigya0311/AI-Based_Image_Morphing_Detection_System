"""
Shared data-contract schemas
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ImageSubmission:
    """Schema for image submission"""
    file_bytes: bytes
    filename: str
    format: Optional[str] = None
    size: Optional[int] = None


@dataclass
class SubmissionResponse:
    """Schema for submission response"""
    submission_id: str
    valid: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ValidationResult:
    """Schema for validation result"""
    valid: bool
    error_message: Optional[str] = None
    format: Optional[str] = None
    size: Optional[int] = None


@dataclass
class BatchSubmission:
    """Schema for batch submission"""
    images: List[ImageSubmission]
    batch_id: Optional[str] = None