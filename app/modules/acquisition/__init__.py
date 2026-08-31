"""
Image acquisition and validation module
"""

from .validator import (
    ImageValidator,
    SubmissionManager,
    ImageAcquisition,
    SubmissionMetadata,
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE
)

# Also expose the class as 'acquisition' for backward compatibility
# But the main class is ImageAcquisition

__all__ = [
    'ImageValidator',
    'SubmissionManager', 
    'ImageAcquisition',
    'SubmissionMetadata',
    'SUPPORTED_FORMATS',
    'MAX_FILE_SIZE'
]