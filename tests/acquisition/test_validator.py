"""
Unit tests for Image Acquisition and Validation module
"""

import unittest
import os
import io
from PIL import Image
from app.modules.acquisition import (
    ImageValidator, 
    SubmissionManager, 
    ImageAcquisition,
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE
)


class TestImageValidator(unittest.TestCase):
    """Test cases for ImageValidator"""
    
    def setUp(self):
        self.validator = ImageValidator()
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='PNG')
        self.image_data = self.image_bytes.getvalue()
    
    def test_validate_size_valid(self):
        """Test size validation with valid size"""
        valid, error = self.validator.validate_size(self.image_data)
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_validate_size_empty(self):
        """Test size validation with empty file"""
        valid, error = self.validator.validate_size(b'')
        self.assertFalse(valid)
        self.assertEqual(error, "File is empty")
    
    def test_validate_size_exceeds_limit(self):
        """Test size validation with file exceeding limit"""
        large_data = b'x' * (MAX_FILE_SIZE + 1)
        valid, error = self.validator.validate_size(large_data)
        self.assertFalse(valid)
        self.assertIn("exceeds maximum limit", error)
    
    def test_validate_format_valid(self):
        """Test format validation with valid format"""
        valid, error = self.validator.validate_format(self.image_data, 'test.png')
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_validate_format_invalid(self):
        """Test format validation with invalid format"""
        # Create a text file as bytes
        text_data = b'This is not an image'
        valid, error = self.validator.validate_format(text_data, 'test.txt')
        self.assertFalse(valid)
        self.assertIsNotNone(error)
    
    def test_validate_format_unsupported(self):
        """Test format validation with unsupported but valid image format"""
        # Create a GIF image (not in supported formats)
        gif_image = Image.new('RGB', (100, 100), color='blue')
        gif_bytes = io.BytesIO()
        gif_image.save(gif_bytes, format='GIF')
        gif_data = gif_bytes.getvalue()
        
        valid, error = self.validator.validate_format(gif_data, 'test.gif')
        self.assertFalse(valid)
        self.assertIn("Unsupported format", error)
    
    def test_validate_image_complete(self):
        """Test complete image validation"""
        valid, error = self.validator.validate_image(self.image_data, 'test.png')
        self.assertTrue(valid)
        self.assertIsNone(error)


class TestSubmissionManager(unittest.TestCase):
    """Test cases for SubmissionManager"""
    
    def setUp(self):
        self.manager = SubmissionManager()
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='PNG')
        self.image_data = self.image_bytes.getvalue()
    
    def test_generate_submission_id(self):
        """Test submission ID generation"""
        id1 = self.manager.generate_submission_id()
        id2 = self.manager.generate_submission_id()
        
        self.assertIsNotNone(id1)
        self.assertIsNotNone(id2)
        self.assertNotEqual(id1, id2)
        self.assertTrue(id1.startswith('SUB_'))
        self.assertTrue(id2.startswith('SUB_'))
    
    def test_compute_file_hash(self):
        """Test file hash computation"""
        hash1 = self.manager.compute_file_hash(self.image_data)
        hash2 = self.manager.compute_file_hash(self.image_data)
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA-256 hex length
        self.assertTrue(all(c in '0123456789abcdef' for c in hash1))
    
    def test_create_submission(self):
        """Test submission creation"""
        metadata = self.manager.create_submission(self.image_data, 'test.png')
        
        self.assertIsNotNone(metadata.submission_id)
        self.assertEqual(metadata.filename, 'test.png')
        self.assertEqual(metadata.format, 'PNG')
        self.assertEqual(metadata.size, len(self.image_data))
        self.assertEqual(metadata.validation_status, 'valid')
        self.assertIsNone(metadata.error_message)
        self.assertIsNotNone(metadata.file_hash)
        self.assertIsNotNone(metadata.timestamp)
    
    def test_get_submission(self):
        """Test retrieving submission by ID"""
        metadata = self.manager.create_submission(self.image_data, 'test.png')
        retrieved = self.manager.get_submission(metadata.submission_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.submission_id, metadata.submission_id)
        self.assertEqual(retrieved.filename, metadata.filename)
    
    def test_get_submission_not_found(self):
        """Test retrieving non-existent submission"""
        retrieved = self.manager.get_submission('NON_EXISTENT_ID')
        self.assertIsNone(retrieved)
    
    def test_get_all_submissions(self):
        """Test getting all submissions"""
        self.manager.create_submission(self.image_data, 'test1.png')
        self.manager.create_submission(self.image_data, 'test2.png')
        
        all_subs = self.manager.get_all_submissions()
        self.assertEqual(len(all_subs), 2)
    
    def test_update_validation_status(self):
        """Test updating validation status"""
        metadata = self.manager.create_submission(self.image_data, 'test.png')
        
        # Update to invalid with error
        result = self.manager.update_validation_status(
            metadata.submission_id, 
            'invalid', 
            'Test error message'
        )
        self.assertTrue(result)
        
        updated = self.manager.get_submission(metadata.submission_id)
        self.assertEqual(updated.validation_status, 'invalid')
        self.assertEqual(updated.error_message, 'Test error message')
    
    def test_update_status_not_found(self):
        """Test updating status for non-existent submission"""
        result = self.manager.update_validation_status('NON_EXISTENT', 'invalid')
        self.assertFalse(result)


class TestImageAcquisition(unittest.TestCase):
    """Test cases for ImageAcquisition"""
    
    def setUp(self):
        self.acquisition = ImageAcquisition()
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='PNG')
        self.image_data = self.image_bytes.getvalue()
    
    def test_process_image_valid(self):
        """Test processing a valid image"""
        result = self.acquisition.process_image(self.image_data, 'test.png')
        
        self.assertTrue(result['valid'])
        self.assertIsNotNone(result['submission_id'])
        self.assertIsNone(result['error'])
        self.assertIsNotNone(result['metadata'])
        self.assertEqual(result['metadata']['filename'], 'test.png')
        self.assertEqual(result['metadata']['format'], 'PNG')
    
    def test_process_image_invalid_format(self):
        """Test processing an invalid format"""
        text_data = b'This is not an image'
        result = self.acquisition.process_image(text_data, 'test.txt')
        
        self.assertFalse(result['valid'])
        self.assertIsNotNone(result['submission_id'])
        self.assertIsNotNone(result['error'])
        self.assertEqual(result['metadata']['validation_status'], 'invalid')
    
    def test_process_image_empty(self):
        """Test processing an empty file"""
        result = self.acquisition.process_image(b'', 'empty.png')
        
        self.assertFalse(result['valid'])
        self.assertIsNotNone(result['error'])
        self.assertEqual(result['metadata']['validation_status'], 'invalid')
    
    def test_get_submission_status(self):
        """Test getting submission status"""
        result = self.acquisition.process_image(self.image_data, 'test.png')
        status = self.acquisition.get_submission_status(result['submission_id'])
        
        self.assertIsNotNone(status)
        self.assertEqual(status['submission_id'], result['submission_id'])
        self.assertEqual(status['status'], 'valid')
    
    def test_get_submission_status_not_found(self):
        """Test getting status for non-existent submission"""
        status = self.acquisition.get_submission_status('NON_EXISTENT')
        self.assertIsNone(status)


if __name__ == '__main__':
    unittest.main()