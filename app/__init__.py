"""
AI-based image morphing detection application package.
"""

from flask import Flask
from app.api import acquisition_bp
from app.modules.acquisition import MAX_FILE_SIZE


def create_app():
    """Application factory"""
    app = Flask(__name__)
    # Allows multipart overhead while the validator enforces the 10 MiB file limit.
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE + (1024 * 1024)
    
    # Register blueprints
    app.register_blueprint(acquisition_bp)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return {'status': 'healthy'}, 200
    
    return app
