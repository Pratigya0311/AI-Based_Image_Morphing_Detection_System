"""
AI-based image morphing detection application package.
"""

from flask import Flask
from app.api import acquisition_bp


def create_app():
    """Application factory"""
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(acquisition_bp)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return {'status': 'healthy'}, 200
    
    return app