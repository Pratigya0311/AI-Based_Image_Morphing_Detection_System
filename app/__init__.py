"""AI-based image morphing detection application package."""

import os
from datetime import timedelta
from pathlib import Path

from flask import Flask
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from app.api import acquisition_bp, analysis_bp, auth_bp
from app.api.auth import init_auth
from app.core.database import database, default_database_url
from app.modules.acquisition import MAX_FILE_SIZE


def create_app():
    """Application factory"""
    app = Flask(__name__)
    environment = os.getenv("FLASK_ENV", "development")
    secret_key = os.getenv("FLASK_SECRET_KEY")
    if environment == "production" and not secret_key:
        raise RuntimeError("FLASK_SECRET_KEY must be set in production")
    app.config.update(
        SECRET_KEY=secret_key or "development-only-change-before-deployment",
        GOOGLE_CLIENT_ID=os.getenv("GOOGLE_CLIENT_ID"),
        GOOGLE_CLIENT_SECRET=os.getenv("GOOGLE_CLIENT_SECRET"),
        GOOGLE_OAUTH_ENABLED=bool(
            os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")
        ),
        FRONTEND_URL=os.getenv("FRONTEND_URL", "http://localhost:5173"),
        BACKEND_URL=os.getenv("BACKEND_URL", "http://localhost:5000"),
        DATABASE_URL=os.getenv("DATABASE_URL", default_database_url()),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=environment == "production",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    )
    if environment == "production" and not app.config["DATABASE_URL"].startswith("postgresql"):
        raise RuntimeError("DATABASE_URL must use PostgreSQL in production")
    database.configure(app.config["DATABASE_URL"])
    database.initialize_schema()
    # Allows multipart overhead while the validator enforces the 10 MiB file limit.
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE + (1024 * 1024)
    
    # Register blueprints
    app.register_blueprint(acquisition_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(auth_bp)
    init_auth(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return {'status': 'healthy'}, 200
    
    return app
