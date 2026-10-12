"""Production WSGI entry point used by Gunicorn-compatible hosts."""

from app import create_app

app = create_app()
