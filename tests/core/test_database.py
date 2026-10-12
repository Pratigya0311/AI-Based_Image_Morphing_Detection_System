"""Tests for the portable PostgreSQL/SQLite relational schema."""

from pathlib import Path
import uuid

from app.core.database import Database


def test_persists_user_activity_in_normalized_records():
    database_path = Path(f"tests/core/.tmp_database_{uuid.uuid4().hex}.db")
    database = Database()
    database.configure(f"sqlite:///{database_path.as_posix()}")
    database.initialize_schema()
    try:
        database.upsert_google_user(
            {
                "sub": "google-subject-1",
                "email": "student@example.com",
                "name": "Student User",
                "picture": "https://example.com/profile.png",
            }
        )
        database.record_analysis("student@example.com", "SUB_1", "Morphed", 0.93)
        database.record_batch("student@example.com", "BATCH_1", 2, 0, "report.pdf")

        assert database.profile_stats("student@example.com") == {
            "images_analyzed": 1,
            "batches_created": 1,
            "morphed_flags": 1,
        }
    finally:
        database_path.unlink(missing_ok=True)
