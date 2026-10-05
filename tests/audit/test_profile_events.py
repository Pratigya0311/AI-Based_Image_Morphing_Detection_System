"""Tests for user-attributed audit records and profile statistics."""

from pathlib import Path
import uuid

from app.modules.audit import AuditLogger


def test_filters_audit_events_by_signed_in_actor():
    database_path = Path(f"tests/audit/.tmp_profile_{uuid.uuid4().hex}.db")
    try:
        logger = AuditLogger(database_path)
        logger.record("user_analysis_completed", "success", actor_email="first@example.com")
        logger.record("user_analysis_completed", "success", actor_email="second@example.com")

        events = logger.list_events(actor_email="first@example.com")

        assert len(events) == 1
        assert events[0].actor_email == "first@example.com"
    finally:
        database_path.unlink(missing_ok=True)
