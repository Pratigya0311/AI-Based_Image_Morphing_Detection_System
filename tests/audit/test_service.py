"""Tests for durable Module 7 audit logging."""

from pathlib import Path
import uuid

from app.modules.audit import AuditLogger


def test_persists_and_filters_structured_audit_events():
    database_path = Path(f"tests/audit/.tmp_audit_{uuid.uuid4().hex}.db")
    try:
        logger = AuditLogger(database_path)
        recorded = logger.record(
            "analysis_completed",
            "success",
            submission_id="SUB_1",
            prediction="Morphed",
            confidence=0.91,
            metadata={"model_version": "v1"},
        )
        logger.record("batch_completed", "success", batch_id="BATCH_1")

        reloaded = AuditLogger(database_path)
        events = reloaded.list_events(submission_id="SUB_1")

        assert len(events) == 1
        assert events[0] == recorded
        assert events[0].metadata == {"model_version": "v1"}
        assert len(reloaded.list_events(batch_id="BATCH_1")) == 1
    finally:
        database_path.unlink(missing_ok=True)
