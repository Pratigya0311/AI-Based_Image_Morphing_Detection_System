"""Audit persistence backed by the shared relational application database."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from sqlalchemy import select

from app.core.database import AuditEventRecord, Database, database


@dataclass(frozen=True)
class AuditEvent:
    """One immutable, operationally useful audit record."""

    event_id: str
    timestamp: str
    event_type: str
    outcome: str
    submission_id: Optional[str] = None
    batch_id: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    actor_email: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Writes audit records to PostgreSQL or an isolated SQLite test database."""

    def __init__(self, database_path: Optional[str | Path] = None) -> None:
        if database_path is None:
            self.database = database
        else:
            self.database = Database()
            self.database.configure(f"sqlite:///{Path(database_path).as_posix()}")
        self.database.initialize_schema()

    @staticmethod
    def _to_event(record: AuditEventRecord) -> AuditEvent:
        timestamp = record.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return AuditEvent(
            event_id=record.event_id,
            timestamp=timestamp.isoformat(),
            event_type=record.event_type,
            outcome=record.outcome,
            submission_id=record.submission_id,
            batch_id=record.batch_id,
            prediction=record.prediction,
            confidence=record.confidence,
            error_message=record.error_message,
            actor_email=record.actor_email,
            metadata=record.metadata_json,
        )

    def record(
        self,
        event_type: str,
        outcome: str,
        *,
        submission_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        error_message: Optional[str] = None,
        actor_email: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AuditEvent:
        """Persist one structured event in the configured relational database."""
        if not event_type.strip() or not outcome.strip():
            raise ValueError("Audit event type and outcome are required")
        record = AuditEventRecord(
            event_id=f"AUDIT_{uuid.uuid4().hex.upper()}",
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            outcome=outcome,
            submission_id=submission_id,
            batch_id=batch_id,
            prediction=prediction,
            confidence=confidence,
            error_message=error_message,
            actor_email=actor_email,
            metadata_json=dict(metadata or {}),
        )
        with self.database.session() as session:
            session.add(record)
        return self._to_event(record)

    def list_events(
        self,
        *,
        submission_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        actor_email: Optional[str] = None,
    ) -> list[AuditEvent]:
        """Return ordered events filtered by correlation fields."""
        statement = select(AuditEventRecord).order_by(AuditEventRecord.timestamp.asc())
        if submission_id:
            statement = statement.where(AuditEventRecord.submission_id == submission_id)
        if batch_id:
            statement = statement.where(AuditEventRecord.batch_id == batch_id)
        if actor_email:
            statement = statement.where(AuditEventRecord.actor_email == actor_email)
        with self.database.session() as session:
            records = list(session.scalars(statement))
        return [self._to_event(record) for record in records]
