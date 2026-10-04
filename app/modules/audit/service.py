"""Durable SQLite append-only operational records for image-analysis activity."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class AuditEvent:
    """An immutable record of one pipeline or batch operation."""

    event_id: str
    timestamp: str
    event_type: str
    outcome: str
    submission_id: Optional[str] = None
    batch_id: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Writes append-only operational records to SQLite."""

    def __init__(self, database_path: Optional[str | Path] = None) -> None:
        self.database_path = Path(database_path or Path("data") / "audit" / "audit.db")
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    submission_id TEXT,
                    batch_id TEXT,
                    prediction TEXT,
                    confidence REAL,
                    error_message TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_submission ON audit_events(submission_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_batch ON audit_events(batch_id)"
            )
            connection.commit()
        finally:
            connection.close()

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
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AuditEvent:
        """Persist one structured audit event and return its immutable record."""
        if not event_type.strip() or not outcome.strip():
            raise ValueError("Audit event type and outcome are required")
        event = AuditEvent(
            event_id=f"AUDIT_{uuid.uuid4().hex.upper()}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            outcome=outcome,
            submission_id=submission_id,
            batch_id=batch_id,
            prediction=prediction,
            confidence=confidence,
            error_message=error_message,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, outcome, submission_id,
                        batch_id, prediction, confidence, error_message, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.timestamp,
                        event.event_type,
                        event.outcome,
                        event.submission_id,
                        event.batch_id,
                        event.prediction,
                        event.confidence,
                        event.error_message,
                        json.dumps(event.metadata, sort_keys=True),
                    ),
                )
                connection.commit()
            finally:
                connection.close()
        return event

    def list_events(
        self, *, submission_id: Optional[str] = None, batch_id: Optional[str] = None
    ) -> list[AuditEvent]:
        """Return audit events, optionally filtered by submission or batch."""
        clauses, parameters = [], []
        if submission_id:
            clauses.append("submission_id = ?")
            parameters.append(submission_id)
        if batch_id:
            clauses.append("batch_id = ?")
            parameters.append(batch_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    f"SELECT * FROM audit_events{where} ORDER BY timestamp ASC", parameters
                ).fetchall()
            finally:
                connection.close()
        return [
            AuditEvent(
                event_id=row["event_id"], timestamp=row["timestamp"], event_type=row["event_type"],
                outcome=row["outcome"], submission_id=row["submission_id"], batch_id=row["batch_id"],
                prediction=row["prediction"], confidence=row["confidence"],
                error_message=row["error_message"], metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]
