"""Portable relational data layer for PostgreSQL deployments and SQLite tests."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import NullPool


def _now() -> datetime:
    return datetime.now(timezone.utc)


def default_database_url() -> str:
    """Provide local persistence when DATABASE_URL is absent during development."""
    path = Path(__file__).resolve().parents[2] / "data" / "database" / "morph_detection.db"
    return f"sqlite:///{path.as_posix()}"


def normalize_database_url(url: str) -> str:
    """Choose Psycopg 3 explicitly for standard PostgreSQL connection URLs."""
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


class Base(DeclarativeBase):
    """Declarative base for every persistent application record."""


class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    google_subject_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    picture_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_login_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analyses: Mapped[list["AnalysisRecord"]] = relationship(back_populates="user")
    batches: Mapped[list["BatchRecord"]] = relationship(back_populates="user")
    audit_events: Mapped[list["AuditEventRecord"]] = relationship(back_populates="user")


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    submission_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    prediction: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped[UserRecord] = relationship(back_populates="analyses")


class BatchRecord(Base):
    __tablename__ = "batch_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    batch_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    successful_count: Mapped[int] = mapped_column(Integer)
    failed_count: Mapped[int] = mapped_column(Integer)
    report_path: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped[UserRecord] = relationship(back_populates="batches")


class AuditEventRecord(Base):
    __tablename__ = "audit_events"

    event_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    event_type: Mapped[str] = mapped_column(String(100), index=True)
    outcome: Mapped[str] = mapped_column(String(32))
    submission_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True, index=True)
    batch_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True, index=True)
    prediction: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)
    actor_email: Mapped[Optional[str]] = mapped_column(String(320), nullable=True, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    user: Mapped[Optional[UserRecord]] = relationship(back_populates="audit_events")


class Database:
    """Owns engine configuration, schema creation, and transactional sessions."""

    def __init__(self) -> None:
        self._url: Optional[str] = None
        self._engine: Optional[Engine] = None
        self._sessions: Optional[sessionmaker[Session]] = None

    @property
    def url(self) -> str:
        return self._url or default_database_url()

    def configure(self, url: Optional[str] = None) -> None:
        configured_url = normalize_database_url(url or os.getenv("DATABASE_URL") or default_database_url())
        if configured_url == self._url and self._engine is not None:
            return
        if configured_url.startswith("sqlite:///"):
            Path(configured_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
            # Avoid retaining a Windows file lock after a short-lived test logger.
            options = {"connect_args": {"check_same_thread": False}, "poolclass": NullPool}
        else:
            options = {"pool_pre_ping": True}
        self._engine = create_engine(configured_url, **options)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)
        self._url = configured_url

    def initialize_schema(self) -> None:
        if self._engine is None:
            self.configure()
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        if self._sessions is None:
            self.configure()
        assert self._sessions is not None
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_google_user(self, profile: dict) -> UserRecord:
        with self.session() as session:
            user = session.scalar(select(UserRecord).where(UserRecord.google_subject_id == profile["sub"]))
            if user is None:
                user = UserRecord(
                    id=str(uuid.uuid4()), google_subject_id=profile["sub"], email=profile["email"],
                    display_name=profile.get("name") or profile["email"], picture_url=profile.get("picture"),
                )
                session.add(user)
            else:
                user.email = profile["email"]
                user.display_name = profile.get("name") or profile["email"]
                user.picture_url = profile.get("picture")
                user.last_login_at = _now()
            session.flush()
            return user

    def user_by_email(self, email: Optional[str]) -> Optional[UserRecord]:
        if not email:
            return None
        with self.session() as session:
            return session.scalar(select(UserRecord).where(UserRecord.email == email))

    def record_analysis(self, email: str, submission_id: str, prediction: str, confidence: float) -> None:
        with self.session() as session:
            user = session.scalar(select(UserRecord).where(UserRecord.email == email))
            if user is not None:
                session.add(AnalysisRecord(id=str(uuid.uuid4()), user_id=user.id, submission_id=submission_id, prediction=prediction, confidence=confidence))

    def record_batch(self, email: str, batch_id: str, successful_count: int, failed_count: int, report_path: Optional[str]) -> None:
        with self.session() as session:
            user = session.scalar(select(UserRecord).where(UserRecord.email == email))
            if user is not None:
                session.add(BatchRecord(id=str(uuid.uuid4()), user_id=user.id, batch_id=batch_id, successful_count=successful_count, failed_count=failed_count, report_path=report_path))

    def profile_stats(self, email: str) -> dict[str, int]:
        with self.session() as session:
            user = session.scalar(select(UserRecord).where(UserRecord.email == email))
            if user is None:
                return {"images_analyzed": 0, "batches_created": 0, "morphed_flags": 0}
            analyses = list(session.scalars(select(AnalysisRecord).where(AnalysisRecord.user_id == user.id)))
            batches = list(session.scalars(select(BatchRecord).where(BatchRecord.user_id == user.id)))
            return {"images_analyzed": len(analyses), "batches_created": len(batches), "morphed_flags": sum(item.prediction == "Morphed" for item in analyses)}


database = Database()
