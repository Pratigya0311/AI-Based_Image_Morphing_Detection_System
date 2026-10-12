"""Portable relational data layer for PostgreSQL deployments and SQLite tests."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import NullPool
from werkzeug.security import generate_password_hash


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
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(32), default="user", server_default="user", index=True)
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


class PendingRegistration(Base):
    """Server-side registration details awaiting email verification."""

    __tablename__ = "pending_registrations"

    token: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


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
        # ``create_all`` creates tables but does not add a newly introduced
        # column to databases that already exist. This tiny additive migration
        # keeps local SQLite and deployed PostgreSQL project databases usable.
        assert self._engine is not None
        user_columns = {column["name"] for column in inspect(self._engine).get_columns("users")}
        if "password_hash" not in user_columns:
            with self._engine.begin() as connection:
                connection.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)"))
        if "role" not in user_columns:
            with self._engine.begin() as connection:
                connection.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(32) DEFAULT 'user'"))

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
                user = session.scalar(select(UserRecord).where(UserRecord.email == profile["email"].lower()))
            if user is None:
                user = UserRecord(
                    id=str(uuid.uuid4()), google_subject_id=profile["sub"], email=profile["email"].lower(),
                    display_name=profile.get("name") or profile["email"], picture_url=profile.get("picture"),
                )
                session.add(user)
            else:
                user.google_subject_id = profile["sub"]
                user.email = profile["email"].lower()
                user.display_name = profile.get("name") or profile["email"]
                user.picture_url = profile.get("picture")
                user.last_login_at = _now()
            session.flush()
            return user

    def create_local_user(self, name: str, email: str, password_hash: str) -> Optional[UserRecord]:
        """Create an email/password user, returning None when the email exists."""
        normalized_email = email.lower()
        with self.session() as session:
            existing = session.scalar(select(UserRecord).where(UserRecord.email == normalized_email))
            if existing is not None:
                return None
            user = UserRecord(
                id=str(uuid.uuid4()),
                # Older project databases require this legacy provider field;
                # a local identifier keeps it unique without storing secrets.
                google_subject_id=f"local:{uuid.uuid4()}",
                email=normalized_email,
                display_name=name,
                password_hash=password_hash,
            )
            session.add(user)
            session.flush()
            return user

    def create_pending_registration(self, name: str, email: str, password_hash: str) -> str:
        """Replace any pending request for an email and return a random verification token."""
        normalized_email = email.lower()
        token = uuid.uuid4().hex
        with self.session() as session:
            pending = session.scalar(select(PendingRegistration).where(PendingRegistration.email == normalized_email))
            if pending is not None:
                session.delete(pending)
                session.flush()
            session.add(PendingRegistration(token=token, display_name=name, email=normalized_email, password_hash=password_hash))
        return token

    def consume_pending_registration(self, token: str, max_age_seconds: int = 3600) -> Optional[UserRecord]:
        """Atomically activate a non-expired pending registration."""
        with self.session() as session:
            pending = session.get(PendingRegistration, token)
            if pending is None or (_now() - pending.created_at).total_seconds() > max_age_seconds:
                if pending is not None:
                    session.delete(pending)
                return None
            if session.scalar(select(UserRecord).where(UserRecord.email == pending.email)) is not None:
                session.delete(pending)
                return None
            user = UserRecord(
                id=str(uuid.uuid4()), google_subject_id=f"local:{uuid.uuid4()}", email=pending.email,
                display_name=pending.display_name, password_hash=pending.password_hash,
            )
            session.add(user)
            session.delete(pending)
            session.flush()
            return user

    def ensure_admin(self, email: Optional[str], initial_password: Optional[str]) -> None:
        """Create or promote the deployment-configured admin without source credentials."""
        if not email or not initial_password:
            return
        normalized_email = email.strip().lower()
        with self.session() as session:
            user = session.scalar(select(UserRecord).where(UserRecord.email == normalized_email))
            if user is None:
                user = UserRecord(
                    id=str(uuid.uuid4()), google_subject_id=f"local:{uuid.uuid4()}",
                    email=normalized_email, display_name="Administrator",
                    password_hash=generate_password_hash(initial_password), role="admin",
                )
                session.add(user)
            else:
                user.role = "admin"
                if not user.password_hash:
                    user.password_hash = generate_password_hash(initial_password)

    def list_users(self) -> list[dict]:
        with self.session() as session:
            users = session.scalars(select(UserRecord).order_by(UserRecord.created_at.desc())).all()
            return [
                {
                    "id": user.id, "name": user.display_name, "email": user.email,
                    "role": user.role, "created_at": user.created_at.isoformat(),
                    "last_signed_in_at": user.last_login_at.isoformat(),
                }
                for user in users
            ]

    def delete_user(self, user_id: str) -> bool:
        with self.session() as session:
            user = session.get(UserRecord, user_id)
            if user is None:
                return False
            for record in session.scalars(select(AnalysisRecord).where(AnalysisRecord.user_id == user_id)):
                session.delete(record)
            for record in session.scalars(select(BatchRecord).where(BatchRecord.user_id == user_id)):
                session.delete(record)
            for event in session.scalars(select(AuditEventRecord).where(AuditEventRecord.user_id == user_id)):
                session.delete(event)
            session.delete(user)
            return True

    def local_user_by_email(self, email: str) -> Optional[UserRecord]:
        with self.session() as session:
            return session.scalar(select(UserRecord).where(UserRecord.email == email.lower()))

    def mark_logged_in(self, user_id: str) -> None:
        with self.session() as session:
            user = session.get(UserRecord, user_id)
            if user is not None:
                user.last_login_at = _now()

    def account_details(self, email: str) -> Optional[dict]:
        """Return non-sensitive account fields for the signed-in profile."""
        with self.session() as session:
            user = session.scalar(select(UserRecord).where(UserRecord.email == email.lower()))
            if user is None:
                return None
            return {
                "name": user.display_name,
                "email": user.email,
                "picture": user.picture_url,
                "has_password": bool(user.password_hash),
                "role": user.role,
                "last_signed_in_at": user.last_login_at.isoformat(),
            }

    def update_password(self, user_id: str, password_hash: str) -> bool:
        with self.session() as session:
            user = session.get(UserRecord, user_id)
            if user is None:
                return False
            user.password_hash = password_hash
            return True

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
