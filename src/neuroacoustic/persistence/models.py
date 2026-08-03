"""SQLAlchemy ORM models for NeuroAcoustic Lab SQLite persistence.

Database schema version: **1** (see ``schema_meta`` table).

Tables
------
- ``schema_meta``: key/value metadata including ``db_schema_version``
- ``tracks``: one row per unique content hash (source identity)
- ``analyses``: one row per (content_hash, analysis_version, config_hash)

Artifact paths are stored as absolute filesystem paths (resolved at write time).
Source audio is never copied into the database.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Explicit software-facing schema version for this SQLite layout.
DB_SCHEMA_VERSION = 1
DB_SCHEMA_VERSION_KEY = "db_schema_version"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base."""


class SchemaMeta(Base):
    """Key/value schema metadata (includes db_schema_version)."""

    __tablename__ = "schema_meta"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class Track(Base):
    """Source identity keyed by content hash (path/filename are informational)."""

    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    native_sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    codec: Mapped[str | None] = mapped_column(String(128), nullable=True)
    container: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )

    analyses: Mapped[list[Analysis]] = relationship(
        back_populates="track", cascade="all, delete-orphan"
    )


class Analysis(Base):
    """One analysis attempt / result for a content hash + version + config."""

    __tablename__ = "analyses"
    __table_args__ = (
        UniqueConstraint(
            "content_hash",
            "analysis_version",
            "config_hash",
            name="uq_analysis_identity",
        ),
        Index("ix_analyses_content_hash", "content_hash"),
        Index("ix_analyses_created_at", "created_at"),
        Index("ix_analyses_status", "status"),
        Index("ix_analyses_identity", "content_hash", "analysis_version", "config_hash"),
        Index("ix_analyses_centroid", "centroid_mean_hz"),
        Index("ix_analyses_flatness", "flatness_mean"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    analysis_version: Mapped[str] = mapped_column(String(32), nullable=False)
    vector_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    fingerprint_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    vector_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Artifact paths (absolute strings)
    artifact_fingerprint_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_waveform_png: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_fft_png: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_fft_log_png: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_spectrogram_png: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Queryable scalar summary columns
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    analysis_sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rms: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_amplitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    centroid_mean_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    flatness_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    entropy_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    f0_median_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    voiced_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    tempo_bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    stereo_width_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    attack_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    tail_decay_t60_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )

    track: Mapped[Track] = relationship(back_populates="analyses")


# Resolve forward refs for type checkers
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    pass
