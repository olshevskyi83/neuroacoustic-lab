"""Pydantic models for source metadata and early fingerprint scaffolding."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProbeBackend(str, Enum):
    SOUNDFILE = "soundfile"
    FFPROBE = "ffprobe"
    HYBRID = "hybrid"
    EXTENSION = "extension"


class SourceMetadata(BaseModel):
    """Technical metadata for an audio source file (Milestone 1)."""

    filename: str
    path: str
    content_hash: str = Field(description="SHA-256 hex digest of file bytes")
    file_size_bytes: int
    container: str | None = None
    codec: str | None = None
    duration_seconds: float | None = None
    native_sample_rate: int | None = None
    analysis_sample_rate: int | None = Field(
        default=None,
        description="Sample rate that will be used for analysis (native unless overridden)",
    )
    channels: int | None = None
    bit_depth: int | None = None
    bitrate: int | None = Field(default=None, description="Encoded bitrate in bit/s when known")
    probe_backend: ProbeBackend = ProbeBackend.EXTENSION
    warnings: list[str] = Field(default_factory=list)

    @field_validator("content_hash")
    @classmethod
    def _hash_looks_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("content_hash must be a 64-character hex SHA-256 digest")
        return value.lower()


class AnalysisConfigSnapshot(BaseModel):
    """Subset of analysis settings recorded with a probe/fingerprint."""

    schema_version: str
    analysis_version: str
    analysis_sample_rate: int | None = None
    silence_peak_threshold: float
    silence_sample_threshold: float
    clipping_threshold: float
    config_path: str | None = None


class QualityMetrics(BaseModel):
    """Load-time quality indicators (expanded in later milestones)."""

    clipping_ratio: float = Field(ge=0.0, le=1.0)
    silence_ratio: float = Field(ge=0.0, le=1.0)
    peak_amplitude: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class ProbeResult(BaseModel):
    """Result of ``neuroacoustic probe``."""

    source: SourceMetadata
    analysis_config: AnalysisConfigSnapshot
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_printable_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
