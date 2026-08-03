"""Pydantic models for source metadata and versioned acoustic fingerprints."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from neuroacoustic.analysis.stats import DistributionStats, TimeSeriesSummary


class ProbeBackend(str, Enum):
    SOUNDFILE = "soundfile"
    FFPROBE = "ffprobe"
    HYBRID = "hybrid"
    EXTENSION = "extension"


class SourceMetadata(BaseModel):
    """Technical metadata for an audio source file."""

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
        description="Sample rate used for analysis (native unless overridden)",
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
    """Analysis settings recorded with a probe/fingerprint."""

    schema_version: str
    analysis_version: str
    analysis_sample_rate: int | None = None
    silence_peak_threshold: float
    silence_sample_threshold: float
    clipping_threshold: float
    config_path: str | None = None
    n_fft: int | None = None
    hop_length: int | None = None
    win_length: int | None = None
    rolloff_percentile: float | None = None
    n_contrast_bands: int | None = None
    vector_version: str | None = None


def build_analysis_config_snapshot(
    config: Any,
    *,
    analysis_sample_rate: int | None = None,
) -> AnalysisConfigSnapshot:
    """Build a snapshot from an ``AppConfig``-like object."""
    return AnalysisConfigSnapshot(
        schema_version=config.project.schema_version,
        analysis_version=config.project.analysis_version,
        analysis_sample_rate=analysis_sample_rate
        if analysis_sample_rate is not None
        else config.audio.analysis_sample_rate,
        silence_peak_threshold=config.audio.silence_peak_threshold,
        silence_sample_threshold=config.audio.silence_sample_threshold,
        clipping_threshold=config.audio.clipping_threshold,
        config_path=str(config.config_path) if config.config_path else None,
        n_fft=config.analysis.n_fft,
        hop_length=config.analysis.hop_length,
        win_length=config.analysis.effective_win_length(),
        rolloff_percentile=config.analysis.rolloff_percentile,
        n_contrast_bands=config.analysis.n_contrast_bands,
        vector_version=config.analysis.vector_version,
    )


class QualityMetrics(BaseModel):
    """Quality indicators (load-time + analysis warnings)."""

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


class FftPeak(BaseModel):
    frequency_hz: float
    magnitude: float


class FftSummary(BaseModel):
    """Whole-signal FFT summary (not the full transform)."""

    peak_frequency_hz: float | None = None
    peak_magnitude: float | None = None
    peaks: list[FftPeak] = Field(default_factory=list)
    summary_frequencies_hz: list[float] = Field(default_factory=list)
    summary_magnitudes: list[float] = Field(default_factory=list)
    log_summary_frequencies_hz: list[float] = Field(default_factory=list)
    log_summary_magnitudes: list[float] = Field(default_factory=list)


class StftSummary(BaseModel):
    """STFT metadata and mean spectrum — never the full magnitude matrix."""

    n_fft: int
    hop_length: int
    win_length: int
    n_frames: int
    n_freq_bins: int
    duration_seconds: float | None = None
    mean_spectrum_frequencies_hz: list[float] = Field(default_factory=list)
    mean_spectrum_magnitudes: list[float] = Field(default_factory=list)


class SpectralSection(BaseModel):
    fft: FftSummary
    stft: StftSummary
    centroid_hz: DistributionStats
    bandwidth_hz: DistributionStats
    rolloff_hz: DistributionStats
    flatness: DistributionStats
    entropy: DistributionStats
    contrast_db: DistributionStats
    contrast_band_means_db: list[float] = Field(default_factory=list)
    centroid_curve: TimeSeriesSummary | None = None
    flatness_curve: TimeSeriesSummary | None = None


class EnergySection(BaseModel):
    rms: float | None = None
    rms_db: float | None = Field(
        default=None,
        description="20*log10(RMS); amplitude dBFS-like, NOT LUFS",
    )
    peak_amplitude: float | None = None
    crest_factor: float | None = None
    crest_factor_db: float | None = None
    zero_crossing_rate: float | None = Field(
        default=None,
        description="Mean frame ZCR in crossings per sample",
    )
    estimated_dynamic_range_db: float | None = Field(
        default=None,
        description=(
            "Heuristic 20*log10(RMS_p95/RMS_p05); estimate only — not LUFS, "
            "not AES/EBU loudness, not True Peak"
        ),
    )
    rms_frame: DistributionStats
    zero_crossing_rate_frame: DistributionStats
    rms_curve: TimeSeriesSummary | None = None


class ArtifactPaths(BaseModel):
    fingerprint_json: str | None = None
    waveform_png: str | None = None
    fft_png: str | None = None
    fft_log_png: str | None = None
    spectrogram_png: str | None = None


class PreliminaryVector(BaseModel):
    """Versioned preliminary comparison vector (not dataset-normalized)."""

    version: str
    values: list[float]
    labels: list[str]
    note: str = (
        "Preliminary unnormalized / heuristically scaled features. "
        "Dataset-level normalization will be introduced when a real corpus exists."
    )


class AcousticFingerprint(BaseModel):
    """Versioned acoustic fingerprint document (Milestone 2)."""

    schema_version: str
    analysis_version: str
    source: SourceMetadata
    analysis_config: AnalysisConfigSnapshot
    quality: QualityMetrics
    spectral: SpectralSection
    pitch: dict[str, Any] = Field(default_factory=dict)
    harmonics: dict[str, Any] = Field(default_factory=dict)
    energy: EnergySection
    envelope: dict[str, Any] = Field(default_factory=dict)
    stereo: dict[str, Any] = Field(default_factory=dict)
    rhythm: dict[str, Any] = Field(default_factory=dict)
    reverberation: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] = Field(default_factory=list)
    vector_meta: PreliminaryVector | None = None
    artifacts: ArtifactPaths = Field(default_factory=ArtifactPaths)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
