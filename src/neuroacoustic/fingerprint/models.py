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
    f0_min_hz: float | None = None
    f0_max_hz: float | None = None
    f0_frame_length: int | None = None
    voiced_prob_threshold: float | None = None
    voiced_prob_floor: float | None = None
    max_harmonics: int | None = None
    harmonic_search_width_fraction: float | None = None
    harmonic_rel_amp_threshold: float | None = None


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
        f0_min_hz=config.analysis.f0_min_hz,
        f0_max_hz=config.analysis.f0_max_hz,
        f0_frame_length=config.analysis.f0_frame_length,
        voiced_prob_threshold=config.analysis.voiced_prob_threshold,
        voiced_prob_floor=config.analysis.voiced_prob_floor,
        max_harmonics=config.analysis.max_harmonics,
        harmonic_search_width_fraction=config.analysis.harmonic_search_width_fraction,
        harmonic_rel_amp_threshold=config.analysis.harmonic_rel_amp_threshold,
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


class PitchSection(BaseModel):
    """Single-F0 pitch / voicing summary (Milestone 3)."""

    method: str = "librosa.pyin"
    f0_min_hz: float | None = None
    f0_max_hz: float | None = None
    frame_count: int = 0
    voiced_frame_count: int = 0
    voiced_ratio: float | None = Field(
        default=None,
        description="Fraction of frames with usable voiced F0",
    )
    mean_voiced_probability: float | None = None
    confidence: float | None = Field(
        default=None,
        description="Heuristic 0–1 confidence from voicing probability and coverage",
    )
    f0_voiced_hz: DistributionStats
    f0_median_hz: float | None = Field(
        default=None,
        description="Median voiced F0 in Hz; null when pitch unavailable (never 0 Hz sentinel)",
    )
    f0_curve: TimeSeriesSummary | None = Field(
        default=None,
        description="Downsampled F0; unvoiced samples are JSON null",
    )
    voiced_probability_curve: TimeSeriesSummary | None = None
    note: str = (
        "Single-F0 estimate (pYIN). Does not represent all sources in polyphonic "
        "or mixed recordings."
    )


class HarmonicsSection(BaseModel):
    """Harmonic structure relative to estimated F0 (Milestone 3)."""

    max_harmonics: int = Field(description="Configured maximum harmonic index searched")
    harmonic_slots_available: float | None = Field(
        default=None,
        description=(
            "Mean over frames of min(max_harmonics, floor((Nyquist-ε)/F0)); "
            "denominator for harmonic_density"
        ),
    )
    frequency_resolution_hz: float | None = Field(
        default=None,
        description="STFT bin width Δf = sample_rate / n_fft (Hz)",
    )
    inharmonicity_resolution_floor: float | None = Field(
        default=None,
        description=(
            "Characteristic relative-frequency uncertainty ≈ mean Δf/(n·F0) "
            "for detected n≥2; differences below this are unresolved"
        ),
    )
    harmonic_count: float | None = Field(
        default=None, description="Mean detected harmonic count over voiced frames"
    )
    harmonic_count_median: float | None = None
    harmonic_density: float | None = Field(
        default=None,
        description=(
            "Mean detected_count / harmonic_slots_available (slot occupancy, 0–1); "
            "not harmonics per Hz"
        ),
    )
    peak_frequencies_hz: list[float | None] = Field(default_factory=list)
    relative_amplitudes: list[float | None] = Field(
        default_factory=list,
        description="Median peak magnitude / fundamental magnitude per harmonic index",
    )
    normalized_distribution: list[float] = Field(default_factory=list)
    harmonic_energy_fraction_estimate: float | None = Field(
        default=None,
        description=(
            "Linear power fraction E_harm/E_band in [0,1]. "
            "NOT HNR dB, not cepstral HNR, not calibrated SNR."
        ),
    )
    inharmonicity_estimate: float | None = Field(
        default=None,
        description="Mean |f_n/(n*F0)-1| for n≥2 (relative error); estimate only",
    )
    harmonic_count_frame: DistributionStats
    harmonic_energy_fraction_estimate_frame: DistributionStats
    inharmonicity_estimate_frame: DistributionStats
    confidence: float | None = None
    frames_analyzed: int = 0
    note: str = (
        "Harmonics are measured relative to a single estimated F0. Polyphonic "
        "mixtures may yield unreliable stacks. "
        "harmonic_energy_fraction_estimate is a linear [0,1] heuristic; "
        "inharmonicity_estimate is unresolved below inharmonicity_resolution_floor."
    )


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
        "Dataset-level normalization will be introduced when a real corpus exists. "
        "Ordering: indices 0–11 are the Milestone 2 spectral/energy prefix "
        "(unchanged scaling); indices 12+ are Milestone 3 pitch/harmonic appends."
    )


class AcousticFingerprint(BaseModel):
    """Versioned acoustic fingerprint document (Milestone 3)."""

    schema_version: str
    analysis_version: str
    source: SourceMetadata
    analysis_config: AnalysisConfigSnapshot
    quality: QualityMetrics
    spectral: SpectralSection
    pitch: PitchSection
    harmonics: HarmonicsSection
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
