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
    # Milestone 4A
    envelope_smooth_frames: int | None = None
    min_tempo_bpm: float | None = None
    max_tempo_bpm: float | None = None
    min_beats_for_tempo: int | None = None
    max_beat_interval_cv: float | None = None
    min_tempo_periodicity: float | None = None
    decay_fit_db_range: float | None = None
    decay_min_r2: float | None = None


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
        envelope_smooth_frames=config.analysis.envelope_smooth_frames,
        min_tempo_bpm=config.analysis.min_tempo_bpm,
        max_tempo_bpm=config.analysis.max_tempo_bpm,
        min_beats_for_tempo=config.analysis.min_beats_for_tempo,
        max_beat_interval_cv=config.analysis.max_beat_interval_cv,
        min_tempo_periodicity=config.analysis.min_tempo_periodicity,
        decay_fit_db_range=config.analysis.decay_fit_db_range,
        decay_min_r2=config.analysis.decay_min_r2,
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


class EnvelopeSection(BaseModel):
    """Smoothed RMS envelope and ADSR-like *estimates* (Milestone 4A)."""

    onset_time_s: float | None = Field(
        default=None, description="First time envelope reaches onset_ratio * peak (s)"
    )
    attack_time_s: float | None = Field(
        default=None, description="Onset → attack_high_ratio * peak (s); estimate"
    )
    decay_time_s: float | None = Field(
        default=None, description="Peak → sustain absolute level (s); estimate"
    )
    sustain_level: float | None = Field(
        default=None,
        description="Median mid-file envelope / peak (relative, ~0–1); estimate",
    )
    release_time_s: float | None = Field(
        default=None, description="Late fall from mid-sustain to release_ratio * peak (s)"
    )
    peak_envelope: float | None = None
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    envelope_rms: DistributionStats
    envelope_curve: TimeSeriesSummary | None = None
    note: str = (
        "ADSR fields describe envelope shape only. They are not a recovered "
        "synthesizer ADSR program."
    )


class StereoSection(BaseModel):
    """Inter-channel stereo measurements (Milestone 4A)."""

    channel_count: int = 1
    is_mono: bool = True
    left_rms: float | None = None
    right_rms: float | None = None
    correlation: float | None = Field(
        default=None, description="Pearson correlation between L and R in [-1, 1]"
    )
    mid_energy: float | None = Field(
        default=None, description="mean(((L+R)/2)^2)"
    )
    side_energy: float | None = Field(
        default=None, description="mean(((L-R)/2)^2)"
    )
    side_to_mid_ratio: float | None = Field(
        default=None,
        description="side_energy / max(mid_energy, floor); dimensionless",
    )
    stereo_width_estimate: float | None = Field(
        default=None,
        description=(
            "clip(0.5*(1-correlation), 0, 1): correlation/phase-opposition "
            "heuristic — not a complete perceptual stereo-width metric; "
            "null for mono / silent channel"
        ),
    )
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    note: str = (
        "Mono files leave correlation/width null (not fabricated). "
        "Width uses the documented 0.5*(1-corr) correlation/phase-opposition "
        "heuristic, not a full perceptual stereo-width metric."
    )


class RhythmSection(BaseModel):
    """Onset strength and optional tempo/beats (Milestone 4A)."""

    onset_strength_mean: float | None = None
    onset_strength_std: float | None = None
    onset_event_count: int = 0
    tempo_bpm: float | None = Field(
        default=None,
        description="Estimated tempo in BPM; null when evidence is insufficient",
    )
    beat_times_s: list[float] = Field(default_factory=list)
    beat_count: int = 0
    beat_interval_cv: float | None = Field(
        default=None,
        description="Coefficient of variation of successive beat intervals; null if <2 intervals",
    )
    tempo_periodicity: float | None = Field(
        default=None,
        description=(
            "Normalized onset-envelope autocorrelation at one beat-period lag R(τ); "
            "primary periodicity evidence for the reliability gate"
        ),
    )
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    onset_strength: DistributionStats
    onset_strength_curve: TimeSeriesSummary | None = None
    note: str = (
        "Tempo may be off by ×2 (half-/double-time). Null tempo means "
        "insufficient rhythmic evidence — not a missing zero BPM. "
        "Accepted only when R(τ) ≥ min_tempo_periodicity with enough stable beats."
    )


class ReverberationSection(BaseModel):
    """File-tail decay heuristic (Milestone 4A) — not room RT60."""

    tail_decay_t60_estimate_seconds: float | None = Field(
        default=None,
        description=(
            "Extrapolated time for 60 dB drop of fitted energy-envelope slope; "
            "file-tail heuristic, NOT calibrated room RT60"
        ),
    )
    decay_slope_db_per_s: float | None = Field(
        default=None, description="Fitted energy-envelope slope (10*log10 energy) in dB/s"
    )
    fit_r_squared: float | None = None
    fit_db_range: float | None = Field(
        default=None, description="Measured post-peak energy drop used in the fit (dB)"
    )
    analyzed_frequency_range_hz: list[float] | None = Field(
        default=None,
        description="Broadband analysis band [low, high] Hz (fullband envelope)",
    )
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    note: str = (
        "Distinguishes file-tail decay from room reverberation. "
        "Do not treat as exact RT60 for arbitrary mixed recordings."
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
        "Indices 0–11: M2 spectral/energy (unchanged). "
        "Indices 12–21: M3 pitch/harmonics (unchanged). "
        "Indices 22+: M4A envelope/stereo/rhythm/decay appends."
    )


class AcousticFingerprint(BaseModel):
    """Versioned acoustic fingerprint document (Milestone 4A)."""

    schema_version: str
    analysis_version: str
    source: SourceMetadata
    analysis_config: AnalysisConfigSnapshot
    quality: QualityMetrics
    spectral: SpectralSection
    pitch: PitchSection
    harmonics: HarmonicsSection
    energy: EnergySection
    envelope: EnvelopeSection
    stereo: StereoSection
    rhythm: RhythmSection
    reverberation: ReverberationSection
    vector: list[float] = Field(default_factory=list)
    vector_meta: PreliminaryVector | None = None
    artifacts: ArtifactPaths = Field(default_factory=ArtifactPaths)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
