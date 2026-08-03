"""Configuration loading from TOML with environment overrides."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from neuroacoustic.exceptions import ConfigError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _repo_root() -> Path:
    """Return repository root assuming package lives in ``src/neuroacoustic``."""
    return Path(__file__).resolve().parents[2]


DEFAULT_CONFIG_PATH = _repo_root() / "config" / "default.toml"


class ProjectConfig(BaseModel):
    name: str = "neuroacoustic-lab"
    analysis_version: str = "0.4.0"
    schema_version: str = "0.4.0"


class PathsConfig(BaseModel):
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")
    database: Path = Path("data/db/neuroacoustic.sqlite")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = False


class AudioConfig(BaseModel):
    supported_extensions: list[str] = Field(
        default_factory=lambda: ["wav", "flac", "aiff", "aif", "mp3"]
    )
    soundfile_extensions: list[str] = Field(
        default_factory=lambda: ["wav", "flac", "aiff", "aif"]
    )
    ffmpeg_extensions: list[str] = Field(default_factory=lambda: ["mp3"])
    analysis_sample_rate: int | None = None
    silence_peak_threshold: float = 1.0e-6
    silence_sample_threshold: float = 1.0e-4
    clipping_threshold: float = 0.999
    max_duration_seconds: float = 3600.0
    min_duration_seconds: float = 0.01

    @field_validator("supported_extensions", "soundfile_extensions", "ffmpeg_extensions")
    @classmethod
    def _normalize_ext(cls, value: list[str]) -> list[str]:
        return [ext.lower().lstrip(".") for ext in value]


class ProbeConfig(BaseModel):
    prefer_ffprobe: bool = True


class AnalysisConfig(BaseModel):
    """Frame-level analysis parameters (Milestone 2–4A)."""

    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    frame_length: int | None = None  # optional alias; prefer win_length
    rolloff_percentile: float = 0.85
    n_contrast_bands: int = 6
    fft_peak_count: int = 8
    max_timeseries_points: int = 256
    fft_summary_bins: int = 128
    amplitude_floor: float = 1.0e-12
    # Pitch (pYIN)
    f0_min_hz: float = 65.0
    f0_max_hz: float = 2000.0
    f0_frame_length: int = 2048
    # Soft confidence warning threshold (high-confidence voicing)
    voiced_prob_threshold: float = 0.5
    # Hard floor for treating a frame as voiced in F0 aggregates
    voiced_prob_floor: float = 0.1
    min_voiced_frames: int = 3
    # Harmonics
    max_harmonics: int = 12
    harmonic_search_width_fraction: float = 0.10
    harmonic_rel_amp_threshold: float = 0.05
    # Envelope / ADSR estimates
    envelope_smooth_frames: int = 5
    envelope_onset_ratio: float = 0.1
    envelope_attack_high_ratio: float = 0.9
    envelope_sustain_start: float = 0.35
    envelope_sustain_end: float = 0.75
    envelope_release_ratio: float = 0.1
    # Rhythm
    min_tempo_bpm: float = 40.0
    max_tempo_bpm: float = 240.0
    min_onset_events_for_tempo: int = 4
    min_duration_for_tempo_seconds: float = 0.5
    # Evidence gate: enough placed beats + stable intervals + onset periodicity
    min_beats_for_tempo: int = 3
    max_beat_interval_cv: float = 0.35
    # Normalized onset-envelope autocorr at one beat-period lag (see rhythm.py)
    min_tempo_periodicity: float = 0.30
    # File-tail decay / reverberation heuristic
    decay_fit_db_range: float = 20.0
    decay_min_r2: float = 0.85
    decay_min_duration_seconds: float = 0.05
    vector_version: str = "0.4.0-preliminary"

    def effective_win_length(self) -> int:
        return int(self.win_length or self.frame_length or self.n_fft)


class AppConfig(BaseModel):
    """Application-wide configuration."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    config_path: Path | None = None

    def resolve_paths(self, base: Path | None = None) -> AppConfig:
        """Resolve relative paths against ``base`` (default: repo root / cwd)."""
        root = base or _repo_root()
        data = self.model_dump()
        paths = data["paths"]
        for key in ("input_dir", "output_dir", "database"):
            p = Path(paths[key])
            if not p.is_absolute():
                paths[key] = str((root / p).resolve())
        return AppConfig.model_validate({**data, "config_path": self.config_path})


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply ``NEUROACOUSTIC_*`` environment overrides."""
    paths = raw.setdefault("paths", {})
    audio = raw.setdefault("audio", {})
    logging_cfg = raw.setdefault("logging", {})

    if db := os.environ.get("NEUROACOUSTIC_DATABASE"):
        paths["database"] = db
    if out := os.environ.get("NEUROACOUSTIC_OUTPUT_DIR"):
        paths["output_dir"] = out
    if level := os.environ.get("NEUROACOUSTIC_LOG_LEVEL"):
        logging_cfg["level"] = level
    rate = os.environ.get("NEUROACOUSTIC_ANALYSIS_SAMPLE_RATE")
    if rate is not None and rate.strip() != "":
        try:
            audio["analysis_sample_rate"] = int(rate)
        except ValueError as exc:
            raise ConfigError(
                f"NEUROACOUSTIC_ANALYSIS_SAMPLE_RATE must be an integer, got {rate!r}"
            ) from exc
    return raw


def load_config(path: Path | str | None = None, *, resolve: bool = True) -> AppConfig:
    """Load configuration from TOML and environment overrides."""
    env_path = os.environ.get("NEUROACOUSTIC_CONFIG")
    config_path = Path(path or env_path or DEFAULT_CONFIG_PATH).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    raw = _load_toml(config_path) if config_path.exists() else {}
    # Map legacy frame_length key onto win_length / n_fft if present.
    analysis = raw.setdefault("analysis", {})
    if "frame_length" in analysis and "win_length" not in analysis:
        analysis["win_length"] = analysis["frame_length"]
    if "frame_length" in analysis and "n_fft" not in analysis:
        analysis["n_fft"] = analysis["frame_length"]

    raw = _apply_env_overrides(raw)
    cfg = AppConfig.model_validate({**raw, "config_path": config_path})
    if resolve:
        cfg = cfg.resolve_paths(_repo_root())
    return cfg
