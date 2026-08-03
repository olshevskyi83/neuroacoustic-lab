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
    analysis_version: str = "0.1.0"
    schema_version: str = "0.1.0"


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
    """Reserved frame parameters for Milestone 2+."""

    frame_length: int | None = None
    hop_length: int | None = None


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
    """Load configuration from TOML and environment overrides.

    Parameters
    ----------
    path:
        Explicit config path. If omitted, uses ``NEUROACOUSTIC_CONFIG`` or
        ``config/default.toml``.
    resolve:
        When True, resolve relative data paths against the repository root.
    """
    env_path = os.environ.get("NEUROACOUSTIC_CONFIG")
    config_path = Path(path or env_path or DEFAULT_CONFIG_PATH).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    raw = _load_toml(config_path) if config_path.exists() else {}
    raw = _apply_env_overrides(raw)
    cfg = AppConfig.model_validate({**raw, "config_path": config_path})
    if resolve:
        cfg = cfg.resolve_paths(_repo_root())
    return cfg
