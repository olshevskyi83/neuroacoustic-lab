"""Audio probing: validation, hashing, and container metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import soundfile as sf

from neuroacoustic.config import AppConfig
from neuroacoustic.exceptions import AudioValidationError, DependencyError
from neuroacoustic.fingerprint.models import (
    ProbeBackend,
    ProbeResult,
    SourceMetadata,
    build_analysis_config_snapshot,
)
from neuroacoustic.logging import get_logger

logger = get_logger(__name__)

_EXTENSION_CONTAINER = {
    "wav": "wav",
    "flac": "flac",
    "aiff": "aiff",
    "aif": "aiff",
    "mp3": "mp3",
}

_EXTENSION_CODEC = {
    "wav": "pcm",
    "flac": "flac",
    "aiff": "pcm",
    "aif": "pcm",
    "mp3": "mp3",
}


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def validate_audio_path(path: Path, config: AppConfig) -> Path:
    """Validate existence and supported extension. Never executes the path as shell."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise AudioValidationError(f"File does not exist: {resolved}")
    if not resolved.is_file():
        raise AudioValidationError(f"Not a regular file: {resolved}")

    ext = resolved.suffix.lower().lstrip(".")
    if ext not in config.audio.supported_extensions:
        raise AudioValidationError(
            f"Unsupported extension '.{ext}'. "
            f"Supported: {', '.join('.' + e for e in config.audio.supported_extensions)}"
        )
    return resolved


def _ffprobe_json(path: Path) -> dict:
    """Run ffprobe with argument array (no shell)."""
    if not ffprobe_available():
        raise DependencyError(
            "ffprobe is not installed. Install FFmpeg to probe MP3 and rich metadata."
        )
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=60,
        )
    except OSError as exc:
        raise DependencyError(f"Failed to execute ffprobe: {exc}") from exc

    if completed.returncode != 0:
        raise AudioValidationError(
            f"ffprobe failed for {path.name}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AudioValidationError(f"ffprobe returned invalid JSON for {path.name}") from exc


def _metadata_from_ffprobe(data: dict) -> dict:
    streams = data.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    stream = audio_streams[0] if audio_streams else {}
    fmt = data.get("format") or {}

    sample_rate = stream.get("sample_rate")
    channels = stream.get("channels")
    duration = stream.get("duration") or fmt.get("duration")
    bit_rate = stream.get("bit_rate") or fmt.get("bit_rate")
    bits_per_sample = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample")

    def _int_or_none(value: object) -> int | None:
        if value is None or value == "N/A":
            return None
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None

    def _float_or_none(value: object) -> float | None:
        if value is None or value == "N/A":
            return None
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return None

    return {
        "container": fmt.get("format_name"),
        "codec": stream.get("codec_name"),
        "duration_seconds": _float_or_none(duration),
        "native_sample_rate": _int_or_none(sample_rate),
        "channels": _int_or_none(channels),
        "bit_depth": _int_or_none(bits_per_sample),
        "bitrate": _int_or_none(bit_rate),
    }


def _metadata_from_soundfile(path: Path) -> dict:
    try:
        info = sf.info(str(path))
    except Exception as exc:  # soundfile raises various RuntimeErrors
        raise AudioValidationError(f"soundfile could not read metadata for {path.name}: {exc}") from exc

    subtype = getattr(info, "subtype", None) or ""
    bit_depth = _bit_depth_from_subtype(subtype)

    return {
        "container": (info.format or "").lower() or None,
        "codec": subtype.lower() if subtype else None,
        "duration_seconds": float(info.duration) if info.duration is not None else None,
        "native_sample_rate": int(info.samplerate) if info.samplerate else None,
        "channels": int(info.channels) if info.channels else None,
        "bit_depth": bit_depth,
        "bitrate": None,
    }


def _bit_depth_from_subtype(subtype: str) -> int | None:
    mapping = {
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "PCM_S8": 8,
        "PCM_U8": 8,
        "FLOAT": 32,
        "DOUBLE": 64,
    }
    key = subtype.upper()
    for prefix, depth in mapping.items():
        if key.startswith(prefix):
            return depth
    return None


def _validate_duration(duration: float | None, config: AppConfig, warnings: list[str]) -> None:
    if duration is None:
        warnings.append("Duration could not be determined from metadata")
        return
    if duration < config.audio.min_duration_seconds:
        raise AudioValidationError(
            f"Duration {duration:.6f}s is below minimum "
            f"{config.audio.min_duration_seconds}s"
        )
    if duration > config.audio.max_duration_seconds:
        warnings.append(
            f"Duration {duration:.3f}s exceeds configured max "
            f"{config.audio.max_duration_seconds}s; use --force in analyze (later)"
        )


def probe_audio(path: Path | str, config: AppConfig) -> ProbeResult:
    """Validate and probe an audio file without decoding all samples.

    Uses soundfile for WAV/FLAC/AIFF and ffprobe for MP3 (and optionally to
    enrich metadata when available).
    """
    resolved = validate_audio_path(Path(path), config)
    ext = resolved.suffix.lower().lstrip(".")
    warnings: list[str] = []

    content_hash = sha256_file(resolved)
    file_size = resolved.stat().st_size

    meta: dict = {
        "container": _EXTENSION_CONTAINER.get(ext),
        "codec": _EXTENSION_CODEC.get(ext),
        "duration_seconds": None,
        "native_sample_rate": None,
        "channels": None,
        "bit_depth": None,
        "bitrate": None,
    }
    backend = ProbeBackend.EXTENSION

    uses_soundfile = ext in config.audio.soundfile_extensions
    uses_ffmpeg = ext in config.audio.ffmpeg_extensions

    if uses_soundfile:
        sf_meta = _metadata_from_soundfile(resolved)
        meta.update({k: v for k, v in sf_meta.items() if v is not None})
        backend = ProbeBackend.SOUNDFILE

        if config.probe.prefer_ffprobe and ffprobe_available():
            try:
                ff_meta = _metadata_from_ffprobe(_ffprobe_json(resolved))
                # Prefer ffprobe for container/codec/bitrate when present.
                for key in ("container", "codec", "bitrate", "bit_depth"):
                    if ff_meta.get(key) is not None:
                        meta[key] = ff_meta[key]
                backend = ProbeBackend.HYBRID
            except (DependencyError, AudioValidationError) as exc:
                warnings.append(f"ffprobe enrichment skipped: {exc}")

    elif uses_ffmpeg:
        if not ffprobe_available():
            raise DependencyError(
                "ffprobe is required to probe MP3 files. "
                "Install FFmpeg (system package) or convert to WAV/FLAC."
            )
        ff_meta = _metadata_from_ffprobe(_ffprobe_json(resolved))
        meta.update({k: v for k, v in ff_meta.items() if v is not None})
        backend = ProbeBackend.FFPROBE
    else:
        raise AudioValidationError(f"No probe backend for extension '.{ext}'")

    if meta["native_sample_rate"] is not None and meta["native_sample_rate"] <= 0:
        raise AudioValidationError(f"Invalid sample rate: {meta['native_sample_rate']}")
    if meta["channels"] is not None and meta["channels"] < 1:
        raise AudioValidationError(f"Invalid channel count: {meta['channels']}")

    _validate_duration(meta["duration_seconds"], config, warnings)

    analysis_rate = config.audio.analysis_sample_rate or meta["native_sample_rate"]

    source = SourceMetadata(
        filename=resolved.name,
        path=str(resolved),
        content_hash=content_hash,
        file_size_bytes=file_size,
        container=meta["container"],
        codec=meta["codec"],
        duration_seconds=meta["duration_seconds"],
        native_sample_rate=meta["native_sample_rate"],
        analysis_sample_rate=analysis_rate,
        channels=meta["channels"],
        bit_depth=meta["bit_depth"],
        bitrate=meta["bitrate"],
        probe_backend=backend,
        warnings=warnings,
    )

    analysis_config = build_analysis_config_snapshot(
        config, analysis_sample_rate=analysis_rate
    )

    logger.info(
        "Probed %s hash=%s duration=%s rate=%s channels=%s backend=%s",
        source.filename,
        source.content_hash[:12],
        source.duration_seconds,
        source.native_sample_rate,
        source.channels,
        source.probe_backend.value,
    )
    return ProbeResult(source=source, analysis_config=analysis_config)
