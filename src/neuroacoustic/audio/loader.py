"""Audio decoding / loading via soundfile with FFmpeg fallback."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from neuroacoustic.audio.preprocessing import preprocess
from neuroacoustic.audio.probe import (
    ffmpeg_available,
    probe_audio,
    validate_audio_path,
)
from neuroacoustic.config import AppConfig
from neuroacoustic.exceptions import AudioDecodeError, DependencyError
from neuroacoustic.fingerprint.models import QualityMetrics, SourceMetadata
from neuroacoustic.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LoadedAudio:
    """In-memory decoded audio ready for analysis."""

    samples: NDArray[np.float32]
    native_sample_rate: int
    analysis_sample_rate: int
    source: SourceMetadata
    quality: QualityMetrics
    decode_backend: str


def _load_soundfile(path: Path) -> tuple[NDArray[np.floating], int]:
    try:
        data, rate = sf.read(str(path), always_2d=False)
    except Exception as exc:
        raise AudioDecodeError(f"soundfile failed to decode {path.name}: {exc}") from exc
    return np.asarray(data), int(rate)


def _load_via_ffmpeg(path: Path) -> tuple[NDArray[np.floating], int]:
    """Decode with FFmpeg to a temporary WAV, then read with soundfile.

    Uses argument arrays only (``shell=False``). Never interpolates the path
    into a shell string.
    """
    if not ffmpeg_available():
        raise DependencyError(
            "ffmpeg is required to decode this format. "
            "Install FFmpeg (system package) or convert to WAV/FLAC/AIFF."
        )

    with tempfile.TemporaryDirectory(prefix="neuroacoustic_ffmpeg_") as tmp:
        out_path = Path(tmp) / "decoded.wav"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-acodec",
            "pcm_f32le",
            str(out_path),
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                shell=False,
                timeout=300,
            )
        except OSError as exc:
            raise DependencyError(f"Failed to execute ffmpeg: {exc}") from exc

        if completed.returncode != 0:
            raise AudioDecodeError(
                f"ffmpeg failed to decode {path.name}: "
                f"{completed.stderr.strip() or 'unknown error'}"
            )
        return _load_soundfile(out_path)


def load_audio(path: Path | str, config: AppConfig) -> LoadedAudio:
    """Probe, decode, preprocess, and return analysis-ready audio.

    Prefer soundfile for WAV/FLAC/AIFF. Use FFmpeg fallback for MP3 and when
    soundfile cannot decode a supported container.
    """
    resolved = validate_audio_path(Path(path), config)
    probe = probe_audio(resolved, config)
    source = probe.source
    ext = resolved.suffix.lower().lstrip(".")

    decode_backend = "soundfile"
    if ext in config.audio.soundfile_extensions:
        try:
            raw, native_sr = _load_soundfile(resolved)
        except AudioDecodeError:
            logger.warning("soundfile decode failed for %s; trying ffmpeg", resolved.name)
            raw, native_sr = _load_via_ffmpeg(resolved)
            decode_backend = "ffmpeg"
    elif ext in config.audio.ffmpeg_extensions:
        raw, native_sr = _load_via_ffmpeg(resolved)
        decode_backend = "ffmpeg"
    else:
        raise AudioDecodeError(f"No decoder for extension '.{ext}'")

    if raw.size == 0:
        raise AudioDecodeError(f"Decoded audio is empty: {resolved.name}")

    samples, analysis_sr, quality = preprocess(raw, native_sr, config.audio)

    # Align source analysis rate with what preprocessing actually used.
    source = source.model_copy(
        update={
            "native_sample_rate": native_sr,
            "analysis_sample_rate": analysis_sr,
            "channels": 1 if samples.ndim == 1 else int(samples.shape[-1]),
        }
    )

    logger.info(
        "Loaded %s samples=%s native_sr=%s analysis_sr=%s backend=%s",
        source.filename,
        samples.shape,
        native_sr,
        analysis_sr,
        decode_backend,
    )
    return LoadedAudio(
        samples=samples,
        native_sample_rate=native_sr,
        analysis_sample_rate=analysis_sr,
        source=source,
        quality=quality,
        decode_backend=decode_backend,
    )
