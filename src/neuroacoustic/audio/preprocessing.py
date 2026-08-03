"""Channel-aware preprocessing: float conversion and optional resampling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neuroacoustic.config import AudioConfig
from neuroacoustic.exceptions import AudioValidationError
from neuroacoustic.fingerprint.models import QualityMetrics


def to_float32(samples: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float32]:
    """Convert samples to float32 in approximately ``[-1, 1]`` for integer PCM.

    Floating-point sources are cast without peak normalization so measured
    levels are preserved. Integer PCM is scaled by the dtype full-scale value.
    """
    if np.issubdtype(samples.dtype, np.floating):
        out = np.asarray(samples, dtype=np.float32)
    elif np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), info.max))
        out = samples.astype(np.float32) / scale
    else:
        raise AudioValidationError(f"Unsupported sample dtype: {samples.dtype}")

    if not np.isfinite(out).all():
        raise AudioValidationError("Audio contains NaN or Inf samples after conversion")
    return out


def ensure_channel_axis(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return shape ``(n_samples, n_channels)``."""
    if samples.ndim == 1:
        return samples.reshape(-1, 1)
    if samples.ndim == 2:
        return samples
    raise AudioValidationError(f"Expected 1-D or 2-D audio, got shape {samples.shape}")


def resample_audio(
    samples: NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
) -> NDArray[np.float32]:
    """Resample multi-channel audio to ``target_sr`` using librosa (per channel).

    Only used when an explicit analysis sample rate differs from the native rate.
    """
    if orig_sr == target_sr:
        return samples
    if target_sr <= 0 or orig_sr <= 0:
        raise AudioValidationError(f"Invalid sample rates: {orig_sr} -> {target_sr}")

    import librosa

    shaped = ensure_channel_axis(samples)
    channels = [
        librosa.resample(shaped[:, ch], orig_sr=orig_sr, target_sr=target_sr)
        for ch in range(shaped.shape[1])
    ]
    stacked = np.stack(channels, axis=1).astype(np.float32)
    if samples.ndim == 1:
        return stacked[:, 0]
    return stacked


def compute_quality_metrics(
    samples: NDArray[np.float32],
    audio_cfg: AudioConfig,
) -> QualityMetrics:
    """Compute basic load-time quality metrics without destructive normalization."""
    flat = samples.reshape(-1)
    peak = float(np.max(np.abs(flat))) if flat.size else 0.0
    clipping_ratio = (
        float(np.mean(np.abs(flat) >= audio_cfg.clipping_threshold)) if flat.size else 0.0
    )
    silence_ratio = (
        float(np.mean(np.abs(flat) < audio_cfg.silence_sample_threshold)) if flat.size else 1.0
    )

    warnings: list[str] = []
    if flat.size == 0:
        warnings.append("Empty sample buffer")
    if peak < audio_cfg.silence_peak_threshold:
        warnings.append(
            f"Near-silent signal (peak={peak:.3e} < "
            f"threshold={audio_cfg.silence_peak_threshold:.3e})"
        )
    if clipping_ratio > 0.0:
        warnings.append(f"Possible clipping: clipping_ratio={clipping_ratio:.6f}")

    return QualityMetrics(
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        peak_amplitude=peak,
        warnings=warnings,
    )


def preprocess(
    samples: NDArray[np.floating] | NDArray[np.integer],
    native_sr: int,
    audio_cfg: AudioConfig,
) -> tuple[NDArray[np.float32], int, QualityMetrics]:
    """Float-convert, optionally resample, and compute quality metrics.

    Returns
    -------
    samples_f32, analysis_sample_rate, quality
    """
    f32 = to_float32(np.asarray(samples))
    analysis_sr = audio_cfg.analysis_sample_rate or native_sr
    if analysis_sr != native_sr:
        f32 = resample_audio(f32, native_sr, analysis_sr)
    quality = compute_quality_metrics(f32, audio_cfg)
    return f32, analysis_sr, quality
