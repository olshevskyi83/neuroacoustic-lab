"""Stereo / channel-relationship measurements.

Input layout: ``(n_samples,)`` mono or ``(n_samples, n_channels)`` as produced
by the loader (soundfile convention).

Mid/side convention
-------------------
- Mid  m = (L + R) / 2
- Side s = (L - R) / 2
- mid_energy  = mean(m²)
- side_energy = mean(s²)
- side_to_mid_ratio = side_energy / mid_energy when mid is usable; **null** when
  mid energy is negligible (e.g. polarity-inverted identical channels).

Stereo width estimate (correlation / phase-opposition heuristic)
----------------------------------------------------------------
When both channels have usable energy:
    stereo_width_estimate = clip(0.5 * (1 - pearson_correlation), 0, 1)

This is a **correlation-based width / phase-opposition estimate**, not a
complete perceptual stereo-width metric (it ignores frequency-dependent
coherence, spaciousness, and level differences beyond what correlation
captures).

Approximate interpretation:
- identical channels → corr ≈ +1 → width ≈ 0 (mono-compatible)
- polarity-inverted → corr ≈ −1 → width ≈ 1 (phase opposition; mono sum cancels)
- uncorrelated → corr ≈ 0 → width ≈ 0.5

Mono files set ``is_mono=True`` and leave width/correlation null (not fabricated).
Silent or one-sided channels suppress width rather than inventing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neuroacoustic.config import AnalysisConfig
from neuroacoustic.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class StereoFeatures:
    """Inter-channel stereo measurements."""

    channel_count: int
    is_mono: bool
    left_rms: float | None
    right_rms: float | None
    correlation: float | None
    mid_energy: float | None
    side_energy: float | None
    side_to_mid_ratio: float | None
    stereo_width_estimate: float | None
    confidence: float | None
    warnings: list[str] = field(default_factory=list)
    note: str = (
        "stereo_width_estimate = 0.5*(1-corr) is a correlation/phase-opposition "
        "heuristic in [0,1], not a full perceptual stereo-width metric. "
        "Mono files leave width null. "
        "side_to_mid_ratio = side_energy/mid_energy when mid is usable."
    )


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def _pearson(a: np.ndarray, b: np.ndarray, floor: float) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 2:
        return None
    a = a[:n]
    b = b[:n]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < floor or sb < floor:
        return None
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    if denom < floor:
        return None
    corr = float(np.sum(a0 * b0) / denom)
    if corr != corr:
        return None
    return float(np.clip(corr, -1.0, 1.0))


def compute_stereo_features(
    samples: np.ndarray,
    sample_rate: int,
    analysis: AnalysisConfig,
) -> StereoFeatures:
    """Compute stereo metrics from ``(n_samples,)`` or ``(n_samples, n_channels)``."""
    del sample_rate
    floor = float(analysis.amplitude_floor)
    arr = np.asarray(samples, dtype=np.float64)
    warnings: list[str] = []

    if arr.ndim == 1:
        return StereoFeatures(
            channel_count=1,
            is_mono=True,
            left_rms=_rms(arr),
            right_rms=None,
            correlation=None,
            mid_energy=None,
            side_energy=None,
            side_to_mid_ratio=None,
            stereo_width_estimate=None,
            confidence=1.0,
            warnings=["mono_file"],
        )

    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio, got shape {arr.shape}")

    n_ch = int(arr.shape[1])
    if n_ch <= 1:
        left = arr[:, 0] if arr.size else np.array([], dtype=np.float64)
        return StereoFeatures(
            channel_count=max(1, n_ch),
            is_mono=True,
            left_rms=_rms(left),
            right_rms=None,
            correlation=None,
            mid_energy=None,
            side_energy=None,
            side_to_mid_ratio=None,
            stereo_width_estimate=None,
            confidence=1.0,
            warnings=["mono_file"],
        )

    left = arr[:, 0]
    right = arr[:, 1]
    if n_ch > 2:
        warnings.append("extra_channels_ignored")

    left_rms = _rms(left)
    right_rms = _rms(right)
    silent_floor = max(floor * 1e3, 1e-8)

    if left_rms < silent_floor and right_rms < silent_floor:
        warnings.append("silent_stereo")
        return StereoFeatures(
            channel_count=n_ch,
            is_mono=False,
            left_rms=left_rms,
            right_rms=right_rms,
            correlation=None,
            mid_energy=0.0,
            side_energy=0.0,
            side_to_mid_ratio=None,
            stereo_width_estimate=None,
            confidence=0.0,
            warnings=warnings,
        )

    one_silent = left_rms < silent_floor or right_rms < silent_floor
    if one_silent:
        warnings.append("one_channel_near_silent")

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_energy = float(np.mean(mid * mid))
    side_energy = float(np.mean(side * side))

    total_ms = mid_energy + side_energy
    if total_ms < floor or mid_energy < 1e-6 * max(total_ms, floor):
        warnings.append("mid_energy_near_zero")
        ratio = None
    else:
        ratio = float(side_energy / mid_energy)

    corr = _pearson(left, right, floor)
    width: float | None = None
    conf = 0.7
    if one_silent:
        corr = None
        width = None
        conf = 0.2
        warnings.append("stereo_width_suppressed_silent_channel")
    elif corr is None:
        warnings.append("correlation_undefined")
        conf = 0.3
    else:
        width = float(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        conf = 0.9
        if corr > 0.98:
            warnings.append("channels_nearly_identical")
        elif corr < -0.98:
            warnings.append("channels_nearly_inverted")
            warnings.append("phase_opposition")
            warnings.append("mono_compatibility_risk")

    logger.debug(
        "stereo L=%.4g R=%.4g corr=%s width=%s side/mid=%s",
        left_rms,
        right_rms,
        corr,
        width,
        ratio,
    )

    return StereoFeatures(
        channel_count=n_ch,
        is_mono=False,
        left_rms=left_rms,
        right_rms=right_rms,
        correlation=corr,
        mid_energy=mid_energy,
        side_energy=side_energy,
        side_to_mid_ratio=ratio,
        stereo_width_estimate=width,
        confidence=float(conf),
        warnings=warnings,
    )
