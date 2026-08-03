"""Fundamental frequency (F0) and voicing analysis via librosa pYIN.

Algorithm
---------
Uses probabilistic YIN (pYIN; Mauch & Dixon 2014) as implemented by
``librosa.pyin``. For each analysis frame pYIN returns:

- ``f0`` (Hz) or NaN when unvoiced / unavailable
- ``voiced_flag`` (bool)
- ``voiced_prob`` in ``[0, 1]`` (voicing probability / confidence)

Configurable ``f0_min_hz`` / ``f0_max_hz`` bound the search range.

Aggregation
-----------
Voiced-F0 statistics (mean, std, median, percentiles, min, max) are computed
**only** over frames that are voiced (``voiced_flag``), have finite ``f0``,
and satisfy ``voiced_prob >= voiced_prob_threshold``.

When no such frames exist, all F0 aggregates are ``null`` — never fabricated
as 0 Hz.

Limitations
-----------
- pYIN estimates a **single** F0 trajectory. Polyphonic or mixed sources are
  not fully described by one F0; results may lock onto one source, octave
  errors, or become unstable. Do not claim a single F0 represents all sources.
- Short / silent / noise-like inputs often yield low voiced ratios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuroacoustic.analysis.stats import DistributionStats, downsample_series, sanitize_array
from neuroacoustic.config import AnalysisConfig


@dataclass
class PitchFeatures:
    method: str
    f0_min_hz: float
    f0_max_hz: float
    frame_count: int
    voiced_frame_count: int
    voiced_ratio: float | None
    mean_voiced_probability: float | None
    f0_voiced: DistributionStats
    f0_median_hz: float | None
    f0_curve_times: list[float]
    f0_curve_values: list[float | None]
    voiced_prob_curve_times: list[float]
    voiced_prob_curve_values: list[float]
    times_seconds: NDArray[np.float64]
    f0_hz: NDArray[np.float64]  # NaN where unvoiced
    voiced_flag: NDArray[np.bool_]
    voiced_probability: NDArray[np.float64]
    confidence: float | None
    warnings: list[str]


def analyze_pitch(
    mono: NDArray[np.floating],
    sample_rate: int,
    analysis: AnalysisConfig,
) -> PitchFeatures:
    """Estimate F0 / voicing with pYIN; return aggregates and compact curves."""
    import librosa

    warnings: list[str] = []
    y = sanitize_array(mono)
    fmin = float(analysis.f0_min_hz)
    fmax = float(min(analysis.f0_max_hz, sample_rate / 2.0 - 1.0))
    hop = int(analysis.hop_length)
    frame_length = int(analysis.f0_frame_length)

    empty_stats = DistributionStats.from_values([])
    if y.size < max(frame_length // 4, 16) or fmax <= fmin:
        warnings.append("Signal too short or invalid F0 range for pitch analysis")
        return PitchFeatures(
            method="librosa.pyin",
            f0_min_hz=fmin,
            f0_max_hz=fmax,
            frame_count=0,
            voiced_frame_count=0,
            voiced_ratio=None,
            mean_voiced_probability=None,
            f0_voiced=empty_stats,
            f0_median_hz=None,
            f0_curve_times=[],
            f0_curve_values=[],
            voiced_prob_curve_times=[],
            voiced_prob_curve_values=[],
            times_seconds=np.asarray([], dtype=np.float64),
            f0_hz=np.asarray([], dtype=np.float64),
            voiced_flag=np.asarray([], dtype=bool),
            voiced_probability=np.asarray([], dtype=np.float64),
            confidence=None,
            warnings=warnings,
        )

    if float(np.max(np.abs(y))) < analysis.amplitude_floor * 10:
        warnings.append("Near-silent input: pitch analysis likely unreliable")

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y.astype(np.float32),
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop,
        fill_na=np.nan,
    )
    f0 = np.asarray(f0, dtype=np.float64)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_prob = sanitize_array(np.asarray(voiced_prob, dtype=np.float64))
    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop).astype(np.float64)

    # Voicing: pYIN voiced_flag + finite F0 + probability floor.
    # Low-F0 tones often have mean prob ~0.4 (below the warning threshold 0.5)
    # but well above the floor. Spurious noise flags typically sit near ~0.01.
    floor = float(analysis.voiced_prob_floor)
    usable = (
        voiced_flag
        & np.isfinite(f0)
        & (f0 > 0)
        & (voiced_prob >= floor)
    )
    voiced_f0 = f0[usable]
    stats = DistributionStats.from_values(voiced_f0)
    voiced_count = int(np.count_nonzero(usable))
    frame_count = int(f0.size)
    voiced_ratio = float(voiced_count / frame_count) if frame_count else None
    mean_prob = float(np.mean(voiced_prob[usable])) if voiced_count else None

    high_conf_count = int(
        np.count_nonzero(usable & (voiced_prob >= float(analysis.voiced_prob_threshold)))
    )

    if voiced_count == 0:
        warnings.append("No voiced frames with reliable F0 (pitch unavailable)")
        f0_median = None
        confidence = 0.0
    else:
        f0_median = stats.median
        # Confidence blends mean voicing probability with coverage.
        coverage = min(1.0, voiced_count / max(analysis.min_voiced_frames, 1))
        confidence = float(
            np.clip((mean_prob or 0.0) * 0.7 + coverage * 0.3, 0.0, 1.0)
        )
        if voiced_count < analysis.min_voiced_frames:
            warnings.append(
                f"Insufficient voiced frames for stable pitch "
                f"({voiced_count} < {analysis.min_voiced_frames})"
            )
        if mean_prob is not None and mean_prob < float(analysis.voiced_prob_threshold):
            warnings.append(
                f"Mean voiced probability {mean_prob:.3f} below threshold "
                f"{analysis.voiced_prob_threshold:.3f} — treat F0 with caution"
            )
        if high_conf_count < analysis.min_voiced_frames and voiced_count >= analysis.min_voiced_frames:
            warnings.append(
                "Few high-probability voiced frames; F0 may be less reliable"
            )

    # Noise-like heuristic: very low voiced ratio on non-silent audio.
    peak = float(np.max(np.abs(y)))
    if peak >= 1e-4 and voiced_ratio is not None and voiced_ratio < 0.05:
        warnings.append("Noise-like or unpitched input: very low voiced-frame ratio")

    # Curves: F0 uses null for unvoiced/unavailable (never coerce to 0 Hz).
    max_pts = analysis.max_timeseries_points
    f0_curve_t, f0_curve_v = _downsample_f0(times, f0, usable, max_pts)
    p_t, p_v = downsample_series(times, voiced_prob, max_pts)

    return PitchFeatures(
        method="librosa.pyin",
        f0_min_hz=fmin,
        f0_max_hz=fmax,
        frame_count=frame_count,
        voiced_frame_count=voiced_count,
        voiced_ratio=voiced_ratio,
        mean_voiced_probability=mean_prob,
        f0_voiced=stats,
        f0_median_hz=f0_median,
        f0_curve_times=f0_curve_t,
        f0_curve_values=f0_curve_v,
        voiced_prob_curve_times=p_t,
        voiced_prob_curve_values=p_v,
        times_seconds=times,
        f0_hz=f0,
        voiced_flag=usable,
        voiced_probability=voiced_prob,
        confidence=confidence,
        warnings=warnings,
    )


def _downsample_f0(
    times: NDArray[np.float64],
    f0: NDArray[np.float64],
    usable: NDArray[np.bool_],
    max_points: int,
) -> tuple[list[float], list[float | None]]:
    n = times.size
    if n == 0:
        return [], []
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points).astype(int)
    out_t = [float(times[i]) for i in idx]
    out_v: list[float | None] = []
    for i in idx:
        if usable[i] and np.isfinite(f0[i]) and f0[i] > 0:
            out_v.append(float(f0[i]))
        else:
            out_v.append(None)
    return out_t, out_v
