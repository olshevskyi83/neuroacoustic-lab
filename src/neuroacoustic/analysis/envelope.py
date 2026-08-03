"""Amplitude envelope and ADSR *estimates* from arbitrary recordings.

These are shape descriptors derived from a smoothed RMS envelope. They are
**not** a synthesizer ADSR program recovered from the file. Mixed music,
polyphony, and compression routinely violate a single-ADSR model.

ADSR estimation procedure (documented)
--------------------------------------
1. Frame RMS envelope at ``hop_length``, then smooth with a short moving average.
2. ``onset_time_s``: first time the envelope reaches ``onset_ratio * peak``.
3. ``attack_time_s``: time from onset to ``attack_high_ratio * peak`` (fallback: peak).
4. ``sustain_level``: median envelope in the mid-file window
   ``[sustain_start, sustain_end]``, divided by peak (relative, ~0–1).
5. ``decay_time_s``: time from peak to first post-peak frame at/below the
   sustain absolute level (0 if the level never falls — sustained tone).
6. ``release_time_s``: time for the late envelope to fall from a mid-sustain
   (or 50% peak for percussive shapes) to ``release_ratio * peak``.

Confidence and warnings flag silent, short, percussive-like, and sustained-like
shapes. Fields may be null when undefined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neuroacoustic.analysis.stats import DistributionStats, downsample_series
from neuroacoustic.config import AnalysisConfig
from neuroacoustic.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class EnvelopeFeatures:
    """Envelope / ADSR-like estimates (heuristic)."""

    rms_envelope: np.ndarray
    times: np.ndarray
    peak_amplitude: float
    onset_time_s: float | None
    attack_time_s: float | None
    decay_time_s: float | None
    sustain_level: float | None
    release_time_s: float | None
    confidence: float | None
    warnings: list[str] = field(default_factory=list)
    envelope_distribution: DistributionStats = field(
        default_factory=DistributionStats
    )
    envelope_curve_times: list[float] = field(default_factory=list)
    envelope_curve_values: list[float] = field(default_factory=list)
    note: str = (
        "ADSR fields are envelope-shape estimates, not a recovered synthesizer "
        "ADSR program."
    )


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.astype(np.float64, copy=True)
    w = int(window)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=np.float64) / w
    pad = w // 2
    padded = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: x.size]


def _first_crossing(env: np.ndarray, level: float, start: int = 0) -> int | None:
    for i in range(max(0, start), env.size):
        if env[i] >= level:
            return i
    return None


def compute_envelope_features(
    y: np.ndarray,
    sample_rate: int,
    analysis: AnalysisConfig,
) -> EnvelopeFeatures:
    """Estimate envelope shape and ADSR-like times from mono waveform ``y``."""
    warnings: list[str] = []
    hop = int(analysis.hop_length)
    floor = float(analysis.amplitude_floor)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    empty = EnvelopeFeatures(
        rms_envelope=np.array([], dtype=np.float64),
        times=np.array([], dtype=np.float64),
        peak_amplitude=0.0,
        onset_time_s=None,
        attack_time_s=None,
        decay_time_s=None,
        sustain_level=None,
        release_time_s=None,
        confidence=0.0,
        warnings=["empty_signal"],
    )

    if y.size == 0:
        return empty

    if y.size < hop:
        env = np.array([float(np.sqrt(np.mean(y * y)))], dtype=np.float64)
        times = np.array([0.0], dtype=np.float64)
    else:
        n_frames = 1 + max(0, (y.size - hop) // hop)
        env = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            frame = y[start : start + hop]
            env[i] = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
        times = (np.arange(n_frames, dtype=np.float64) * hop) / float(sample_rate)

    env = _moving_average(env, int(analysis.envelope_smooth_frames))
    peak = float(np.max(env)) if env.size else 0.0
    dist = DistributionStats.from_values(env)
    curve_t, curve_v = downsample_series(times, env, analysis.max_timeseries_points)

    if peak < max(floor * 1e3, 1e-8):
        warnings.append("near_silent_envelope")
        return EnvelopeFeatures(
            rms_envelope=env,
            times=times,
            peak_amplitude=peak,
            onset_time_s=None,
            attack_time_s=None,
            decay_time_s=None,
            sustain_level=None,
            release_time_s=None,
            confidence=0.0,
            warnings=warnings,
            envelope_distribution=dist,
            envelope_curve_times=curve_t,
            envelope_curve_values=curve_v,
        )

    if times.size > 0 and times[-1] < 0.05:
        warnings.append("very_short_signal")

    onset_level = peak * float(analysis.envelope_onset_ratio)
    attack_high = peak * float(analysis.envelope_attack_high_ratio)
    release_level = peak * float(analysis.envelope_release_ratio)

    onset_idx = _first_crossing(env, onset_level, 0)
    onset_time = float(times[onset_idx]) if onset_idx is not None else None

    peak_idx = int(np.argmax(env))
    peak_time = float(times[peak_idx])

    attack_time: float | None = None
    if onset_idx is not None:
        high_idx = _first_crossing(env, attack_high, onset_idx)
        if high_idx is not None and high_idx >= onset_idx:
            attack_time = float(times[high_idx] - times[onset_idx])
        else:
            attack_time = float(max(0.0, peak_time - times[onset_idx]))
            warnings.append("attack_peak_fallback")

    i0 = int(env.size * float(analysis.envelope_sustain_start))
    i1 = max(i0 + 1, int(env.size * float(analysis.envelope_sustain_end)))
    i0 = max(0, min(i0, env.size - 1))
    i1 = max(i0 + 1, min(i1, env.size))
    sustain_abs = float(np.median(env[i0:i1])) if i1 > i0 else float(env[peak_idx])
    sustain_level = float(np.clip(sustain_abs / peak, 0.0, 1.0))

    decay_time: float | None = None
    if peak_idx < env.size - 1:
        target = max(sustain_abs, peak * 0.15)
        for i in range(peak_idx + 1, env.size):
            if env[i] <= target:
                decay_time = float(times[i] - peak_time)
                break
        if decay_time is None:
            decay_time = 0.0
            warnings.append("no_post_peak_decay_to_sustain")
    else:
        # Peak at/near end — no post-peak region (typical for sustained tones)
        decay_time = 0.0
        warnings.append("no_post_peak_decay_to_sustain")

    release_time: float | None = None
    mid_sustain = max(sustain_abs, peak * 0.2)
    last_high = None
    for i in range(env.size - 1, max(peak_idx, 0) - 1, -1):
        if env[i] >= mid_sustain:
            last_high = i
            break
    if last_high is not None and last_high < env.size - 1:
        low_idx = None
        for i in range(last_high, env.size):
            if env[i] <= release_level:
                low_idx = i
                break
        if low_idx is not None:
            release_time = float(times[low_idx] - times[last_high])
        else:
            release_time = float(times[-1] - times[last_high])
            warnings.append("release_did_not_reach_floor")
    else:
        half = peak * 0.5
        half_idx = None
        for i in range(peak_idx, env.size):
            if env[i] <= half:
                half_idx = i
                break
        if half_idx is not None:
            low_idx = None
            for i in range(half_idx, env.size):
                if env[i] <= release_level:
                    low_idx = i
                    break
            if low_idx is not None:
                release_time = float(times[low_idx] - times[half_idx])
            else:
                release_time = float(times[-1] - times[half_idx])
        else:
            release_time = None
            warnings.append("release_unresolved")

    conf = 0.55
    if "very_short_signal" in warnings:
        conf *= 0.4
    if sustain_level is not None and sustain_level < 0.25 and attack_time is not None and attack_time < 0.05:
        conf = min(1.0, conf + 0.25)
        warnings.append("percussive_like_envelope")
    if sustain_level is not None and sustain_level > 0.7:
        conf = min(1.0, conf + 0.15)
        warnings.append("sustained_like_envelope")
    if attack_time is not None and attack_time > 0.3:
        warnings.append("slow_attack_estimate")
    warnings.append("not_synthesizer_adsr")
    conf = float(np.clip(conf, 0.0, 1.0))

    logger.debug(
        "envelope onset=%s attack=%s decay=%s sustain=%s release=%s conf=%.3f",
        onset_time,
        attack_time,
        decay_time,
        sustain_level,
        release_time,
        conf,
    )

    return EnvelopeFeatures(
        rms_envelope=env,
        times=times,
        peak_amplitude=peak,
        onset_time_s=onset_time,
        attack_time_s=attack_time,
        decay_time_s=decay_time,
        sustain_level=sustain_level,
        release_time_s=release_time,
        confidence=conf,
        warnings=warnings,
        envelope_distribution=dist,
        envelope_curve_times=curve_t,
        envelope_curve_values=curve_v,
    )
