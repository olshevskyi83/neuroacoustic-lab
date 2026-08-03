"""File-tail decay / reverberation *heuristics*.

This module estimates exponential decay from the energy envelope of a recording.
Results are named to avoid claiming exact room RT60 on arbitrary mixed music.

- ``tail_decay_t60_estimate_seconds``: time for a linear-in-dB fit of the
  post-peak energy envelope to fall 60 dB, extrapolated from the fitted slope.
- Applies best to controlled impulses / exponential decays.
- Distinguishes **file-tail decay** (what is in the file) from **room RT60**.
- Returns null when the fit is poor or the measurable decay range is too small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neuroacoustic.config import AnalysisConfig
from neuroacoustic.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ReverbFeatures:
    """Decay / reverberation heuristic summary."""

    tail_decay_t60_estimate_seconds: float | None
    decay_slope_db_per_s: float | None
    fit_r_squared: float | None
    fit_db_range: float | None
    analyzed_frequency_range_hz: tuple[float, float] | None
    confidence: float | None
    warnings: list[str] = field(default_factory=list)
    note: str = (
        "tail_decay_t60_estimate_seconds is a file-tail exponential-decay "
        "heuristic, NOT calibrated room RT60 for mixed recordings."
    )


def _frame_energy_db(
    y: np.ndarray, hop: int, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size < hop:
        e = np.array([float(np.mean(y * y))], dtype=np.float64)
        return e, 10.0 * np.log10(np.maximum(e, floor))

    n_frames = 1 + max(0, (y.size - hop) // hop)
    energy = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame = y[i * hop : i * hop + hop]
        energy[i] = float(np.mean(frame * frame)) if frame.size else 0.0
    db = 10.0 * np.log10(np.maximum(energy, floor))
    return energy, db


def compute_reverb_features(
    y: np.ndarray,
    sample_rate: int,
    analysis: AnalysisConfig,
) -> ReverbFeatures:
    """Estimate file-tail T60-like decay from mono ``y``."""
    warnings: list[str] = [
        "file_tail_decay_not_room_rt60",
        "not_exact_rt60_for_mixed_music",
    ]
    hop = int(analysis.hop_length)
    floor = float(analysis.amplitude_floor)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    nyquist = float(sample_rate) / 2.0
    freq_range = (20.0, min(nyquist, 8000.0))

    if y.size < hop * 4:
        warnings.append("too_short_for_decay_fit")
        return ReverbFeatures(
            tail_decay_t60_estimate_seconds=None,
            decay_slope_db_per_s=None,
            fit_r_squared=None,
            fit_db_range=None,
            analyzed_frequency_range_hz=freq_range,
            confidence=0.0,
            warnings=warnings,
        )

    _energy, db = _frame_energy_db(y, hop, floor)
    times = (np.arange(db.size, dtype=np.float64) * hop) / float(sample_rate)
    peak_idx = int(np.argmax(db))
    peak_db = float(db[peak_idx])

    # Fit from shortly after peak while level remains within decay_fit_db_range
    start = min(peak_idx + 1, db.size - 1)
    target_drop = float(analysis.decay_fit_db_range)
    end = start
    for i in range(start, db.size):
        if peak_db - float(db[i]) >= target_drop:
            end = i
            break
        end = i

    # Require some post-peak region
    if end <= start + 2:
        warnings.append("insufficient_post_peak_frames")
        return ReverbFeatures(
            tail_decay_t60_estimate_seconds=None,
            decay_slope_db_per_s=None,
            fit_r_squared=None,
            fit_db_range=float(peak_db - float(db[end])) if end > start else None,
            analyzed_frequency_range_hz=freq_range,
            confidence=0.0,
            warnings=warnings,
        )

    t = times[start : end + 1]
    d = db[start : end + 1]
    measured_range = float(d[0] - d[-1]) if d.size else 0.0
    duration = float(t[-1] - t[0]) if t.size else 0.0

    if duration < float(analysis.decay_min_duration_seconds):
        warnings.append("decay_region_too_short")
        return ReverbFeatures(
            tail_decay_t60_estimate_seconds=None,
            decay_slope_db_per_s=None,
            fit_r_squared=None,
            fit_db_range=measured_range,
            analyzed_frequency_range_hz=freq_range,
            confidence=0.0,
            warnings=warnings,
        )

    if measured_range < 6.0:
        warnings.append("insufficient_decay_dynamic_range")
        return ReverbFeatures(
            tail_decay_t60_estimate_seconds=None,
            decay_slope_db_per_s=None,
            fit_r_squared=None,
            fit_db_range=measured_range,
            analyzed_frequency_range_hz=freq_range,
            confidence=0.15,
            warnings=warnings,
        )

    # Linear fit: db = a + b * t  (b should be negative for decay)
    A = np.column_stack([np.ones(t.size), t])
    try:
        coef, _residuals, _rank, _s = np.linalg.lstsq(A, d, rcond=None)
    except np.linalg.LinAlgError:
        warnings.append("decay_fit_failed")
        return ReverbFeatures(
            tail_decay_t60_estimate_seconds=None,
            decay_slope_db_per_s=None,
            fit_r_squared=None,
            fit_db_range=measured_range,
            analyzed_frequency_range_hz=freq_range,
            confidence=0.0,
            warnings=warnings,
        )

    intercept, slope = float(coef[0]), float(coef[1])
    pred = intercept + slope * t
    ss_res = float(np.sum((d - pred) ** 2))
    ss_tot = float(np.sum((d - np.mean(d)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > floor else 0.0
    r2 = float(np.clip(r2, 0.0, 1.0))

    if slope >= -1e-6:
        warnings.append("non_decaying_or_rising_tail")
        return ReverbFeatures(
            tail_decay_t60_estimate_seconds=None,
            decay_slope_db_per_s=slope,
            fit_r_squared=r2,
            fit_db_range=measured_range,
            analyzed_frequency_range_hz=freq_range,
            confidence=0.1,
            warnings=warnings,
        )

    # T60: time for 60 dB drop at fitted slope (energy dB uses 10*log10; for
    # amplitude RT60 people often use 20*log10 — we use energy envelope so
    # 60 dB energy ≈ 30 dB amplitude. Document clearly: slope is in energy-dB/s,
    # T60_energy = 60 / |slope|.
    t60 = float(60.0 / abs(slope))
    min_r2 = float(analysis.decay_min_r2)

    if r2 < min_r2:
        warnings.append("poor_exponential_fit")
        return ReverbFeatures(
            tail_decay_t60_estimate_seconds=None,
            decay_slope_db_per_s=slope,
            fit_r_squared=r2,
            fit_db_range=measured_range,
            analyzed_frequency_range_hz=freq_range,
            confidence=float(0.2 * r2),
            warnings=warnings,
        )

    # Confidence: fit quality × measurable range / 20 dB
    conf = float(
        np.clip(0.5 * r2 + 0.5 * min(1.0, measured_range / target_drop), 0.0, 1.0)
    )
    if measured_range < target_drop:
        warnings.append("t60_extrapolated_beyond_measured_range")
        conf *= 0.85

    logger.debug(
        "decay t60=%.3fs slope=%.3f dB/s r2=%.3f range=%.1f dB conf=%.3f",
        t60,
        slope,
        r2,
        measured_range,
        conf,
    )

    return ReverbFeatures(
        tail_decay_t60_estimate_seconds=t60,
        decay_slope_db_per_s=slope,
        fit_r_squared=r2,
        fit_db_range=measured_range,
        analyzed_frequency_range_hz=freq_range,
        confidence=conf,
        warnings=warnings,
    )
