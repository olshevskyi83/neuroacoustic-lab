"""Time-domain energy features (Milestone 2).

Formulas
--------
RMS energy (linear amplitude)
    ``RMS = sqrt(mean(x^2))`` over the full signal and per frame.

Zero-crossing rate (crossings per sample)
    Frame ZCR = ``(1 / (N-1)) * sum_n 1[sign(x[n]) != sign(x[n-1])]``
    as returned by ``librosa.feature.zero_crossing_rate``.

Peak amplitude (linear)
    ``peak = max(|x|)``.

Crest factor (dimensionless)
    ``crest = peak / max(RMS, floor)``.
    Also reported in decibels as ``20 * log10(crest)`` — this is **not** LUFS.

Estimated dynamic range (dB, heuristic)
    ``DR_est = 20 * log10( max(RMS_p95, floor) / max(RMS_p05, floor) )``
    using frame-RMS percentiles. This is an **estimate** of short-term level
    spread, not a standards-compliant loudness or True Peak measurement, and
    must not be labeled LUFS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuroacoustic.analysis.stats import DistributionStats, downsample_series, sanitize_array
from neuroacoustic.config import AnalysisConfig


@dataclass
class EnergyFeatures:
    rms: float | None
    rms_db: float | None
    peak_amplitude: float | None
    crest_factor: float | None
    crest_factor_db: float | None
    zero_crossing_rate: float | None
    estimated_dynamic_range_db: float | None
    rms_frame_stats: DistributionStats
    zcr_frame_stats: DistributionStats
    rms_series: tuple[list[float], list[float]]
    warnings: list[str]


def analyze_energy(
    mono: NDArray[np.floating],
    sample_rate: int,
    analysis: AnalysisConfig,
) -> EnergyFeatures:
    """Compute global and frame-level energy / time-domain features."""
    import librosa

    warnings: list[str] = []
    y = sanitize_array(mono)
    floor = float(analysis.amplitude_floor)
    hop = int(analysis.hop_length)
    frame = int(analysis.effective_win_length())

    if y.size == 0:
        warnings.append("Empty signal for energy analysis")
        empty = DistributionStats.from_values([])
        return EnergyFeatures(
            rms=None,
            rms_db=None,
            peak_amplitude=None,
            crest_factor=None,
            crest_factor_db=None,
            zero_crossing_rate=None,
            estimated_dynamic_range_db=None,
            rms_frame_stats=empty,
            zcr_frame_stats=empty,
            rms_series=([], []),
            warnings=warnings,
        )

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y * y)))
    if peak < floor and rms < floor:
        warnings.append("Near-silent signal: energy metrics floored")

    crest = float(peak / max(rms, floor))
    crest_db = float(20.0 * np.log10(max(crest, floor)))
    rms_db = float(20.0 * np.log10(max(rms, floor)))

    rms_frames = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    zcr_frames = librosa.feature.zero_crossing_rate(
        y, frame_length=frame, hop_length=hop
    )[0]
    times = librosa.frames_to_time(
        np.arange(rms_frames.size),
        sr=sample_rate,
        hop_length=hop,
        n_fft=frame,
    )

    rms_stats = DistributionStats.from_values(rms_frames)
    zcr_stats = DistributionStats.from_values(zcr_frames)

    # Estimated dynamic range from frame-RMS percentiles (heuristic).
    if (
        rms_stats.p95 is not None
        and rms_stats.p05 is not None
        and rms_stats.p95 > floor
        and rms_stats.p05 > floor
    ):
        dr_est = float(20.0 * np.log10(rms_stats.p95 / max(rms_stats.p05, floor)))
    elif rms_stats.maximum is not None and rms_stats.minimum is not None:
        dr_est = float(
            20.0 * np.log10(max(rms_stats.maximum, floor) / max(rms_stats.minimum, floor))
        )
        warnings.append(
            "estimated_dynamic_range_db fell back to min/max frame RMS ratio"
        )
    else:
        dr_est = None
        warnings.append("estimated_dynamic_range_db unavailable")

    global_zcr = float(np.mean(zcr_frames)) if zcr_frames.size else 0.0

    return EnergyFeatures(
        rms=rms,
        rms_db=rms_db,
        peak_amplitude=peak,
        crest_factor=crest,
        crest_factor_db=crest_db,
        zero_crossing_rate=global_zcr,
        estimated_dynamic_range_db=dr_est,
        rms_frame_stats=rms_stats,
        zcr_frame_stats=zcr_stats,
        rms_series=downsample_series(times, rms_frames, analysis.max_timeseries_points),
        warnings=warnings,
    )
