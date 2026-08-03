"""Onset strength, tempo, and beat estimates.

Tempo is returned only when onset evidence is sufficient; otherwise ``tempo_bpm``
is null (never an invented beat grid). Half-time / double-time ambiguity is
inherent to autocorrelation-based tempo estimators and is always warned when a
tempo is reported.

Tempo reliability gate (evidence-based)
---------------------------------------
A candidate tempo from ``librosa.beat.beat_track`` is accepted only if **all**
of the following hold:

1. ``duration >= min_duration_for_tempo_seconds``
2. ``onset_event_count >= min_onset_events_for_tempo``
3. ``beat_count >= min_beats_for_tempo`` (enough placed beats for interval stats)
4. If ≥2 beat intervals exist:
   ``CV(intervals) = std/mean <= max_beat_interval_cv``
5. **Periodicity:** normalized onset-envelope autocorrelation at the lag of one
   beat period (60/tempo seconds) satisfies
   ``R(τ) >= min_tempo_periodicity``

``R(τ)`` is the standard lag autocorrelation of the zero-mean onset-strength
envelope. Periodic click tracks yield high ``R(τ)`` (typically ≳ 0.6);
broadband noise may still receive a plausible BPM and a near-regular forced
beat grid from dynamic programming, but ``R(τ)`` remains low (typically ≲ 0.2).
That separation — not a fixture-specific BPM blacklist — is the reliability
criterion.

When any gate fails: ``tempo_bpm=null``, ``beat_times_s=[]``, low confidence,
and an explicit warning (e.g. ``tempo_suppressed_insufficient_periodicity``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from neuroacoustic.analysis.stats import DistributionStats, downsample_series
from neuroacoustic.config import AnalysisConfig
from neuroacoustic.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RhythmFeatures:
    """Rhythm / beat analysis summary."""

    onset_strength: np.ndarray
    onset_times: np.ndarray
    onset_strength_mean: float | None
    onset_strength_std: float | None
    onset_event_count: int
    tempo_bpm: float | None
    beat_times_s: list[float]
    beat_count: int
    beat_interval_cv: float | None
    tempo_periodicity: float | None
    confidence: float | None
    warnings: list[str] = field(default_factory=list)
    onset_strength_distribution: DistributionStats = field(
        default_factory=DistributionStats
    )
    onset_curve_times: list[float] = field(default_factory=list)
    onset_curve_values: list[float] = field(default_factory=list)
    note: str = (
        "Tempo estimates may be off by a factor of 2 (half-/double-time). "
        "Null tempo means insufficient rhythmic evidence. "
        "Acceptance requires onset-envelope autocorrelation at the beat period "
        "plus stable beat intervals (see module docstring)."
    )


def _onset_autocorr_at_tempo(
    onset_env: np.ndarray,
    *,
    tempo_bpm: float,
    sample_rate: int,
    hop_length: int,
) -> float | None:
    """Normalized autocorrelation of onset strength at one beat-period lag."""
    if tempo_bpm <= 0 or onset_env.size < 4:
        return None
    period_s = 60.0 / float(tempo_bpm)
    lag = int(round(period_s * float(sample_rate) / float(hop_length)))
    if lag < 1 or lag >= onset_env.size:
        return None
    x = onset_env.astype(np.float64) - float(np.mean(onset_env))
    denom = float(np.dot(x, x))
    if denom <= 1e-20:
        return None
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def _beat_interval_cv(beat_times: np.ndarray) -> float | None:
    if beat_times.size < 3:
        return None
    intervals = np.diff(beat_times)
    mean_iv = float(np.mean(intervals))
    if mean_iv <= 1e-12:
        return None
    return float(np.std(intervals) / mean_iv)


def compute_rhythm_features(
    y: np.ndarray,
    sample_rate: int,
    analysis: AnalysisConfig,
) -> RhythmFeatures:
    """Estimate onset strength and optional tempo/beats from mono ``y``."""
    warnings: list[str] = []
    hop = int(analysis.hop_length)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    duration = float(y.size) / float(sample_rate) if sample_rate > 0 else 0.0
    silent_floor = max(float(analysis.amplitude_floor) * 1e3, 1e-8)

    def _empty(**kwargs: object) -> RhythmFeatures:
        empty = np.array([], dtype=np.float64)
        base = dict(
            onset_strength=empty,
            onset_times=empty,
            onset_strength_mean=None,
            onset_strength_std=None,
            onset_event_count=0,
            tempo_bpm=None,
            beat_times_s=[],
            beat_count=0,
            beat_interval_cv=None,
            tempo_periodicity=None,
            confidence=0.0,
            warnings=warnings,
        )
        base.update(kwargs)
        return RhythmFeatures(**base)  # type: ignore[arg-type]

    if y.size < hop or float(np.max(np.abs(y))) < silent_floor:
        warnings.append("insufficient_signal_for_rhythm")
        return _empty()

    onset_env = librosa.onset.onset_strength(
        y=y, sr=sample_rate, hop_length=hop, aggregate=np.median
    )
    onset_env = np.asarray(onset_env, dtype=np.float64)
    times = librosa.frames_to_time(
        np.arange(onset_env.size), sr=sample_rate, hop_length=hop
    )
    times = np.asarray(times, dtype=np.float64)

    mean_os = float(np.mean(onset_env)) if onset_env.size else None
    std_os = float(np.std(onset_env)) if onset_env.size else None
    dist = DistributionStats.from_values(onset_env)
    curve_t, curve_v = downsample_series(times, onset_env, analysis.max_timeseries_points)

    try:
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop,
            units="frames",
        )
        onset_event_count = int(np.asarray(onset_frames).size)
    except Exception:  # noqa: BLE001
        onset_event_count = 0
        warnings.append("onset_detect_failed")

    tempo_bpm: float | None = None
    beat_times: list[float] = []
    beat_count = 0
    beat_interval_cv: float | None = None
    tempo_periodicity: float | None = None
    conf = 0.0

    rhythmic_contrast = 0.0
    if mean_os is not None and mean_os > 1e-12 and std_os is not None:
        rhythmic_contrast = float(std_os / mean_os)

    if duration < float(analysis.min_duration_for_tempo_seconds):
        warnings.append("too_short_for_tempo")
    elif onset_event_count < int(analysis.min_onset_events_for_tempo):
        warnings.append("too_few_onsets_for_tempo")
    elif rhythmic_contrast < 0.05:
        # Near-flat onset envelope (true drones); noise has high contrast.
        warnings.append("low_onset_contrast_drone_or_noise")
    else:
        try:
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sample_rate,
                hop_length=hop,
                start_bpm=120.0,
                units="time",
            )
            tempo_val = float(np.atleast_1d(tempo)[0])
            lo = float(analysis.min_tempo_bpm)
            hi = float(analysis.max_tempo_bpm)
            beat_arr = np.asarray(beats, dtype=np.float64).reshape(-1)
            beat_count = int(beat_arr.size)
            beat_interval_cv = _beat_interval_cv(beat_arr)
            tempo_periodicity = _onset_autocorr_at_tempo(
                onset_env,
                tempo_bpm=tempo_val,
                sample_rate=sample_rate,
                hop_length=hop,
            )

            if tempo_val < lo or tempo_val > hi or tempo_val != tempo_val:
                warnings.append("tempo_out_of_range")
                conf = 0.1
            elif beat_count < int(analysis.min_beats_for_tempo):
                warnings.append("tempo_suppressed_too_few_beats")
                conf = 0.15
            elif (
                beat_interval_cv is not None
                and beat_interval_cv > float(analysis.max_beat_interval_cv)
            ):
                warnings.append("tempo_suppressed_unstable_beat_intervals")
                conf = 0.2
            elif tempo_periodicity is None:
                warnings.append("tempo_suppressed_periodicity_undefined")
                conf = 0.15
            elif tempo_periodicity < float(analysis.min_tempo_periodicity):
                # Primary anti-false-positive gate for noise / aperiodic textures.
                warnings.append("tempo_suppressed_insufficient_periodicity")
                conf = float(np.clip(0.1 + 0.4 * max(0.0, tempo_periodicity), 0.0, 0.35))
            else:
                # Accepted tempo
                tempo_bpm = tempo_val
                beat_times = [float(t) for t in beat_arr.tolist()]
                expected_beats = duration * (tempo_val / 60.0)
                if expected_beats > 0 and beat_arr.size > 0:
                    density_ratio = min(beat_arr.size, expected_beats) / max(
                        beat_arr.size, expected_beats
                    )
                else:
                    density_ratio = 0.0
                stability = 1.0
                if beat_interval_cv is not None:
                    stability = float(
                        np.clip(
                            1.0 - beat_interval_cv / max(float(analysis.max_beat_interval_cv), 1e-6),
                            0.0,
                            1.0,
                        )
                    )
                periodicity_score = float(
                    np.clip(
                        (tempo_periodicity - float(analysis.min_tempo_periodicity))
                        / max(1.0 - float(analysis.min_tempo_periodicity), 1e-6),
                        0.0,
                        1.0,
                    )
                )
                conf = float(
                    np.clip(
                        0.45 * periodicity_score
                        + 0.30 * stability
                        + 0.25 * density_ratio,
                        0.0,
                        1.0,
                    )
                )
                warnings.append("half_double_time_ambiguity")
                if conf < 0.4:
                    warnings.append("low_tempo_confidence")
        except Exception as exc:  # noqa: BLE001
            logger.warning("beat_track failed: %s", exc)
            warnings.append("beat_track_failed")
            tempo_bpm = None
            beat_times = []
            beat_count = 0
            conf = 0.0

    logger.debug(
        "rhythm tempo=%s events=%d beats=%d periodicity=%s cv=%s conf=%.3f",
        tempo_bpm,
        onset_event_count,
        beat_count,
        tempo_periodicity,
        beat_interval_cv,
        conf,
    )

    return RhythmFeatures(
        onset_strength=onset_env,
        onset_times=times,
        onset_strength_mean=mean_os,
        onset_strength_std=std_os,
        onset_event_count=onset_event_count,
        tempo_bpm=tempo_bpm,
        beat_times_s=beat_times,
        beat_count=beat_count if tempo_bpm is not None else 0,
        beat_interval_cv=beat_interval_cv if tempo_bpm is not None else None,
        tempo_periodicity=tempo_periodicity,
        confidence=float(conf),
        warnings=warnings,
        onset_strength_distribution=dist,
        onset_curve_times=curve_t,
        onset_curve_values=curve_v,
    )
