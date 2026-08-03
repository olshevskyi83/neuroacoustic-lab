"""Deterministic analysis-configuration hashing for cache identity.

Cache identity
--------------
One completed analysis is uniquely identified by::

    content_hash (SHA-256 of file bytes)
  + analysis_version
  + config_hash

``config_hash`` is SHA-256 of a **canonical JSON** object of scientific
parameters that affect numeric fingerprint outputs. Path / logging / UI
options are intentionally excluded.

Included in the hash
--------------------
- ``analysis_sample_rate`` (native preserved as JSON ``null`` when unset)
- All ``AnalysisConfig`` fields that drive STFT, pitch, harmonics, envelope,
  stereo, rhythm, and decay (n_fft, hop_length, win_length, floors,
  thresholds, tempo/decay gates, ``vector_version``, …)
- Audio quality thresholds that appear in the fingerprint quality section:
  ``silence_peak_threshold``, ``silence_sample_threshold``,
  ``clipping_threshold``

Excluded from the hash
----------------------
- ``paths.*`` (input/output/database directories)
- ``logging.*``
- ``project.name``
- ``config_path``
- ``probe.prefer_ffprobe``
- CLI flags such as ``--plots``, ``--force``, ``--json``, ``--output-dir``
- ``schema_version`` / ``analysis_version`` (version is a separate identity
  component; schema version is recorded on the row but does not enter the
  scientific config hash)

Serialization: ``json.dumps(..., sort_keys=True, separators=(",", ":"),
ensure_ascii=True)`` over the payload, then SHA-256 hex digest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from neuroacoustic.config import AppConfig


def scientific_config_payload(
    config: AppConfig,
    *,
    analysis_sample_rate: int | None = None,
) -> dict[str, Any]:
    """Return the canonical dict used for ``config_hash`` (not yet hashed)."""
    a = config.analysis
    audio = config.audio
    rate = (
        analysis_sample_rate
        if analysis_sample_rate is not None
        else audio.analysis_sample_rate
    )
    return {
        "analysis_sample_rate": rate,
        "silence_peak_threshold": audio.silence_peak_threshold,
        "silence_sample_threshold": audio.silence_sample_threshold,
        "clipping_threshold": audio.clipping_threshold,
        "n_fft": a.n_fft,
        "hop_length": a.hop_length,
        "win_length": a.effective_win_length(),
        "rolloff_percentile": a.rolloff_percentile,
        "n_contrast_bands": a.n_contrast_bands,
        "fft_peak_count": a.fft_peak_count,
        "max_timeseries_points": a.max_timeseries_points,
        "fft_summary_bins": a.fft_summary_bins,
        "amplitude_floor": a.amplitude_floor,
        "f0_min_hz": a.f0_min_hz,
        "f0_max_hz": a.f0_max_hz,
        "f0_frame_length": a.f0_frame_length,
        "voiced_prob_threshold": a.voiced_prob_threshold,
        "voiced_prob_floor": a.voiced_prob_floor,
        "min_voiced_frames": a.min_voiced_frames,
        "max_harmonics": a.max_harmonics,
        "harmonic_search_width_fraction": a.harmonic_search_width_fraction,
        "harmonic_rel_amp_threshold": a.harmonic_rel_amp_threshold,
        "envelope_smooth_frames": a.envelope_smooth_frames,
        "envelope_onset_ratio": a.envelope_onset_ratio,
        "envelope_attack_high_ratio": a.envelope_attack_high_ratio,
        "envelope_sustain_start": a.envelope_sustain_start,
        "envelope_sustain_end": a.envelope_sustain_end,
        "envelope_release_ratio": a.envelope_release_ratio,
        "min_tempo_bpm": a.min_tempo_bpm,
        "max_tempo_bpm": a.max_tempo_bpm,
        "min_onset_events_for_tempo": a.min_onset_events_for_tempo,
        "min_duration_for_tempo_seconds": a.min_duration_for_tempo_seconds,
        "min_beats_for_tempo": a.min_beats_for_tempo,
        "max_beat_interval_cv": a.max_beat_interval_cv,
        "min_tempo_periodicity": a.min_tempo_periodicity,
        "decay_fit_db_range": a.decay_fit_db_range,
        "decay_min_r2": a.decay_min_r2,
        "decay_min_duration_seconds": a.decay_min_duration_seconds,
        "vector_version": a.vector_version,
    }


def compute_config_hash(
    config: AppConfig,
    *,
    analysis_sample_rate: int | None = None,
) -> str:
    """SHA-256 hex digest of the canonical scientific configuration payload."""
    payload = scientific_config_payload(
        config, analysis_sample_rate=analysis_sample_rate
    )
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
