"""Preliminary comparison-vector construction.

Features with incompatible physical units are NOT mixed into a final similarity
embedding. This module builds an explicitly versioned **preliminary** vector with
heuristic per-feature scales. Dataset-level normalization is deferred.

Vector ordering (``vector_version`` = ``0.4.0-preliminary``)
------------------------------------------------------------
Indices 0–11 preserve the Milestone 2 spectral/energy prefix and scaling:

0  peak_freq_norm                 peak FFT freq / Nyquist → [0,1]
1  centroid_mean_norm             mean centroid / Nyquist → [0,1]
2  bandwidth_mean_norm            mean bandwidth / Nyquist → [0,1]
3  rolloff_mean_norm              mean rolloff / Nyquist → [0,1]
4  flatness_mean                  mean spectral flatness (already ~0–1)
5  entropy_mean                   mean spectral entropy (already ~0–1)
6  contrast_mean_norm             mean contrast_db / 100 → [0,1]
7  rms                            linear RMS (unscaled amplitude)
8  peak_amplitude                 linear peak (unscaled)
9  crest_factor_norm              crest / 50 → [0,1]
10 zcr                            mean zero-crossing rate (per sample)
11 estimated_dynamic_range_db_norm  DR_est_db / 120 → [0,1]

Milestone 3 appends (null pitch/harmonics → 0.0, never a fake 0 Hz feature):

12 voiced_ratio                   voiced frames / frames (0–1)
13 f0_median_norm                 median voiced F0 / f0_max_hz → [0,1]
14 f0_std_norm                    voiced F0 std / f0_max_hz → [0,1]
15 mean_voiced_probability        mean pYIN voiced prob on usable frames
16 pitch_confidence               heuristic pitch confidence 0–1
17 harmonic_count_norm            mean harmonic count / max_harmonics → [0,1]
18 harmonic_density               slot occupancy 0–1
19 harmonic_energy_fraction_estimate  linear E_harm/E_band in [0,1]
20 inharmonicity_estimate_norm    min(inharmonicity_est, 1.0)
21 harmonics_confidence           heuristic harmonics confidence 0–1

Milestone 4A appends (null → 0.0; mono width/corr → 0.0; null tempo → 0.0):

22 attack_time_norm               attack_s / 2.0 → [0,1]
23 decay_time_norm                decay_s / 2.0 → [0,1]
24 sustain_level                  relative sustain (already ~0–1)
25 release_time_norm              release_s / 2.0 → [0,1]
26 envelope_confidence            0–1
27 stereo_correlation_norm        (corr+1)/2 → [0,1]; mono/undefined → 0.0
28 stereo_width_estimate          0–1; mono/undefined → 0.0
29 side_to_mid_ratio_norm         min(ratio, 5)/5 → [0,1]; undefined → 0.0
30 is_stereo                      1.0 if stereo else 0.0
31 tempo_bpm_norm                 tempo/200 → [0,1]; null tempo → 0.0
32 onset_strength_mean_norm       min(mean, 10)/10 → [0,1]
33 rhythm_confidence              0–1
34 tail_decay_t60_norm            min(t60, 10)/10 → [0,1]; null → 0.0
35 reverb_confidence              0–1
"""

from __future__ import annotations

from neuroacoustic.analysis.energy import EnergyFeatures
from neuroacoustic.analysis.envelope import EnvelopeFeatures
from neuroacoustic.analysis.harmonics import HarmonicFeatures
from neuroacoustic.analysis.pitch import PitchFeatures
from neuroacoustic.analysis.reverb import ReverbFeatures
from neuroacoustic.analysis.rhythm import RhythmFeatures
from neuroacoustic.analysis.spectral import SpectralFeatures
from neuroacoustic.analysis.spectrum import FftSummaryData
from neuroacoustic.analysis.stereo import StereoFeatures
from neuroacoustic.config import AnalysisConfig
from neuroacoustic.fingerprint.models import PreliminaryVector


def _finite(value: float, default: float = 0.0) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        return default
    return float(value)


def _n(value: float | None, scale: float) -> float:
    if value is None or scale <= 0:
        return 0.0
    return _finite(max(0.0, min(1.0, value / scale)))


def build_preliminary_vector(
    *,
    fft: FftSummaryData,
    spectral: SpectralFeatures,
    energy: EnergyFeatures,
    pitch: PitchFeatures | None = None,
    harmonics: HarmonicFeatures | None = None,
    envelope: EnvelopeFeatures | None = None,
    stereo: StereoFeatures | None = None,
    rhythm: RhythmFeatures | None = None,
    reverb: ReverbFeatures | None = None,
    sample_rate: int,
    analysis: AnalysisConfig,
) -> PreliminaryVector:
    """Assemble preliminary vector: M2+M3 prefix + optional M4A append."""
    nyquist = max(sample_rate / 2.0, 1.0)
    labels = [
        "peak_freq_norm",
        "centroid_mean_norm",
        "bandwidth_mean_norm",
        "rolloff_mean_norm",
        "flatness_mean",
        "entropy_mean",
        "contrast_mean_norm",
        "rms",
        "peak_amplitude",
        "crest_factor_norm",
        "zcr",
        "estimated_dynamic_range_db_norm",
        "voiced_ratio",
        "f0_median_norm",
        "f0_std_norm",
        "mean_voiced_probability",
        "pitch_confidence",
        "harmonic_count_norm",
        "harmonic_density",
        "harmonic_energy_fraction_estimate",
        "inharmonicity_estimate_norm",
        "harmonics_confidence",
        "attack_time_norm",
        "decay_time_norm",
        "sustain_level",
        "release_time_norm",
        "envelope_confidence",
        "stereo_correlation_norm",
        "stereo_width_estimate",
        "side_to_mid_ratio_norm",
        "is_stereo",
        "tempo_bpm_norm",
        "onset_strength_mean_norm",
        "rhythm_confidence",
        "tail_decay_t60_norm",
        "reverb_confidence",
    ]

    peak_f = fft.peak_frequency_hz
    values = [
        _n(peak_f, nyquist),
        _n(spectral.centroid_hz.mean, nyquist),
        _n(spectral.bandwidth_hz.mean, nyquist),
        _n(spectral.rolloff_hz.mean, nyquist),
        _finite(float(spectral.flatness.mean or 0.0)),
        _finite(float(spectral.entropy.mean or 0.0)),
        _n(spectral.contrast_db.mean, 100.0),
        _finite(float(energy.rms or 0.0)),
        _finite(float(energy.peak_amplitude or 0.0)),
        _n(energy.crest_factor, 50.0),
        _finite(float(energy.zero_crossing_rate or 0.0)),
        _n(energy.estimated_dynamic_range_db, 120.0),
    ]

    f0_max = float(analysis.f0_max_hz)
    if pitch is None:
        values.extend([0.0] * 5)
    else:
        values.extend(
            [
                _finite(float(pitch.voiced_ratio or 0.0)),
                _n(pitch.f0_median_hz, f0_max),
                _n(pitch.f0_voiced.std, f0_max),
                _finite(float(pitch.mean_voiced_probability or 0.0)),
                _finite(float(pitch.confidence or 0.0)),
            ]
        )

    if harmonics is None:
        values.extend([0.0] * 5)
    else:
        values.extend(
            [
                _n(harmonics.harmonic_count, float(analysis.max_harmonics)),
                _finite(float(harmonics.harmonic_density or 0.0)),
                _finite(float(harmonics.harmonic_energy_fraction_estimate or 0.0)),
                _finite(min(float(harmonics.inharmonicity_estimate or 0.0), 1.0)),
                _finite(float(harmonics.confidence or 0.0)),
            ]
        )

    # Milestone 4A append (indices 22–35)
    if envelope is None:
        values.extend([0.0] * 5)
    else:
        values.extend(
            [
                _n(envelope.attack_time_s, 2.0),
                _n(envelope.decay_time_s, 2.0),
                _finite(float(envelope.sustain_level or 0.0)),
                _n(envelope.release_time_s, 2.0),
                _finite(float(envelope.confidence or 0.0)),
            ]
        )

    if stereo is None or stereo.is_mono:
        values.extend([0.0, 0.0, 0.0, 0.0])
    else:
        corr_norm = 0.0
        if stereo.correlation is not None:
            corr_norm = _finite(0.5 * (float(stereo.correlation) + 1.0))
        values.extend(
            [
                corr_norm,
                _finite(float(stereo.stereo_width_estimate or 0.0)),
                _n(stereo.side_to_mid_ratio, 5.0),
                1.0,
            ]
        )

    if rhythm is None:
        values.extend([0.0, 0.0, 0.0])
    else:
        values.extend(
            [
                _n(rhythm.tempo_bpm, 200.0),
                _n(rhythm.onset_strength_mean, 10.0),
                _finite(float(rhythm.confidence or 0.0)),
            ]
        )

    if reverb is None:
        values.extend([0.0, 0.0])
    else:
        values.extend(
            [
                _n(reverb.tail_decay_t60_estimate_seconds, 10.0),
                _finite(float(reverb.confidence or 0.0)),
            ]
        )

    clean = [_finite(v) for v in values]
    assert len(clean) == len(labels)
    return PreliminaryVector(
        version=analysis.vector_version,
        values=clean,
        labels=labels,
    )
