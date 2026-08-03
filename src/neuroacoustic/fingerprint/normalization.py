"""Preliminary comparison-vector construction (Milestone 2).

Features with incompatible physical units are NOT mixed into a final similarity
embedding. This module builds an explicitly versioned **preliminary** vector with
heuristic per-feature scales so early experiments can proceed. Dataset-level
normalization (z-score / robust scaling on a real corpus) is deferred.
"""

from __future__ import annotations

from neuroacoustic.analysis.energy import EnergyFeatures
from neuroacoustic.analysis.spectral import SpectralFeatures
from neuroacoustic.analysis.spectrum import FftSummaryData
from neuroacoustic.config import AnalysisConfig
from neuroacoustic.fingerprint.models import PreliminaryVector


def build_preliminary_vector(
    *,
    fft: FftSummaryData,
    spectral: SpectralFeatures,
    energy: EnergyFeatures,
    sample_rate: int,
    analysis: AnalysisConfig,
) -> PreliminaryVector:
    """Assemble a fixed-order preliminary vector with rough [0, 1]-ish scaling."""
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
    ]

    def _n(value: float | None, scale: float) -> float:
        if value is None or scale <= 0:
            return 0.0
        return float(max(0.0, min(1.0, value / scale)))

    peak_f = fft.peak_frequency_hz
    values = [
        _n(peak_f, nyquist),
        _n(spectral.centroid_hz.mean, nyquist),
        _n(spectral.bandwidth_hz.mean, nyquist),
        _n(spectral.rolloff_hz.mean, nyquist),
        float(spectral.flatness.mean or 0.0),
        float(spectral.entropy.mean or 0.0),
        _n(spectral.contrast_db.mean, 100.0),  # dB scale heuristic
        float(energy.rms or 0.0),
        float(energy.peak_amplitude or 0.0),
        _n(energy.crest_factor, 50.0),
        float(energy.zero_crossing_rate or 0.0),
        _n(energy.estimated_dynamic_range_db, 120.0),
    ]
    # Ensure JSON-safe finite floats.
    clean = []
    for v in values:
        if v != v or v in (float("inf"), float("-inf")):  # NaN / Inf
            clean.append(0.0)
        else:
            clean.append(float(v))

    return PreliminaryVector(
        version=analysis.vector_version,
        values=clean,
        labels=labels,
    )
