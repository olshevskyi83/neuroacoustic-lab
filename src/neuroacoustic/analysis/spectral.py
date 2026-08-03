"""Frame-level spectral descriptors (Milestone 2).

All features are computed from the magnitude STFT ``S[f, t] = |X[f, t]|``.
Frame-wise series are aggregated with :class:`DistributionStats`.

Formulas (per frame ``t``)
--------------------------
Spectral centroid (Hz)
    ``C_t = sum_f f · S[f,t] / sum_f S[f,t]``

Spectral bandwidth (Hz)
    ``B_t = sqrt( sum_f (f - C_t)^2 · S[f,t] / sum_f S[f,t] )``

Spectral rolloff (Hz)
    Smallest ``f_r`` such that ``sum_{f<=f_r} S[f,t]^2 >= ρ · sum_f S[f,t]^2``
    with configurable ``ρ`` (default 0.85).

Spectral flatness (dimensionless, ~0–1)
    ``F_t = exp(mean_f log(S[f,t] + ε)) / (mean_f (S[f,t] + ε))``
    Pure tones → near 0; noise-like spectra → nearer 1.

Spectral entropy (dimensionless, normalized ~0–1)
    Let ``p_f = S[f,t]^2 / sum S^2``. Then
    ``H_t = -sum_f p_f log(p_f + ε) / log(N_bins)``.

Spectral contrast (dB)
    Per-band peak-to-valley differences via ``librosa.feature.spectral_contrast``.
    Stored as the mean contrast across bands per frame, plus band-mean stats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroacoustic.analysis.spectrum import StftResult
from neuroacoustic.analysis.stats import DistributionStats, downsample_series, sanitize_array
from neuroacoustic.config import AnalysisConfig


@dataclass
class SpectralFeatures:
    centroid_hz: DistributionStats
    bandwidth_hz: DistributionStats
    rolloff_hz: DistributionStats
    flatness: DistributionStats
    entropy: DistributionStats
    contrast_db: DistributionStats
    contrast_band_means_db: list[float]
    centroid_series: tuple[list[float], list[float]]
    flatness_series: tuple[list[float], list[float]]
    mean_spectrum_hz: list[float]
    mean_spectrum_magnitude: list[float]
    stft_n_frames: int
    stft_n_freq_bins: int
    warnings: list[str]


def analyze_spectral(
    stft: StftResult,
    analysis: AnalysisConfig,
) -> SpectralFeatures:
    """Compute spectral descriptors from an in-memory STFT."""
    import librosa

    warnings: list[str] = []
    mag = sanitize_array(stft.magnitude)
    n_freq, n_frames = mag.shape if mag.ndim == 2 else (0, 0)
    if n_frames == 0 or n_freq == 0 or not np.any(mag > 0):
        warnings.append("Spectral analysis skipped or empty (silent / zero magnitude STFT)")
        empty = DistributionStats.from_values([])
        return SpectralFeatures(
            centroid_hz=empty,
            bandwidth_hz=empty,
            rolloff_hz=empty,
            flatness=empty,
            entropy=empty,
            contrast_db=empty,
            contrast_band_means_db=[],
            centroid_series=([], []),
            flatness_series=([], []),
            mean_spectrum_hz=[],
            mean_spectrum_magnitude=[],
            stft_n_frames=n_frames,
            stft_n_freq_bins=n_freq,
            warnings=warnings,
        )

    sr = stft.sample_rate
    hop = stft.hop_length
    n_fft = stft.n_fft
    times = stft.times_seconds

    centroid = librosa.feature.spectral_centroid(
        S=mag, sr=sr, n_fft=n_fft, hop_length=hop
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        S=mag, sr=sr, n_fft=n_fft, hop_length=hop
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=mag,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        roll_percent=analysis.rolloff_percentile,
    )[0]
    flatness = librosa.feature.spectral_flatness(S=mag)[0]
    entropy = _spectral_entropy(mag, analysis.amplitude_floor)

    contrast = librosa.feature.spectral_contrast(
        S=mag,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_bands=analysis.n_contrast_bands,
    )
    # contrast shape: (n_bands+1, n_frames); mean across bands per frame
    contrast_mean = np.mean(contrast, axis=0)
    contrast_band_means = [float(x) for x in np.mean(contrast, axis=1)]

    mean_mag = np.mean(mag, axis=1)
    # Cap mean-spectrum bins stored in JSON (reuse FFT summary bin count).
    if mean_mag.size > analysis.fft_summary_bins:
        idx = np.linspace(0, mean_mag.size - 1, analysis.fft_summary_bins).astype(int)
        mean_f = [float(stft.frequencies_hz[i]) for i in idx]
        mean_m = [float(mean_mag[i]) for i in idx]
    else:
        mean_f = [float(x) for x in stft.frequencies_hz]
        mean_m = [float(x) for x in mean_mag]

    max_pts = analysis.max_timeseries_points
    return SpectralFeatures(
        centroid_hz=DistributionStats.from_values(centroid),
        bandwidth_hz=DistributionStats.from_values(bandwidth),
        rolloff_hz=DistributionStats.from_values(rolloff),
        flatness=DistributionStats.from_values(flatness),
        entropy=DistributionStats.from_values(entropy),
        contrast_db=DistributionStats.from_values(contrast_mean),
        contrast_band_means_db=contrast_band_means,
        centroid_series=downsample_series(times, centroid, max_pts),
        flatness_series=downsample_series(times, flatness, max_pts),
        mean_spectrum_hz=mean_f,
        mean_spectrum_magnitude=mean_m,
        stft_n_frames=n_frames,
        stft_n_freq_bins=n_freq,
        warnings=warnings,
    )


def _spectral_entropy(mag: np.ndarray, floor: float) -> np.ndarray:
    """Normalized Shannon entropy of the power spectrum per frame."""
    power = np.maximum(mag.astype(np.float64) ** 2, floor)
    total = np.sum(power, axis=0, keepdims=True)
    total = np.maximum(total, floor)
    p = power / total
    log_n = np.log(max(mag.shape[0], 2))
    entropy = -np.sum(p * np.log(p + floor), axis=0) / log_n
    return sanitize_array(entropy)
