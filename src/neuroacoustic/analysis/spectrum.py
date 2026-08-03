"""FFT / STFT primitives for Milestone 2.

Units
-----
- Time: seconds
- Frequency: hertz (Hz)
- Magnitude: linear amplitude of the windowed DFT (not normalized to dB here)
- Power: magnitude squared

The full complex / magnitude STFT matrix is retained only in memory for feature
extraction and plotting. Fingerprint JSON stores summaries only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuroacoustic.analysis.stats import sanitize_array
from neuroacoustic.config import AnalysisConfig


@dataclass
class StftResult:
    """In-memory STFT used for analysis (not serialized whole into JSON)."""

    frequencies_hz: NDArray[np.float64]
    times_seconds: NDArray[np.float64]
    magnitude: NDArray[np.float64]  # shape (n_freq, n_frames)
    n_fft: int
    hop_length: int
    win_length: int
    sample_rate: int


@dataclass
class FftSummaryData:
    """Whole-signal FFT summary used to build the fingerprint section."""

    frequencies_hz: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    peak_frequency_hz: float | None
    peak_magnitude: float | None
    peak_frequencies_hz: list[float]
    peak_magnitudes: list[float]
    summary_frequencies_hz: list[float]
    summary_magnitudes: list[float]
    log_summary_frequencies_hz: list[float]
    log_summary_magnitudes: list[float]


def compute_stft(
    mono: NDArray[np.floating],
    sample_rate: int,
    analysis: AnalysisConfig,
) -> StftResult:
    """Compute a magnitude STFT with a Hann window.

    Algorithm
    ---------
    ``X[k, t] = sum_n x[n] w[n - t H] exp(-j 2 π k n / N)`` via ``librosa.stft``,
    then ``|X|``. Frequencies are ``k * sr / N`` (Hz). Frame times are
    ``t * hop / sr`` (seconds), centered per librosa convention.
    """
    import librosa

    n_fft = int(analysis.n_fft)
    hop = int(analysis.hop_length)
    win = int(analysis.effective_win_length())
    y = sanitize_array(mono)
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window="hann",
        center=True,
    )
    magnitude = np.abs(stft).astype(np.float64)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft).astype(np.float64)
    times = librosa.frames_to_time(
        np.arange(magnitude.shape[1]),
        sr=sample_rate,
        hop_length=hop,
        n_fft=n_fft,
    ).astype(np.float64)
    return StftResult(
        frequencies_hz=freqs,
        times_seconds=times,
        magnitude=magnitude,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        sample_rate=sample_rate,
    )


def compute_fft_summary(
    mono: NDArray[np.floating],
    sample_rate: int,
    analysis: AnalysisConfig,
) -> FftSummaryData:
    """RFFT magnitude of the full mono signal with peak and binned summaries.

    Peak picking selects the top-K magnitude bins after a simple local-maxima
    filter when length allows; otherwise the global top-K bins.
    """
    y = sanitize_array(mono)
    n = y.size
    if n == 0:
        empty = np.asarray([], dtype=np.float64)
        return FftSummaryData(
            frequencies_hz=empty,
            magnitude=empty,
            peak_frequency_hz=None,
            peak_magnitude=None,
            peak_frequencies_hz=[],
            peak_magnitudes=[],
            summary_frequencies_hz=[],
            summary_magnitudes=[],
            log_summary_frequencies_hz=[],
            log_summary_magnitudes=[],
        )

    # Use next pow2 length only for FFT efficiency; do not zero-pad beyond 2x
    # for extremely short clips to keep frequency resolution honest.
    n_fft = int(2 ** np.ceil(np.log2(max(n, analysis.n_fft))))
    n_fft = min(n_fft, max(n, analysis.n_fft) * 2)
    spec = np.fft.rfft(y, n=n_fft)
    mag = np.abs(spec).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate).astype(np.float64)

    peak_idx = int(np.argmax(mag)) if mag.size else 0
    peak_f = float(freqs[peak_idx]) if mag.size else None
    peak_m = float(mag[peak_idx]) if mag.size else None

    peak_freqs, peak_mags = _top_peaks(freqs, mag, analysis.fft_peak_count)

    lin_f, lin_m = _resample_spectrum(freqs, mag, analysis.fft_summary_bins, log=False)
    log_f, log_m = _resample_spectrum(freqs, mag, analysis.fft_summary_bins, log=True)

    return FftSummaryData(
        frequencies_hz=freqs,
        magnitude=mag,
        peak_frequency_hz=peak_f,
        peak_magnitude=peak_m,
        peak_frequencies_hz=peak_freqs,
        peak_magnitudes=peak_mags,
        summary_frequencies_hz=lin_f,
        summary_magnitudes=lin_m,
        log_summary_frequencies_hz=log_f,
        log_summary_magnitudes=log_m,
    )


def _top_peaks(
    freqs: NDArray[np.float64],
    mag: NDArray[np.float64],
    count: int,
) -> tuple[list[float], list[float]]:
    if mag.size == 0 or count <= 0:
        return [], []
    # Prefer local maxima; fall back to global argsort.
    if mag.size >= 3:
        local = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1
        if local.size == 0:
            local = np.argsort(mag)[::-1][:count]
        else:
            local = local[np.argsort(mag[local])[::-1]]
        idxs = local[:count]
    else:
        idxs = np.argsort(mag)[::-1][:count]
    return [float(freqs[i]) for i in idxs], [float(mag[i]) for i in idxs]


def _resample_spectrum(
    freqs: NDArray[np.float64],
    mag: NDArray[np.float64],
    bins: int,
    *,
    log: bool,
) -> tuple[list[float], list[float]]:
    if mag.size == 0 or bins <= 0:
        return [], []
    f_min = float(freqs[1]) if freqs.size > 1 and freqs[0] == 0 else float(freqs[0])
    f_max = float(freqs[-1])
    if f_max <= f_min:
        return [float(freqs[0])], [float(mag[0])]
    if log:
        f_min = max(f_min, 1.0)
        grid = np.geomspace(f_min, f_max, bins)
    else:
        grid = np.linspace(f_min, f_max, bins)
    sampled = np.interp(grid, freqs, mag)
    return [float(x) for x in grid], [float(x) for x in sampled]


def mean_spectrum(stft: StftResult) -> tuple[list[float], list[float]]:
    """Time-averaged magnitude spectrum (summary; not the full STFT)."""
    if stft.magnitude.size == 0:
        return [], []
    mean_mag = np.mean(stft.magnitude, axis=1)
    return [float(x) for x in stft.frequencies_hz], [float(x) for x in mean_mag]
