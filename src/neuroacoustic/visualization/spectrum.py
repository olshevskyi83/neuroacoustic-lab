"""Non-interactive spectrum / waveform plots (Milestone 2).

Uses the matplotlib Agg backend — never opens a GUI window.
Axes are labeled with physical units (seconds, Hz, linear amplitude / dB).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuroacoustic.analysis.spectrum import FftSummaryData, StftResult
from neuroacoustic.logging import get_logger

logger = get_logger(__name__)


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _atomic_savefig(fig, out_path: Path, *, dpi: int = 120) -> None:  # type: ignore[no-untyped-def]
    """Save a figure via a same-directory temp file then ``os.replace``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix or ".png"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=suffix,
        dir=str(out_path.parent),
    )
    os.close(fd)
    try:
        fig.savefig(tmp_name, dpi=dpi, format=suffix.lstrip(".") or "png")
        os.replace(tmp_name, out_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def plot_waveform(
    samples: NDArray[np.floating],
    sample_rate: int,
    out_path: Path,
    *,
    title: str = "Waveform",
) -> Path:
    """Save a time-domain waveform overview (amplitude vs seconds)."""
    _use_agg()
    import matplotlib.pyplot as plt

    y = np.asarray(samples, dtype=np.float64)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    t = np.arange(y.size, dtype=np.float64) / float(sample_rate)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, y, color="#1f4e79", linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (linear)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _atomic_savefig(fig, out_path)
    plt.close(fig)
    logger.info("Wrote waveform plot %s", out_path)
    return out_path


def plot_fft_magnitude(
    fft: FftSummaryData,
    out_path: Path,
    *,
    title: str = "FFT magnitude",
    log_frequency: bool = False,
) -> Path:
    """Save linear- or log-frequency FFT magnitude plot."""
    _use_agg()
    import matplotlib.pyplot as plt

    if log_frequency and fft.log_summary_frequencies_hz:
        freqs = np.asarray(fft.log_summary_frequencies_hz, dtype=np.float64)
        mags = np.asarray(fft.log_summary_magnitudes, dtype=np.float64)
    elif fft.summary_frequencies_hz:
        freqs = np.asarray(fft.summary_frequencies_hz, dtype=np.float64)
        mags = np.asarray(fft.summary_magnitudes, dtype=np.float64)
    else:
        freqs = fft.frequencies_hz
        mags = fft.magnitude

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, mags, color="#0b6e4f", linewidth=0.9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (linear)")
    ax.set_title(title)
    if log_frequency:
        ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _atomic_savefig(fig, out_path)
    plt.close(fig)
    logger.info("Wrote FFT plot %s", out_path)
    return out_path


def plot_spectrogram(
    stft: StftResult,
    out_path: Path,
    *,
    title: str = "Spectrogram",
    amplitude_floor: float = 1e-12,
) -> Path:
    """Save a log-power spectrogram (dB relative to peak, not LUFS)."""
    _use_agg()
    import matplotlib.pyplot as plt

    mag = np.maximum(stft.magnitude, amplitude_floor)
    # Power in dB relative to the spectrogram peak (display scale only).
    power_db = 20.0 * np.log10(mag / np.maximum(np.max(mag), amplitude_floor))

    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [
        float(stft.times_seconds[0]) if stft.times_seconds.size else 0.0,
        float(stft.times_seconds[-1]) if stft.times_seconds.size else 0.0,
        float(stft.frequencies_hz[0]) if stft.frequencies_hz.size else 0.0,
        float(stft.frequencies_hz[-1]) if stft.frequencies_hz.size else 0.0,
    ]
    im = ax.imshow(
        power_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Power (dB re peak)")
    fig.tight_layout()
    _atomic_savefig(fig, out_path)
    plt.close(fig)
    logger.info("Wrote spectrogram plot %s", out_path)
    return out_path
