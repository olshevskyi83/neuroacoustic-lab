"""Harmonic analysis relative to an estimated fundamental (Milestone 3).

Given a per-frame F0 from pitch analysis and an in-memory magnitude STFT,
this module locates spectral peaks near integer multiples ``n * F0``.

Definitions
-----------
Harmonic peak frequency (Hz)
    Frequency of the magnitude peak inside a search window centered at
    ``n * F0``, refined with **parabolic interpolation** over adjacent STFT
    bins (Jacobsen / quadratic peak fit). Window half-width =
    ``harmonic_search_width_fraction * F0`` (at least one bin).

Harmonic amplitude (relative, dimensionless)
    Interpolated peak magnitude at harmonic ``n``, divided by the magnitude
    at the detected fundamental peak (n=1) in the same frame.

Detected harmonic count
    Number of harmonics ``n = 1..N`` with relative amplitude ≥
    ``harmonic_rel_amp_threshold`` and ``n * F0 < Nyquist``.

Harmonic slots available
    ``N_avail = min(max_harmonics, floor((Nyquist - ε) / F0))``.
    This is the number of integer multiples of F0 that fit below Nyquist,
    capped by the configured ``max_harmonics``.

Harmonic density (dimensionless, 0–1)
    ``mean(detected_count / N_avail)`` over usable voiced frames.
    Occupancy of **available** harmonic slots — **not** harmonics per Hz.
    Using ``N_avail`` (not always ``max_harmonics``) keeps density comparable
    when F0, sample rate, or Nyquist change (high F0 → fewer possible slots).

Normalized harmonic distribution
    Median relative amplitudes for indices ``0..max_harmonics-1``, with
    undetected slots as 0, then renormalized to sum to 1 when any peak exists.

Harmonic energy fraction estimate (linear ratio, 0–1) — **estimate**
    Per voiced frame:
    ``η = E_harm / max(E_band, floor)``
    where ``E_harm`` is the sum of squared magnitudes in the harmonic search
    windows and ``E_band`` is the sum of squared magnitudes from ~F0 to the
    last searched harmonic. Stored field:
    ``harmonic_energy_fraction_estimate``.

    This is a **linear power fraction in [0, 1]**, **not** HNR in dB, not
    cepstral HNR, and not a calibrated SNR.

Inharmonicity estimate (dimensionless relative error) — **estimate**
    Mean over detected harmonics ``n ≥ 2`` of
    ``|f_n / (n * f_1) - 1|``,
    where ``f_1`` is the **measured** (interpolated) fundamental peak
    frequency in that frame — not the possibly drifted pYIN F0.
    Search windows are still centered using pYIN F0.

Resolution floor / uncertainty
    STFT bin width ``Δf = sample_rate / n_fft``. Without interpolation, peak
    frequency uncertainty is ~Δf. Parabolic interpolation reduces this for
    isolated tones, but residual bias remains for close partials / noise.
    Reported ``inharmonicity_resolution_floor`` ≈ mean over used ``n≥2`` of
    ``Δf / (n * F0)`` — a characteristic relative-frequency uncertainty.
    Differences smaller than this floor are **not** meaningful.

Limitations
-----------
- Assumes a single F0. Polyphonic / mixed recordings may produce misleading
  harmonic stacks.
- Weak fundamentals or noise yield low counts and low confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuroacoustic.analysis.pitch import PitchFeatures
from neuroacoustic.analysis.spectrum import StftResult
from neuroacoustic.analysis.stats import DistributionStats, sanitize_array
from neuroacoustic.config import AnalysisConfig


@dataclass
class HarmonicFeatures:
    max_harmonics: int
    harmonic_slots_available: float | None
    frequency_resolution_hz: float | None
    inharmonicity_resolution_floor: float | None
    harmonic_count: float | None
    harmonic_count_median: float | None
    harmonic_density: float | None
    peak_frequencies_hz: list[float | None]
    relative_amplitudes: list[float | None]
    normalized_distribution: list[float]
    # Linear power fraction in [0, 1] — NOT HNR dB.
    harmonic_energy_fraction_estimate: float | None
    inharmonicity_estimate: float | None
    harmonic_count_stats: DistributionStats
    harmonic_energy_fraction_stats: DistributionStats
    inharmonicity_stats: DistributionStats
    confidence: float | None
    frames_analyzed: int
    warnings: list[str]


def _parabolic_peak(
    freqs: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    peak_bin: int,
) -> tuple[float, float]:
    """Refine peak frequency/magnitude with a 3-point parabolic fit."""
    if peak_bin <= 0 or peak_bin >= spectrum.size - 1:
        return float(freqs[peak_bin]), float(spectrum[peak_bin])
    alpha = float(spectrum[peak_bin - 1])
    beta = float(spectrum[peak_bin])
    gamma = float(spectrum[peak_bin + 1])
    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-20:
        return float(freqs[peak_bin]), beta
    p = 0.5 * (alpha - gamma) / denom
    p = float(np.clip(p, -0.5, 0.5))
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    freq = float(freqs[peak_bin] + p * df)
    mag = float(beta - 0.25 * (alpha - gamma) * p)
    return freq, max(mag, 0.0)


def analyze_harmonics(
    stft: StftResult,
    pitch: PitchFeatures,
    analysis: AnalysisConfig,
) -> HarmonicFeatures:
    """Estimate harmonic structure from STFT peaks guided by pYIN F0."""
    warnings: list[str] = []
    max_h = int(analysis.max_harmonics)
    empty = DistributionStats.from_values([])
    zeros_dist = [0.0] * max_h
    df = (
        float(stft.sample_rate) / float(stft.n_fft)
        if stft.n_fft > 0
        else None
    )

    if (
        pitch.voiced_frame_count == 0
        or stft.magnitude.size == 0
        or pitch.times_seconds.size == 0
    ):
        warnings.append("Harmonic analysis skipped: no voiced pitch / empty STFT")
        return HarmonicFeatures(
            max_harmonics=max_h,
            harmonic_slots_available=None,
            frequency_resolution_hz=df,
            inharmonicity_resolution_floor=None,
            harmonic_count=None,
            harmonic_count_median=None,
            harmonic_density=None,
            peak_frequencies_hz=[None] * max_h,
            relative_amplitudes=[None] * max_h,
            normalized_distribution=zeros_dist,
            harmonic_energy_fraction_estimate=None,
            inharmonicity_estimate=None,
            harmonic_count_stats=empty,
            harmonic_energy_fraction_stats=empty,
            inharmonicity_stats=empty,
            confidence=None,
            frames_analyzed=0,
            warnings=warnings,
        )

    freqs = stft.frequencies_hz
    mag = sanitize_array(stft.magnitude)
    nyquist = float(stft.sample_rate) / 2.0
    width_frac = float(analysis.harmonic_search_width_fraction)
    rel_thresh = float(analysis.harmonic_rel_amp_threshold)
    floor = float(analysis.amplitude_floor)

    stft_times = stft.times_seconds
    counts: list[float] = []
    densities: list[float] = []
    slots_list: list[float] = []
    hnrs: list[float] = []
    inharms: list[float] = []
    floor_terms: list[float] = []
    rel_lists: list[list[float]] = [[] for _ in range(max_h)]
    freq_lists: list[list[float]] = [[] for _ in range(max_h)]

    usable_idx = np.where(pitch.voiced_flag & np.isfinite(pitch.f0_hz) & (pitch.f0_hz > 0))[0]
    for pi in usable_idx:
        f0 = float(pitch.f0_hz[pi])
        t = float(pitch.times_seconds[pi])
        si = int(np.argmin(np.abs(stft_times - t)))
        spectrum = mag[:, si]
        n_avail = int(min(max_h, max(1, int(np.floor((nyquist - 1e-9) / f0)))))
        result = _frame_harmonics(
            spectrum,
            freqs,
            f0,
            max_h=max_h,
            n_avail=n_avail,
            nyquist=nyquist,
            width_frac=width_frac,
            rel_thresh=rel_thresh,
            floor=floor,
            df=df or 0.0,
        )
        if result is None:
            continue
        count, rels, peak_fs, hnr, inh, floor_term = result
        counts.append(float(count))
        slots_list.append(float(n_avail))
        densities.append(float(count / n_avail))
        hnrs.append(hnr)
        if inh is not None:
            inharms.append(inh)
        if floor_term is not None:
            floor_terms.append(floor_term)
        for n in range(max_h):
            if rels[n] is not None:
                rel_lists[n].append(float(rels[n]))
                freq_lists[n].append(float(peak_fs[n]))  # type: ignore[arg-type]

    frames_analyzed = len(counts)
    if frames_analyzed == 0:
        warnings.append("No harmonic frames could be measured despite voiced pitch")
        return HarmonicFeatures(
            max_harmonics=max_h,
            harmonic_slots_available=None,
            frequency_resolution_hz=df,
            inharmonicity_resolution_floor=None,
            harmonic_count=None,
            harmonic_count_median=None,
            harmonic_density=None,
            peak_frequencies_hz=[None] * max_h,
            relative_amplitudes=[None] * max_h,
            normalized_distribution=zeros_dist,
            harmonic_energy_fraction_estimate=None,
            inharmonicity_estimate=None,
            harmonic_count_stats=empty,
            harmonic_energy_fraction_stats=empty,
            inharmonicity_stats=empty,
            confidence=0.0,
            frames_analyzed=0,
            warnings=warnings,
        )

    count_stats = DistributionStats.from_values(counts)
    hnr_stats = DistributionStats.from_values(hnrs)
    inh_stats = DistributionStats.from_values(inharms)
    density_mean = float(np.mean(densities))
    slots_mean = float(np.mean(slots_list))
    res_floor = float(np.mean(floor_terms)) if floor_terms else None

    peak_freqs: list[float | None] = []
    rel_amps: list[float | None] = []
    for n in range(max_h):
        if freq_lists[n]:
            peak_freqs.append(float(np.median(freq_lists[n])))
            rel_amps.append(float(np.median(rel_lists[n])))
        else:
            peak_freqs.append(None)
            rel_amps.append(None)

    raw = [float(a) if a is not None and a > 0 else 0.0 for a in rel_amps]
    total = float(sum(raw))
    if total > 0:
        normalized = [v / total for v in raw]
    else:
        normalized = zeros_dist
        warnings.append("Normalized harmonic distribution empty (no peaks above threshold)")

    mean_count = count_stats.mean
    hnr_mean = hnr_stats.mean
    inh_mean = inh_stats.mean

    pitch_conf = pitch.confidence if pitch.confidence is not None else 0.0
    count_factor = min(1.0, (mean_count or 0.0) / 3.0)
    conf = float(np.clip(pitch_conf * 0.6 + count_factor * 0.4, 0.0, 1.0))

    if hnr_mean is not None and hnr_mean < 0.15 and (pitch.voiced_ratio or 0) < 0.2:
        warnings.append(
            "Unreliable harmonic estimates (noise-like / low harmonic energy fraction)"
        )
    if pitch.f0_median_hz is not None and (mean_count or 0) < 1.0:
        warnings.append("Weak fundamental / few harmonic peaks detected")
    if (
        inh_mean is not None
        and res_floor is not None
        and inh_mean < res_floor
    ):
        warnings.append(
            "Inharmonicity estimate below STFT resolution floor — treat as unresolved"
        )

    return HarmonicFeatures(
        max_harmonics=max_h,
        harmonic_slots_available=slots_mean,
        frequency_resolution_hz=df,
        inharmonicity_resolution_floor=res_floor,
        harmonic_count=mean_count,
        harmonic_count_median=count_stats.median,
        harmonic_density=density_mean,
        peak_frequencies_hz=peak_freqs,
        relative_amplitudes=rel_amps,
        normalized_distribution=normalized,
        harmonic_energy_fraction_estimate=hnr_mean,
        inharmonicity_estimate=inh_mean,
        harmonic_count_stats=count_stats,
        harmonic_energy_fraction_stats=hnr_stats,
        inharmonicity_stats=inh_stats,
        confidence=conf,
        frames_analyzed=frames_analyzed,
        warnings=warnings,
    )


def _frame_harmonics(
    spectrum: NDArray[np.float64],
    freqs: NDArray[np.float64],
    f0: float,
    *,
    max_h: int,
    n_avail: int,
    nyquist: float,
    width_frac: float,
    rel_thresh: float,
    floor: float,
    df: float,
) -> tuple[
    int,
    list[float | None],
    list[float | None],
    float,
    float | None,
    float | None,
] | None:
    if f0 <= 0 or not np.isfinite(f0):
        return None

    rels: list[float | None] = [None] * max_h
    peak_fs: list[float | None] = [None] * max_h
    fund_mag = None
    fund_freq = None
    harm_energy = 0.0
    last_n = min(max_h, n_avail)
    band_lo = f0 * (1.0 - width_frac)
    band_hi = min(nyquist, f0 * last_n * (1.0 + width_frac))
    band_mask = (freqs >= max(0.0, band_lo)) & (freqs <= band_hi)
    total_energy = float(np.sum(spectrum[band_mask] ** 2)) if np.any(band_mask) else 0.0

    inharm_terms: list[float] = []
    floor_terms: list[float] = []
    count = 0

    for n in range(1, last_n + 1):
        target = n * f0
        if target >= nyquist:
            break
        half = max(width_frac * target, 2.0 * df if df > 0 else 1.0)
        # Keep windows from swallowing the neighboring harmonic (±F0).
        half = min(half, 0.45 * f0)
        lo = target - half
        hi = target + half
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            continue
        global_idxs = np.flatnonzero(mask)
        local_idx = int(np.argmax(spectrum[mask]))
        peak_bin = int(global_idxs[local_idx])
        freq, mag = _parabolic_peak(freqs, spectrum, peak_bin)
        harm_energy += mag * mag

        if n == 1:
            fund_mag = max(mag, floor)
            fund_freq = freq
            rels[0] = 1.0
            peak_fs[0] = freq
            if mag >= floor:
                count += 1
            continue

        assert fund_mag is not None and fund_freq is not None
        rel = float(mag / fund_mag)
        if rel >= rel_thresh:
            rels[n - 1] = rel
            peak_fs[n - 1] = freq
            count += 1
            # Inharmonicity relative to the *measured* fundamental peak, not
            # pYIN F0 (which can drift on inharmonic stacks).
            inharm_terms.append(abs(freq / (n * fund_freq) - 1.0))
            if df > 0 and fund_freq > 0:
                floor_terms.append(df / (n * fund_freq))

    if fund_mag is None:
        return None

    energy_frac = float(np.clip(harm_energy / max(total_energy, floor), 0.0, 1.0))
    inh = float(np.mean(inharm_terms)) if inharm_terms else None
    res_floor = float(np.mean(floor_terms)) if floor_terms else None
    return count, rels, peak_fs, energy_frac, inh, res_floor
