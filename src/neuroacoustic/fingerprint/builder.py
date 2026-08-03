"""Build versioned AcousticFingerprint documents from analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroacoustic.analysis.energy import EnergyFeatures
from neuroacoustic.analysis.harmonics import HarmonicFeatures
from neuroacoustic.analysis.pitch import PitchFeatures
from neuroacoustic.analysis.spectral import SpectralFeatures
from neuroacoustic.analysis.spectrum import FftSummaryData, StftResult
from neuroacoustic.analysis.stats import DistributionStats, TimeSeriesSummary
from neuroacoustic.fingerprint.models import (
    AcousticFingerprint,
    ArtifactPaths,
    EnergySection,
    FftPeak,
    FftSummary,
    HarmonicsSection,
    PitchSection,
    QualityMetrics,
    SpectralSection,
    StftSummary,
    build_analysis_config_snapshot,
)
from neuroacoustic.fingerprint.normalization import build_preliminary_vector

if TYPE_CHECKING:
    from neuroacoustic.audio.loader import LoadedAudio
    from neuroacoustic.config import AppConfig


def build_fingerprint(
    loaded: LoadedAudio,
    config: AppConfig,
    *,
    fft: FftSummaryData,
    stft: StftResult,
    spectral: SpectralFeatures,
    energy: EnergyFeatures,
    pitch: PitchFeatures,
    harmonics: HarmonicFeatures,
    artifacts: ArtifactPaths | None = None,
) -> AcousticFingerprint:
    """Assemble the versioned fingerprint (no raw STFT / pitch matrices)."""
    warnings = list(loaded.quality.warnings)
    warnings.extend(spectral.warnings)
    warnings.extend(energy.warnings)
    warnings.extend(pitch.warnings)
    warnings.extend(harmonics.warnings)

    quality = QualityMetrics(
        clipping_ratio=loaded.quality.clipping_ratio,
        silence_ratio=loaded.quality.silence_ratio,
        peak_amplitude=loaded.quality.peak_amplitude,
        warnings=warnings,
    )

    fft_section = FftSummary(
        peak_frequency_hz=fft.peak_frequency_hz,
        peak_magnitude=fft.peak_magnitude,
        peaks=[
            FftPeak(frequency_hz=f, magnitude=m)
            for f, m in zip(fft.peak_frequencies_hz, fft.peak_magnitudes, strict=False)
        ],
        summary_frequencies_hz=fft.summary_frequencies_hz,
        summary_magnitudes=fft.summary_magnitudes,
        log_summary_frequencies_hz=fft.log_summary_frequencies_hz,
        log_summary_magnitudes=fft.log_summary_magnitudes,
    )

    duration = None
    if stft.times_seconds.size:
        duration = float(stft.times_seconds[-1])
    elif loaded.source.duration_seconds is not None:
        duration = float(loaded.source.duration_seconds)

    spectral_section = SpectralSection(
        fft=fft_section,
        stft=StftSummary(
            n_fft=stft.n_fft,
            hop_length=stft.hop_length,
            win_length=stft.win_length,
            n_frames=spectral.stft_n_frames,
            n_freq_bins=spectral.stft_n_freq_bins,
            duration_seconds=duration,
            mean_spectrum_frequencies_hz=spectral.mean_spectrum_hz,
            mean_spectrum_magnitudes=spectral.mean_spectrum_magnitude,
        ),
        centroid_hz=spectral.centroid_hz,
        bandwidth_hz=spectral.bandwidth_hz,
        rolloff_hz=spectral.rolloff_hz,
        flatness=spectral.flatness,
        entropy=spectral.entropy,
        contrast_db=spectral.contrast_db,
        contrast_band_means_db=spectral.contrast_band_means_db,
        centroid_curve=TimeSeriesSummary(
            times_seconds=spectral.centroid_series[0],
            values=list(spectral.centroid_series[1]),
            unit="Hz",
            description="Downsampled spectral centroid over time",
        ),
        flatness_curve=TimeSeriesSummary(
            times_seconds=spectral.flatness_series[0],
            values=list(spectral.flatness_series[1]),
            unit="dimensionless",
            description="Downsampled spectral flatness over time",
        ),
    )

    energy_section = EnergySection(
        rms=energy.rms,
        rms_db=energy.rms_db,
        peak_amplitude=energy.peak_amplitude,
        crest_factor=energy.crest_factor,
        crest_factor_db=energy.crest_factor_db,
        zero_crossing_rate=energy.zero_crossing_rate,
        estimated_dynamic_range_db=energy.estimated_dynamic_range_db,
        rms_frame=energy.rms_frame_stats,
        zero_crossing_rate_frame=energy.zcr_frame_stats,
        rms_curve=TimeSeriesSummary(
            times_seconds=energy.rms_series[0],
            values=list(energy.rms_series[1]),
            unit="linear_amplitude",
            description="Downsampled frame RMS over time",
        ),
    )

    pitch_section = PitchSection(
        method=pitch.method,
        f0_min_hz=pitch.f0_min_hz,
        f0_max_hz=pitch.f0_max_hz,
        frame_count=pitch.frame_count,
        voiced_frame_count=pitch.voiced_frame_count,
        voiced_ratio=pitch.voiced_ratio,
        mean_voiced_probability=pitch.mean_voiced_probability,
        confidence=pitch.confidence,
        f0_voiced_hz=pitch.f0_voiced if pitch.f0_voiced.count else DistributionStats.from_values([]),
        f0_median_hz=pitch.f0_median_hz,
        f0_curve=TimeSeriesSummary(
            times_seconds=pitch.f0_curve_times,
            values=list(pitch.f0_curve_values),
            unit="Hz",
            description="Downsampled F0; null where unvoiced / unavailable",
        ),
        voiced_probability_curve=TimeSeriesSummary(
            times_seconds=pitch.voiced_prob_curve_times,
            values=list(pitch.voiced_prob_curve_values),
            unit="probability",
            description="Downsampled pYIN voiced probability",
        ),
    )

    harmonics_section = HarmonicsSection(
        max_harmonics=harmonics.max_harmonics,
        harmonic_slots_available=harmonics.harmonic_slots_available,
        frequency_resolution_hz=harmonics.frequency_resolution_hz,
        inharmonicity_resolution_floor=harmonics.inharmonicity_resolution_floor,
        harmonic_count=harmonics.harmonic_count,
        harmonic_count_median=harmonics.harmonic_count_median,
        harmonic_density=harmonics.harmonic_density,
        peak_frequencies_hz=list(harmonics.peak_frequencies_hz),
        relative_amplitudes=list(harmonics.relative_amplitudes),
        normalized_distribution=list(harmonics.normalized_distribution),
        harmonic_energy_fraction_estimate=harmonics.harmonic_energy_fraction_estimate,
        inharmonicity_estimate=harmonics.inharmonicity_estimate,
        harmonic_count_frame=harmonics.harmonic_count_stats,
        harmonic_energy_fraction_estimate_frame=harmonics.harmonic_energy_fraction_stats,
        inharmonicity_estimate_frame=harmonics.inharmonicity_stats,
        confidence=harmonics.confidence,
        frames_analyzed=harmonics.frames_analyzed,
    )

    vector_meta = build_preliminary_vector(
        fft=fft,
        spectral=spectral,
        energy=energy,
        pitch=pitch,
        harmonics=harmonics,
        sample_rate=loaded.analysis_sample_rate,
        analysis=config.analysis,
    )

    return AcousticFingerprint(
        schema_version=config.project.schema_version,
        analysis_version=config.project.analysis_version,
        source=loaded.source,
        analysis_config=build_analysis_config_snapshot(
            config, analysis_sample_rate=loaded.analysis_sample_rate
        ),
        quality=quality,
        spectral=spectral_section,
        pitch=pitch_section,
        harmonics=harmonics_section,
        energy=energy_section,
        vector=list(vector_meta.values),
        vector_meta=vector_meta,
        artifacts=artifacts or ArtifactPaths(),
    )


__all__ = ["build_analysis_config_snapshot", "build_fingerprint"]
