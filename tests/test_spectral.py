"""Spectral and energy feature tests (Milestone 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuroacoustic.analysis.energy import analyze_energy
from neuroacoustic.analysis.spectral import analyze_spectral
from neuroacoustic.analysis.spectrum import compute_fft_summary, compute_stft
from neuroacoustic.analysis.stats import to_mono
from neuroacoustic.audio.loader import load_audio
from neuroacoustic.config import AppConfig


def _mono_features(path: Path, app_config: AppConfig):
    loaded = load_audio(path, app_config)
    mono = to_mono(loaded.samples)
    sr = loaded.analysis_sample_rate
    fft = compute_fft_summary(mono, sr, app_config.analysis)
    stft = compute_stft(mono, sr, app_config.analysis)
    spectral = analyze_spectral(stft, app_config.analysis)
    energy = analyze_energy(mono, sr, app_config.analysis)
    return loaded, fft, spectral, energy


def test_sine_440_peak_near_440(fixtures_dir: Path, app_config: AppConfig) -> None:
    _, fft, spectral, energy = _mono_features(fixtures_dir / "sine_440hz.wav", app_config)
    assert fft.peak_frequency_hz is not None
    assert abs(fft.peak_frequency_hz - 440.0) < 5.0
    assert spectral.centroid_hz.mean is not None
    assert abs(spectral.centroid_hz.mean - 440.0) < 40.0
    assert energy.peak_amplitude is not None and energy.peak_amplitude > 0.1
    assert energy.rms is not None and energy.rms > 0.05


def test_sine_110_peak_near_110(fixtures_dir: Path, app_config: AppConfig) -> None:
    _, fft, spectral, _ = _mono_features(fixtures_dir / "sine_110hz.wav", app_config)
    assert fft.peak_frequency_hz is not None
    assert abs(fft.peak_frequency_hz - 110.0) < 5.0
    assert spectral.centroid_hz.mean is not None
    assert abs(spectral.centroid_hz.mean - 110.0) < 30.0


def test_additive_peaks_near_harmonics(fixtures_dir: Path, app_config: AppConfig) -> None:
    _, fft, _, _ = _mono_features(fixtures_dir / "additive_harmonics.wav", app_config)
    peaks = sorted(fft.peak_frequencies_hz)
    assert peaks, "expected harmonic peaks"
    # Strongest components should include 110 Hz family members within tolerance.
    expected = [110.0, 220.0, 330.0, 440.0]
    matched = 0
    for target in expected:
        if any(abs(p - target) < 8.0 for p in peaks[:8]):
            matched += 1
    assert matched >= 3, f"expected >=3 harmonic matches, got peaks={peaks[:8]}"


def test_noise_higher_flatness_and_entropy_than_sine(
    fixtures_dir: Path, app_config: AppConfig
) -> None:
    _, _, sine_spec, _ = _mono_features(fixtures_dir / "sine_440hz.wav", app_config)
    _, _, noise_spec, _ = _mono_features(fixtures_dir / "white_noise.wav", app_config)
    assert sine_spec.flatness.mean is not None
    assert noise_spec.flatness.mean is not None
    assert noise_spec.flatness.mean > sine_spec.flatness.mean
    assert sine_spec.entropy.mean is not None
    assert noise_spec.entropy.mean is not None
    assert noise_spec.entropy.mean > sine_spec.entropy.mean


def test_silence_safe_energy_and_spectral(
    fixtures_dir: Path, app_config: AppConfig
) -> None:
    loaded, fft, spectral, energy = _mono_features(fixtures_dir / "silence.wav", app_config)
    assert loaded.quality.peak_amplitude < app_config.audio.silence_peak_threshold
    assert energy.peak_amplitude == 0.0 or energy.peak_amplitude is not None
    assert energy.rms is not None
    # No NaNs in distribution stats
    for stats in (
        spectral.centroid_hz,
        spectral.flatness,
        spectral.entropy,
        energy.rms_frame_stats,
    ):
        for value in (
            stats.mean,
            stats.std,
            stats.median,
            stats.p05,
            stats.p95,
            stats.minimum,
            stats.maximum,
        ):
            if value is not None:
                assert np.isfinite(value)


def test_crest_factor_and_dynamic_range_estimate(
    fixtures_dir: Path, app_config: AppConfig
) -> None:
    _, _, _, energy = _mono_features(fixtures_dir / "sine_440hz.wav", app_config)
    assert energy.crest_factor is not None and energy.crest_factor >= 1.0
    # Pure sine crest factor ≈ sqrt(2) ≈ 1.414
    assert abs(energy.crest_factor - np.sqrt(2)) < 0.15
    assert energy.estimated_dynamic_range_db is not None
    assert np.isfinite(energy.estimated_dynamic_range_db)


def test_short_file_does_not_crash(tmp_path: Path, app_config: AppConfig) -> None:
    import soundfile as sf

    sr = 44100
    y = (0.2 * np.sin(2 * np.pi * 440 * np.arange(int(0.05 * sr)) / sr)).astype(np.float32)
    path = tmp_path / "short_sine.wav"
    sf.write(str(path), y, sr)
    _, fft, spectral, energy = _mono_features(path, app_config)
    assert fft.peak_frequency_hz is not None
    assert spectral.flatness.count >= 0
    assert energy.rms is not None
