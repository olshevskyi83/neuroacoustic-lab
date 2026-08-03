"""Harmonic analysis tests (Milestone 3)."""

from __future__ import annotations

import math
from pathlib import Path

from neuroacoustic.analysis.harmonics import analyze_harmonics
from neuroacoustic.analysis.pitch import analyze_pitch
from neuroacoustic.analysis.spectrum import compute_stft
from neuroacoustic.analysis.stats import to_mono
from neuroacoustic.audio.loader import load_audio
from neuroacoustic.config import AppConfig
from neuroacoustic.pipeline import run_pipeline


def _harmonics(path: Path, app_config: AppConfig):
    loaded = load_audio(path, app_config)
    mono = to_mono(loaded.samples)
    sr = loaded.analysis_sample_rate
    stft = compute_stft(mono, sr, app_config.analysis)
    pitch = analyze_pitch(mono, sr, app_config.analysis)
    return pitch, analyze_harmonics(stft, pitch, app_config.analysis)


def test_additive_more_harmonics_than_sine(
    fixtures_dir: Path, app_config: AppConfig
) -> None:
    _, sine_h = _harmonics(fixtures_dir / "sine_440hz.wav", app_config)
    _, add_h = _harmonics(fixtures_dir / "additive_harmonics.wav", app_config)
    assert add_h.harmonic_count is not None
    assert sine_h.harmonic_count is not None
    assert add_h.harmonic_count > sine_h.harmonic_count
    assert add_h.harmonic_count >= 3.0


def test_additive_peak_positions(fixtures_dir: Path, app_config: AppConfig) -> None:
    pitch, harm = _harmonics(fixtures_dir / "additive_harmonics.wav", app_config)
    assert pitch.f0_median_hz is not None
    assert abs(pitch.f0_median_hz - 110.0) / 110.0 < 0.08
    expected = [110.0, 220.0, 330.0, 440.0]
    peaks = [p for p in harm.peak_frequencies_hz if p is not None]
    # Allow ~half-bin error after parabolic interpolation (Δf≈21.5 Hz).
    matched = 0
    for target in expected:
        if any(abs(p - target) < 12.0 for p in peaks):
            matched += 1
    assert matched >= 3, f"expected harmonic peaks near {expected}, got {peaks[:8]}"


def test_detuned_inharmonicity_discrimination(
    fixtures_dir: Path, app_config: AppConfig
) -> None:
    """Detuning is >> STFT bin width; estimate must clearly exceed exact stack."""
    _, exact = _harmonics(fixtures_dir / "additive_harmonics.wav", app_config)
    _, detuned = _harmonics(fixtures_dir / "detuned_partials.wav", app_config)
    assert exact.inharmonicity_estimate is not None
    assert detuned.inharmonicity_estimate is not None
    assert exact.frequency_resolution_hz is not None
    assert detuned.inharmonicity_resolution_floor is not None

    # Exact harmonics should be near zero (interpolation residual).
    assert exact.inharmonicity_estimate < 0.02
    # Detuned fixture: true mean |f_n/(n·110)-1| ≈ 0.071 for n=2..4.
    assert detuned.inharmonicity_estimate > 0.04
    # Meaningful gap — not a trivial non-null check.
    gap = detuned.inharmonicity_estimate - exact.inharmonicity_estimate
    assert gap > 0.03, (
        f"inharmonicity gap {gap:.4f} too small "
        f"(exact={exact.inharmonicity_estimate:.4f}, "
        f"detuned={detuned.inharmonicity_estimate:.4f}, "
        f"floor={detuned.inharmonicity_resolution_floor})"
    )
    # Detuned estimate should be at least comparable to the resolution floor.
    assert detuned.inharmonicity_estimate >= 0.5 * detuned.inharmonicity_resolution_floor


def test_noise_and_silence_not_confident(
    fixtures_dir: Path, app_config: AppConfig
) -> None:
    _, noise_h = _harmonics(fixtures_dir / "white_noise.wav", app_config)
    _, silence_h = _harmonics(fixtures_dir / "silence.wav", app_config)
    assert (noise_h.confidence or 0.0) < 0.45
    assert silence_h.harmonic_count is None or silence_h.frames_analyzed == 0
    assert (silence_h.confidence or 0.0) < 0.2


def test_density_uses_available_slots(
    fixtures_dir: Path, app_config: AppConfig
) -> None:
    """Pure sine → ~1 detected / N_avail; with max_harmonics=12 and F0≪Nyquist, ≈1/12."""
    _, sine_h = _harmonics(fixtures_dir / "sine_440hz.wav", app_config)
    assert sine_h.harmonic_slots_available is not None
    assert sine_h.harmonic_density is not None
    assert sine_h.harmonic_count is not None
    assert abs(sine_h.harmonic_slots_available - app_config.analysis.max_harmonics) < 0.01
    expected = sine_h.harmonic_count / sine_h.harmonic_slots_available
    assert abs(sine_h.harmonic_density - expected) < 1e-6
    assert abs(sine_h.harmonic_density - (1.0 / app_config.analysis.max_harmonics)) < 0.05


def test_relative_amplitudes_and_energy_fraction(
    fixtures_dir: Path, app_config: AppConfig
) -> None:
    _, harm = _harmonics(fixtures_dir / "additive_harmonics.wav", app_config)
    assert harm.relative_amplitudes
    assert harm.relative_amplitudes[0] is not None
    assert abs(harm.relative_amplitudes[0] - 1.0) < 1e-6
    assert harm.harmonic_energy_fraction_estimate is not None
    assert 0.0 < harm.harmonic_energy_fraction_estimate <= 1.0
    dist_sum = sum(harm.normalized_distribution)
    assert abs(dist_sum - 1.0) < 1e-6


def test_harmonics_fingerprint_no_nan(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    for name in ("additive_harmonics.wav", "silence.wav", "white_noise.wav"):
        result = run_pipeline(
            fixtures_dir / name, app_config, output_dir=tmp_path / name, plots=False
        )
        payload = result.fingerprint.to_json_dict()
        assert "harmonic_energy_fraction_estimate" in payload["harmonics"]
        assert "harmonic_to_noise_estimate" not in payload["harmonics"]

        def walk(obj: object) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)
            elif isinstance(obj, float):
                assert math.isfinite(obj)

        walk(payload)
