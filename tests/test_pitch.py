"""Pitch / voicing tests (Milestone 3)."""

from __future__ import annotations

import math
from pathlib import Path

from neuroacoustic.analysis.pitch import analyze_pitch
from neuroacoustic.analysis.stats import to_mono
from neuroacoustic.audio.loader import load_audio
from neuroacoustic.config import AppConfig
from neuroacoustic.pipeline import run_pipeline


def _pitch(path: Path, app_config: AppConfig):
    loaded = load_audio(path, app_config)
    return analyze_pitch(to_mono(loaded.samples), loaded.analysis_sample_rate, app_config.analysis)


def test_sine_440_f0(fixtures_dir: Path, app_config: AppConfig) -> None:
    pitch = _pitch(fixtures_dir / "sine_440hz.wav", app_config)
    assert pitch.f0_median_hz is not None
    # pYIN on a clean tone: allow ~3% tolerance (resolution / windowing).
    assert abs(pitch.f0_median_hz - 440.0) / 440.0 < 0.03
    assert pitch.voiced_ratio is not None and pitch.voiced_ratio > 0.5
    assert pitch.confidence is not None and pitch.confidence > 0.4
    assert all(v is None or v > 0 for v in pitch.f0_curve_values)


def test_sine_110_f0(fixtures_dir: Path, app_config: AppConfig) -> None:
    pitch = _pitch(fixtures_dir / "sine_110hz.wav", app_config)
    assert pitch.f0_median_hz is not None
    assert abs(pitch.f0_median_hz - 110.0) / 110.0 < 0.05
    assert pitch.voiced_frame_count >= app_config.analysis.min_voiced_frames


def test_silence_pitch_null(fixtures_dir: Path, app_config: AppConfig) -> None:
    pitch = _pitch(fixtures_dir / "silence.wav", app_config)
    assert pitch.f0_median_hz is None
    assert pitch.voiced_frame_count == 0
    assert pitch.f0_voiced.count == 0
    # Must not fabricate 0 Hz as a pitch value in the curve.
    assert all(v is None for v in pitch.f0_curve_values)


def test_noise_low_voicing(fixtures_dir: Path, app_config: AppConfig) -> None:
    pitch = _pitch(fixtures_dir / "white_noise.wav", app_config)
    # White noise should not yield a confident stable pitch fingerprint.
    assert (pitch.voiced_ratio or 0.0) < 0.25 or (pitch.confidence or 0.0) < 0.35


def test_short_audio_safe(fixtures_dir: Path, app_config: AppConfig) -> None:
    pitch = _pitch(fixtures_dir / "short_sine_440hz.wav", app_config)
    # May or may not find voiced frames; must not crash or invent 0 Hz median.
    if pitch.f0_median_hz is not None:
        assert pitch.f0_median_hz > 0
        assert abs(pitch.f0_median_hz - 440.0) / 440.0 < 0.1


def test_pitch_fingerprint_json_finite(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    result = run_pipeline(
        fixtures_dir / "sine_440hz.wav", app_config, output_dir=tmp_path, plots=False
    )
    payload = result.fingerprint.to_json_dict()
    assert payload["pitch"]["f0_median_hz"] is not None
    assert payload["pitch"]["f0_median_hz"] != 0

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
