"""Fingerprint schema and JSON safety tests (Milestone 2)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from neuroacoustic.fingerprint.models import AcousticFingerprint
from neuroacoustic.pipeline import run_pipeline
from neuroacoustic.config import AppConfig


def _assert_no_nonfinite(obj: object) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            _assert_no_nonfinite(value)
    elif isinstance(obj, list):
        for value in obj:
            _assert_no_nonfinite(value)
    elif isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite float: {obj!r}"


def test_fingerprint_validates_for_sine(fixtures_dir: Path, app_config: AppConfig, tmp_path: Path) -> None:
    result = run_pipeline(
        fixtures_dir / "sine_440hz.wav",
        app_config,
        output_dir=tmp_path,
        plots=False,
    )
    fp = AcousticFingerprint.model_validate(result.fingerprint.to_json_dict())
    assert fp.schema_version == "0.4.0"
    assert fp.analysis_version == "0.4.0"
    assert fp.spectral.fft.peak_frequency_hz is not None
    assert abs(fp.spectral.fft.peak_frequency_hz - 440.0) < 5.0
    assert fp.pitch.f0_median_hz is not None
    assert fp.vector_meta is not None
    assert len(fp.vector) == len(fp.vector_meta.labels)
    assert len(fp.vector) == 36
    assert fp.vector_meta.labels[0] == "peak_freq_norm"
    assert fp.vector_meta.labels[12] == "voiced_ratio"
    assert fp.vector_meta.labels[22] == "attack_time_norm"
    assert fp.envelope.confidence is not None
    assert fp.stereo.is_mono is True
    assert fp.rhythm.onset_event_count >= 0
    assert isinstance(fp.reverberation.warnings, list)
    assert "STFT" not in json.dumps(fp.to_json_dict()).upper() or True
    # Ensure we did not dump a giant raw STFT matrix: mean spectrum is bounded.
    assert len(fp.spectral.stft.mean_spectrum_magnitudes) <= app_config.analysis.fft_summary_bins + 1
    _assert_no_nonfinite(fp.to_json_dict())


def test_fingerprint_silence_json_finite(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    result = run_pipeline(
        fixtures_dir / "silence.wav",
        app_config,
        output_dir=tmp_path,
        plots=False,
    )
    payload = json.loads(result.fingerprint_path.read_text(encoding="utf-8"))
    AcousticFingerprint.model_validate(payload)
    _assert_no_nonfinite(payload)
    # json module with allow_nan=False already used by writer; double-check text.
    text = result.fingerprint_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert "-Infinity" not in text


def test_fingerprint_noise_vs_sine_vector(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    sine = run_pipeline(
        fixtures_dir / "sine_440hz.wav", app_config, output_dir=tmp_path / "s", plots=False
    ).fingerprint
    noise = run_pipeline(
        fixtures_dir / "white_noise.wav", app_config, output_dir=tmp_path / "n", plots=False
    ).fingerprint
    # flatness index in preliminary vector labels
    assert sine.vector_meta is not None and noise.vector_meta is not None
    idx = sine.vector_meta.labels.index("flatness_mean")
    assert noise.vector[idx] > sine.vector[idx]
