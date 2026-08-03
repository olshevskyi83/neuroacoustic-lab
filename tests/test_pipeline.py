"""Pipeline integration tests (Milestone 2)."""

from __future__ import annotations

import json
from pathlib import Path

from neuroacoustic.config import AppConfig
from neuroacoustic.pipeline import run_pipeline


def test_pipeline_writes_json_and_plots(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    result = run_pipeline(
        fixtures_dir / "additive_harmonics.wav",
        app_config,
        output_dir=tmp_path,
        plots=True,
    )
    assert result.fingerprint_path.exists()
    payload = json.loads(result.fingerprint_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "0.4.0"
    arts = result.fingerprint.artifacts
    assert arts.waveform_png and Path(arts.waveform_png).exists()
    assert arts.fft_png and Path(arts.fft_png).exists()
    assert arts.fft_log_png and Path(arts.fft_log_png).exists()
    assert arts.spectrogram_png and Path(arts.spectrogram_png).exists()


def test_pipeline_compare_four_signals(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    names = [
        "sine_440hz.wav",
        "additive_harmonics.wav",
        "white_noise.wav",
        "silence.wav",
    ]
    results = {}
    for name in names:
        results[name] = run_pipeline(
            fixtures_dir / name,
            app_config,
            output_dir=tmp_path / Path(name).stem,
            plots=False,
        ).fingerprint

    sine = results["sine_440hz.wav"]
    noise = results["white_noise.wav"]
    silence = results["silence.wav"]
    additive = results["additive_harmonics.wav"]

    assert abs((sine.spectral.fft.peak_frequency_hz or 0) - 440.0) < 5.0
    assert (noise.spectral.flatness.mean or 0) > (sine.spectral.flatness.mean or 1)
    assert (silence.energy.peak_amplitude or 0) < 1e-5
    assert additive.spectral.fft.peaks
    for fp in results.values():
        text = json.dumps(fp.to_json_dict())
        assert "NaN" not in text
        assert "Infinity" not in text
