"""CLI smoke tests for Milestone 1."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from neuroacoustic.cli import app

runner = CliRunner()


def test_doctor_exits_zero() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "neuroacoustic" in result.stdout.lower() or "Doctor" in result.stdout or "python" in result.stdout


def test_doctor_detects_ffmpeg_when_available() -> None:
    from neuroacoustic.audio.probe import ffmpeg_available, ffprobe_available

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    if ffmpeg_available() and ffprobe_available():
        assert "ffmpeg" in result.stdout
        assert "ffprobe" in result.stdout
        # Status column should show ok for both tools.
        assert "All core checks passed" in result.stdout
        assert "/usr/bin/ffmpeg" in result.stdout or "ffmpeg" in result.stdout.lower()


def test_probe_wav_table(fixtures_dir: Path) -> None:
    path = fixtures_dir / "sine_440hz.wav"
    result = runner.invoke(app, ["probe", str(path)])
    assert result.exit_code == 0
    assert "content_hash" in result.stdout
    assert "44100" in result.stdout


def test_probe_json(fixtures_dir: Path) -> None:
    path = fixtures_dir / "sine_110hz.wav"
    result = runner.invoke(app, ["probe", str(path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["source"]["filename"] == "sine_110hz.wav"
    assert len(payload["source"]["content_hash"]) == 64


def test_probe_mp3_cli_when_ffmpeg_available(fixtures_dir: Path, tmp_path: Path) -> None:
    import subprocess

    from neuroacoustic.audio.probe import ffmpeg_available, ffprobe_available

    if not (ffmpeg_available() and ffprobe_available()):
        import pytest

        pytest.skip("ffmpeg/ffprobe not available")

    mp3_path = tmp_path / "cli_sine.mp3"
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(fixtures_dir / "sine_110hz.wav"),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(mp3_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if encoded.returncode != 0:
        import pytest

        pytest.skip(f"ffmpeg MP3 encode failed: {encoded.stderr.strip()}")

    result = runner.invoke(app, ["probe", str(mp3_path), "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["source"]["filename"] == "cli_sine.mp3"
    assert payload["source"]["probe_backend"] == "ffprobe"
    assert payload["source"]["native_sample_rate"] == 44100


def test_probe_missing_file_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["probe", str(tmp_path / "nope.wav")])
    assert result.exit_code != 0


def test_analyze_runs(fixtures_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "analyze",
            str(fixtures_dir / "sine_440hz.wav"),
            "--output-dir",
            str(tmp_path),
            "--no-plots",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "fingerprint" in result.stdout.lower() or "centroid" in result.stdout.lower()


def test_analyze_json(fixtures_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "analyze",
            str(fixtures_dir / "white_noise.wav"),
            "--output-dir",
            str(tmp_path),
            "--no-plots",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "0.4.0"
    assert "spectral" in payload
    assert "energy" in payload


def test_db_list_not_implemented() -> None:
    result = runner.invoke(app, ["db", "list"])
    assert result.exit_code == 2
