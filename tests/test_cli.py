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


def test_probe_missing_file_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["probe", str(tmp_path / "nope.wav")])
    assert result.exit_code != 0


def test_analyze_not_implemented(fixtures_dir: Path) -> None:
    result = runner.invoke(app, ["analyze", str(fixtures_dir / "sine_440hz.wav")])
    assert result.exit_code == 2
    assert "Milestone 1" in result.stdout or "Milestone 1" in result.stderr


def test_db_list_not_implemented() -> None:
    result = runner.invoke(app, ["db", "list"])
    assert result.exit_code == 2
