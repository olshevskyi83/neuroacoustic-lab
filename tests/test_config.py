"""Tests for configuration and Pydantic source models."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroacoustic.config import load_config
from neuroacoustic.fingerprint.models import SourceMetadata


def test_load_default_config(repo_root: Path) -> None:
    cfg = load_config(repo_root / "config" / "default.toml")
    assert cfg.project.schema_version == "0.4.0"
    assert cfg.project.analysis_version == "0.4.0"
    assert "wav" in cfg.audio.supported_extensions
    assert cfg.audio.analysis_sample_rate is None
    assert cfg.analysis.n_fft == 2048
    assert cfg.analysis.hop_length == 512


def test_env_override_output_dir(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROACOUSTIC_OUTPUT_DIR", "/tmp/neuroacoustic-out")
    cfg = load_config(repo_root / "config" / "default.toml", resolve=False)
    assert Path(cfg.paths.output_dir) == Path("/tmp/neuroacoustic-out")


def test_env_override_sample_rate(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROACOUSTIC_ANALYSIS_SAMPLE_RATE", "48000")
    cfg = load_config(repo_root / "config" / "default.toml")
    assert cfg.audio.analysis_sample_rate == 48000


def test_source_metadata_rejects_bad_hash() -> None:
    with pytest.raises(ValueError):
        SourceMetadata(
            filename="x.wav",
            path="/tmp/x.wav",
            content_hash="deadbeef",
            file_size_bytes=1,
        )
