"""Shared pytest fixtures for Milestone 1."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroacoustic.config import AppConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    """Ensure synthetic fixtures exist (generated once per session)."""
    out = repo_root / "tests" / "fixtures"
    needed = [
        "sine_110hz.wav",
        "sine_440hz.wav",
        "additive_harmonics.wav",
        "white_noise.wav",
        "silence.wav",
        "impulse_decay.wav",
        "stereo_correlated.wav",
    ]
    if not all((out / name).exists() for name in needed):
        import sys

        scripts = str(repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from generate_test_signals import write_all

        write_all(out)
    return out


@pytest.fixture
def app_config(repo_root: Path) -> AppConfig:
    return load_config(repo_root / "config" / "default.toml")
