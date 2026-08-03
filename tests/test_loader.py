"""Tests for audio probe and loader (Milestone 1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from neuroacoustic.audio.loader import load_audio
from neuroacoustic.audio.preprocessing import to_float32
from neuroacoustic.audio.probe import probe_audio, sha256_file, validate_audio_path
from neuroacoustic.config import AppConfig
from neuroacoustic.exceptions import AudioValidationError, DependencyError
from neuroacoustic.fingerprint.models import ProbeResult, SourceMetadata


def test_sha256_matches_hashlib(fixtures_dir: Path) -> None:
    path = fixtures_dir / "sine_440hz.wav"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == expected


def test_probe_sine_metadata(fixtures_dir: Path, app_config: AppConfig) -> None:
    result = probe_audio(fixtures_dir / "sine_110hz.wav", app_config)
    assert isinstance(result, ProbeResult)
    SourceMetadata.model_validate(result.source.model_dump())
    src = result.source
    assert src.filename == "sine_110hz.wav"
    assert src.native_sample_rate == 44100
    assert src.channels == 1
    assert src.duration_seconds is not None
    assert abs(src.duration_seconds - 1.0) < 0.02
    assert src.analysis_sample_rate == 44100
    assert len(src.content_hash) == 64
    assert src.bit_depth == 16


def test_probe_rejects_missing_file(tmp_path: Path, app_config: AppConfig) -> None:
    with pytest.raises(AudioValidationError, match="does not exist"):
        probe_audio(tmp_path / "missing.wav", app_config)


def test_probe_rejects_unsupported_extension(tmp_path: Path, app_config: AppConfig) -> None:
    bad = tmp_path / "file.txt"
    bad.write_text("not audio")
    with pytest.raises(AudioValidationError, match="Unsupported extension"):
        validate_audio_path(bad, app_config)


def test_load_sine_float_range(fixtures_dir: Path, app_config: AppConfig) -> None:
    loaded = load_audio(fixtures_dir / "sine_440hz.wav", app_config)
    assert loaded.samples.dtype == np.float32
    assert loaded.samples.ndim == 1
    assert np.isfinite(loaded.samples).all()
    assert float(np.max(np.abs(loaded.samples))) <= 1.0 + 1e-5
    assert loaded.decode_backend == "soundfile"
    assert loaded.quality.peak_amplitude > 0.1


def test_load_silence_quality(fixtures_dir: Path, app_config: AppConfig) -> None:
    loaded = load_audio(fixtures_dir / "silence.wav", app_config)
    assert loaded.quality.peak_amplitude < app_config.audio.silence_peak_threshold
    assert any("Near-silent" in w for w in loaded.quality.warnings)
    # Must not crash; samples remain finite zeros.
    assert np.isfinite(loaded.samples).all()


def test_load_stereo(fixtures_dir: Path, app_config: AppConfig) -> None:
    loaded = load_audio(fixtures_dir / "stereo_correlated.wav", app_config)
    assert loaded.samples.ndim == 2
    assert loaded.samples.shape[1] == 2
    assert loaded.source.channels == 2


def test_to_float32_integer_pcm() -> None:
    pcm = np.array([0, 16384, -16384], dtype=np.int16)
    out = to_float32(pcm)
    assert out.dtype == np.float32
    assert abs(out[1] - 0.5) < 0.01


def test_analysis_sample_rate_override(fixtures_dir: Path, app_config: AppConfig) -> None:
    cfg = app_config.model_copy(
        update={"audio": app_config.audio.model_copy(update={"analysis_sample_rate": 22050})}
    )
    loaded = load_audio(fixtures_dir / "sine_440hz.wav", cfg)
    assert loaded.native_sample_rate == 44100
    assert loaded.analysis_sample_rate == 22050
    expected_len = int(round(1.0 * 22050))
    assert abs(loaded.samples.shape[0] - expected_len) <= 2


def test_probe_result_json_roundtrip(fixtures_dir: Path, app_config: AppConfig) -> None:
    result = probe_audio(fixtures_dir / "additive_harmonics.wav", app_config)
    payload = result.to_printable_dict()
    restored = ProbeResult.model_validate(payload)
    assert restored.source.content_hash == result.source.content_hash


def test_mp3_probe_requires_ffmpeg(tmp_path: Path, app_config: AppConfig) -> None:
    """Without real MP3 bytes, extension check still routes to ffmpeg backend."""
    fake = tmp_path / "fake.mp3"
    fake.write_bytes(b"not-a-real-mp3")
    from neuroacoustic.audio.probe import ffprobe_available

    if ffprobe_available():
        with pytest.raises((AudioValidationError, DependencyError)):
            probe_audio(fake, app_config)
    else:
        with pytest.raises(DependencyError, match="ffprobe"):
            probe_audio(fake, app_config)


def _encode_mp3(src_wav: Path, dest_mp3: Path) -> None:
    import subprocess

    from neuroacoustic.audio.probe import ffmpeg_available

    if not ffmpeg_available():
        pytest.skip("ffmpeg not available")
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src_wav),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(dest_mp3),
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg MP3 encode failed: {completed.stderr.strip()}")


def test_mp3_probe_and_load_when_ffmpeg_available(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    """MP3 probe/decode must work when FFmpeg is installed (Milestone 1)."""
    from neuroacoustic.audio.probe import ffmpeg_available, ffprobe_available

    if not (ffmpeg_available() and ffprobe_available()):
        pytest.skip("ffmpeg/ffprobe not available")

    mp3_path = tmp_path / "sine_440hz.mp3"
    _encode_mp3(fixtures_dir / "sine_440hz.wav", mp3_path)

    probed = probe_audio(mp3_path, app_config)
    assert probed.source.probe_backend.value == "ffprobe"
    assert probed.source.native_sample_rate == 44100
    assert probed.source.channels == 1
    assert probed.source.duration_seconds is not None
    assert probed.source.duration_seconds > 0.5
    assert probed.source.codec is not None

    loaded = load_audio(mp3_path, app_config)
    assert loaded.decode_backend == "ffmpeg"
    assert loaded.samples.ndim == 1
    assert loaded.samples.size > 0
    assert np.isfinite(loaded.samples).all()
    assert loaded.quality.peak_amplitude > 0.05
    # Source file must remain untouched.
    assert mp3_path.exists()


def test_load_does_not_modify_source(fixtures_dir: Path, app_config: AppConfig) -> None:
    path = fixtures_dir / "white_noise.wav"
    before = path.read_bytes()
    load_audio(path, app_config)
    assert path.read_bytes() == before
