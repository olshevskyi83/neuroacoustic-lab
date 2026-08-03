"""End-to-end analysis pipeline (Milestone 4A)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from neuroacoustic.analysis.energy import analyze_energy
from neuroacoustic.analysis.envelope import compute_envelope_features
from neuroacoustic.analysis.harmonics import analyze_harmonics
from neuroacoustic.analysis.pitch import analyze_pitch
from neuroacoustic.analysis.reverb import compute_reverb_features
from neuroacoustic.analysis.rhythm import compute_rhythm_features
from neuroacoustic.analysis.spectral import analyze_spectral
from neuroacoustic.analysis.spectrum import compute_fft_summary, compute_stft
from neuroacoustic.analysis.stats import to_mono
from neuroacoustic.analysis.stereo import compute_stereo_features
from neuroacoustic.audio.loader import LoadedAudio, load_audio
from neuroacoustic.config import AppConfig
from neuroacoustic.exceptions import AudioValidationError, NeuroAcousticError
from neuroacoustic.fingerprint.builder import build_fingerprint
from neuroacoustic.fingerprint.models import AcousticFingerprint, ArtifactPaths
from neuroacoustic.logging import get_logger
from neuroacoustic.visualization import (
    plot_fft_magnitude,
    plot_spectrogram,
    plot_waveform,
)

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    fingerprint: AcousticFingerprint
    fingerprint_path: Path
    loaded: LoadedAudio


def _stem_safe(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "track"


def _assert_json_finite(payload: object, path: str = "$") -> None:
    """Raise if any NaN / Infinity sneaks into the serializable structure."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            _assert_json_finite(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, value in enumerate(payload):
            _assert_json_finite(value, f"{path}[{i}]")
    elif isinstance(payload, float):
        if payload != payload or payload in (float("inf"), float("-inf")):
            raise NeuroAcousticError(f"Non-finite float at {path}: {payload!r}")


def run_pipeline(
    path: Path | str,
    config: AppConfig,
    *,
    output_dir: Path | None = None,
    plots: bool = True,
    force: bool = False,
) -> PipelineResult:
    """Load audio, run analysis through M4A sections, write JSON/plots."""
    audio_path = Path(path)
    out_dir = Path(output_dir or config.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_audio(audio_path, config)
    duration = loaded.source.duration_seconds
    if (
        duration is not None
        and duration > config.audio.max_duration_seconds
        and not force
    ):
        raise AudioValidationError(
            f"Duration {duration:.3f}s exceeds max "
            f"{config.audio.max_duration_seconds}s; pass --force to override"
        )

    mono = to_mono(loaded.samples)
    sr = loaded.analysis_sample_rate

    fft = compute_fft_summary(mono, sr, config.analysis)
    stft = compute_stft(mono, sr, config.analysis)
    spectral = analyze_spectral(stft, config.analysis)
    energy = analyze_energy(mono, sr, config.analysis)
    pitch = analyze_pitch(mono, sr, config.analysis)
    harmonics = analyze_harmonics(stft, pitch, config.analysis)
    envelope = compute_envelope_features(mono, sr, config.analysis)
    stereo = compute_stereo_features(loaded.samples, sr, config.analysis)
    rhythm = compute_rhythm_features(mono, sr, config.analysis)
    reverb = compute_reverb_features(mono, sr, config.analysis)

    stem = _stem_safe(loaded.source.filename)
    content_short = loaded.source.content_hash[:12]
    base = out_dir / f"{stem}_{content_short}"

    artifacts = ArtifactPaths()
    if plots:
        artifacts.waveform_png = str(
            plot_waveform(
                loaded.samples,
                sr,
                Path(f"{base}_waveform.png"),
                title=f"Waveform — {loaded.source.filename}",
            )
        )
        artifacts.fft_png = str(
            plot_fft_magnitude(
                fft,
                Path(f"{base}_fft.png"),
                title=f"FFT magnitude — {loaded.source.filename}",
                log_frequency=False,
            )
        )
        artifacts.fft_log_png = str(
            plot_fft_magnitude(
                fft,
                Path(f"{base}_fft_log.png"),
                title=f"FFT magnitude (log-frequency) — {loaded.source.filename}",
                log_frequency=True,
            )
        )
        artifacts.spectrogram_png = str(
            plot_spectrogram(
                stft,
                Path(f"{base}_spectrogram.png"),
                title=f"Spectrogram — {loaded.source.filename}",
                amplitude_floor=config.analysis.amplitude_floor,
            )
        )

    fingerprint = build_fingerprint(
        loaded,
        config,
        fft=fft,
        stft=stft,
        spectral=spectral,
        energy=energy,
        pitch=pitch,
        harmonics=harmonics,
        envelope=envelope,
        stereo=stereo,
        rhythm=rhythm,
        reverb=reverb,
        artifacts=artifacts,
    )

    fp_path = Path(f"{base}_fingerprint.json")
    payload = fingerprint.to_json_dict()
    _assert_json_finite(payload)
    fp_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fingerprint.artifacts.fingerprint_json = str(fp_path)
    payload = fingerprint.to_json_dict()
    _assert_json_finite(payload)
    fp_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    logger.info("Wrote fingerprint %s", fp_path)
    return PipelineResult(fingerprint=fingerprint, fingerprint_path=fp_path, loaded=loaded)
