"""End-to-end analysis pipeline (Milestone 4B)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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
from neuroacoustic.persistence.config_hash import compute_config_hash
from neuroacoustic.persistence.database import assert_db_usable, init_db, session_scope
from neuroacoustic.persistence.repository import (
    PersistDisposition,
    fingerprint_from_analysis,
    get_completed_analysis,
    persist_completed_analysis,
    record_failed_analysis,
    update_analysis_artifact_paths,
    upsert_track,
)
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
    disposition: PersistDisposition | None = None
    analysis_id: int | None = None
    config_hash: str | None = None
    reused: bool = False
    warnings: list[str] = field(default_factory=list)


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


def _write_fingerprint_json(fingerprint: AcousticFingerprint, fp_path: Path) -> AcousticFingerprint:
    payload = fingerprint.to_json_dict()
    _assert_json_finite(payload)
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fingerprint.artifacts.fingerprint_json = str(fp_path.resolve())
    payload = fingerprint.to_json_dict()
    _assert_json_finite(payload)
    fp_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return fingerprint


def _artifact_base(out_dir: Path, loaded: LoadedAudio) -> Path:
    stem = _stem_safe(loaded.source.filename)
    content_short = loaded.source.content_hash[:12]
    return out_dir / f"{stem}_{content_short}"


def _path_exists(value: str | None) -> bool:
    return bool(value) and Path(value).exists()


def _ensure_presentation_artifacts(
    *,
    fingerprint: AcousticFingerprint,
    loaded: LoadedAudio,
    config: AppConfig,
    out_dir: Path,
    plots: bool,
    cached_artifact_paths: dict[str, str | None] | None = None,
) -> tuple[AcousticFingerprint, Path, list[str]]:
    """Materialize requested presentation artifacts for a reused fingerprint.

    Scientific identity is unchanged. Plots / JSON paths are presentation-only
    (excluded from ``config_hash``). Missing requested plots are regenerated
    from the loaded audio when practical; otherwise a clear warning is emitted.
    Never claims plots were created when they were not.
    """
    warnings: list[str] = []
    cached_artifact_paths = cached_artifact_paths or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _artifact_base(out_dir, loaded)
    target_fp = Path(f"{base}_fingerprint.json").resolve()

    existing_fp_s = cached_artifact_paths.get("fingerprint_json")
    existing_fp = Path(existing_fp_s) if existing_fp_s else None
    if existing_fp is not None and existing_fp.exists():
        if existing_fp.resolve() != target_fp:
            warnings.append(
                f"cached_fingerprint_json_at_original_location:{existing_fp}"
            )
            fingerprint = _write_fingerprint_json(fingerprint, target_fp)
            fp_path = target_fp
            warnings.append(f"fingerprint_json_copied_to:{target_fp}")
        else:
            fingerprint.artifacts.fingerprint_json = str(existing_fp.resolve())
            fp_path = existing_fp.resolve()
    else:
        fingerprint = _write_fingerprint_json(fingerprint, target_fp)
        fp_path = target_fp
        if existing_fp_s:
            warnings.append("cached_fingerprint_json_missing_regenerated")

    arts = fingerprint.artifacts
    arts.waveform_png = arts.waveform_png or cached_artifact_paths.get("waveform_png")
    arts.fft_png = arts.fft_png or cached_artifact_paths.get("fft_png")
    arts.fft_log_png = arts.fft_log_png or cached_artifact_paths.get("fft_log_png")
    arts.spectrogram_png = arts.spectrogram_png or cached_artifact_paths.get(
        "spectrogram_png"
    )

    if not plots:
        for label, path in (
            ("waveform_png", arts.waveform_png),
            ("fft_png", arts.fft_png),
            ("fft_log_png", arts.fft_log_png),
            ("spectrogram_png", arts.spectrogram_png),
        ):
            if path and not Path(path).exists():
                warnings.append(f"cached_artifact_missing:{label}")
        fingerprint.artifacts = arts
        return fingerprint, fp_path, warnings

    need_waveform = not _path_exists(arts.waveform_png)
    need_fft = not _path_exists(arts.fft_png)
    need_fft_log = not _path_exists(arts.fft_log_png)
    need_spec = not _path_exists(arts.spectrogram_png)

    for label, path in (
        ("waveform_png", arts.waveform_png),
        ("fft_png", arts.fft_png),
        ("fft_log_png", arts.fft_log_png),
        ("spectrogram_png", arts.spectrogram_png),
    ):
        if path and Path(path).exists():
            try:
                if Path(path).resolve().parent != out_dir.resolve():
                    warnings.append(f"cached_plot_at_original_location:{label}={path}")
            except OSError:
                pass

    if need_waveform or need_fft or need_fft_log or need_spec:
        mono = to_mono(loaded.samples)
        sr = loaded.analysis_sample_rate
        try:
            if need_waveform:
                arts.waveform_png = str(
                    plot_waveform(
                        loaded.samples,
                        sr,
                        Path(f"{base}_waveform.png"),
                        title=f"Waveform — {loaded.source.filename}",
                    ).resolve()
                )
                warnings.append("plot_regenerated:waveform_png")
            if need_fft or need_fft_log:
                fft = compute_fft_summary(mono, sr, config.analysis)
                if need_fft:
                    arts.fft_png = str(
                        plot_fft_magnitude(
                            fft,
                            Path(f"{base}_fft.png"),
                            title=f"FFT magnitude — {loaded.source.filename}",
                            log_frequency=False,
                        ).resolve()
                    )
                    warnings.append("plot_regenerated:fft_png")
                if need_fft_log:
                    arts.fft_log_png = str(
                        plot_fft_magnitude(
                            fft,
                            Path(f"{base}_fft_log.png"),
                            title=f"FFT magnitude (log-frequency) — {loaded.source.filename}",
                            log_frequency=True,
                        ).resolve()
                    )
                    warnings.append("plot_regenerated:fft_log_png")
            if need_spec:
                stft = compute_stft(mono, sr, config.analysis)
                arts.spectrogram_png = str(
                    plot_spectrogram(
                        stft,
                        Path(f"{base}_spectrogram.png"),
                        title=f"Spectrogram — {loaded.source.filename}",
                        amplitude_floor=config.analysis.amplitude_floor,
                    ).resolve()
                )
                warnings.append("plot_regenerated:spectrogram_png")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"plot_regeneration_failed:{exc}")
            logger.warning("Plot regeneration on cache reuse failed: %s", exc)

    for label, path in (
        ("waveform_png", arts.waveform_png),
        ("fft_png", arts.fft_png),
        ("fft_log_png", arts.fft_log_png),
        ("spectrogram_png", arts.spectrogram_png),
    ):
        if not _path_exists(path):
            warnings.append(f"requested_plot_unavailable:{label}")

    fingerprint.artifacts = arts
    fingerprint.artifacts.fingerprint_json = str(fp_path)
    return fingerprint, fp_path, warnings


def _run_analysis(
    loaded: LoadedAudio,
    config: AppConfig,
    *,
    out_dir: Path,
    plots: bool,
) -> tuple[AcousticFingerprint, Path]:
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

    base = _artifact_base(out_dir, loaded)
    artifacts = ArtifactPaths()
    if plots:
        artifacts.waveform_png = str(
            plot_waveform(
                loaded.samples,
                sr,
                Path(f"{base}_waveform.png"),
                title=f"Waveform — {loaded.source.filename}",
            ).resolve()
        )
        artifacts.fft_png = str(
            plot_fft_magnitude(
                fft,
                Path(f"{base}_fft.png"),
                title=f"FFT magnitude — {loaded.source.filename}",
                log_frequency=False,
            ).resolve()
        )
        artifacts.fft_log_png = str(
            plot_fft_magnitude(
                fft,
                Path(f"{base}_fft_log.png"),
                title=f"FFT magnitude (log-frequency) — {loaded.source.filename}",
                log_frequency=True,
            ).resolve()
        )
        artifacts.spectrogram_png = str(
            plot_spectrogram(
                stft,
                Path(f"{base}_spectrogram.png"),
                title=f"Spectrogram — {loaded.source.filename}",
                amplitude_floor=config.analysis.amplitude_floor,
            ).resolve()
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
    fp_path = Path(f"{base}_fingerprint.json").resolve()
    fingerprint = _write_fingerprint_json(fingerprint, fp_path)
    logger.info("Wrote fingerprint %s", fp_path)
    return fingerprint, fp_path


def run_pipeline(
    path: Path | str,
    config: AppConfig,
    *,
    output_dir: Path | None = None,
    plots: bool = True,
    force: bool = False,
    database: Path | str | None = None,
) -> PipelineResult:
    """Load audio, analyze, optionally persist to SQLite.

    Persistence
    -----------
    When ``database`` is set, the pipeline looks up a **completed** analysis
    with identity ``(content_hash, analysis_version, config_hash)``. On hit
    and ``force=False``, the stored fingerprint is reused (disposition
    ``reused``). Failed / incomplete rows are never reused.

    ``force=True``:
    - Allows files longer than ``max_duration_seconds``.
    - Bypasses cache reuse and re-runs analysis.
    - **Only after a successful re-analysis** replaces the existing completed
      row for the same identity in place (same primary key; disposition
      ``forced``). The previous completed fingerprint remains intact if the
      forced run fails — ``record_failed_analysis`` refuses to overwrite a
      completed row. No second completed row is created for the identity.

    Presentation artifacts (plots / output directory) are independent of the
    scientific cache identity. On reuse, missing requested plots may be
    regenerated without creating a new analysis identity.

    A failure never leaves a row marked ``completed``.
    """
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

    config_hash = compute_config_hash(
        config, analysis_sample_rate=loaded.analysis_sample_rate
    )
    analysis_version = config.project.analysis_version
    pipeline_warnings: list[str] = []

    if database is not None:
        db_path = Path(database)
        init_db(db_path)
        engine = assert_db_usable(db_path)

        if not force:
            cached_id: int | None = None
            cached_arts: dict[str, str | None] = {}
            fp: AcousticFingerprint | None = None
            with Session(engine) as session:
                cached = get_completed_analysis(
                    session,
                    content_hash=loaded.source.content_hash,
                    analysis_version=analysis_version,
                    config_hash=config_hash,
                )
                if cached is not None:
                    fp = fingerprint_from_analysis(cached)
                    cached_arts = {
                        "fingerprint_json": cached.artifact_fingerprint_json,
                        "waveform_png": cached.artifact_waveform_png,
                        "fft_png": cached.artifact_fft_png,
                        "fft_log_png": cached.artifact_fft_log_png,
                        "spectrogram_png": cached.artifact_spectrogram_png,
                    }
                    upsert_track(
                        session,
                        content_hash=loaded.source.content_hash,
                        source_path=loaded.source.path,
                        filename=loaded.source.filename,
                        file_size_bytes=loaded.source.file_size_bytes,
                        duration_seconds=loaded.source.duration_seconds,
                        native_sample_rate=loaded.source.native_sample_rate,
                        channels=loaded.source.channels,
                        codec=loaded.source.codec,
                        container=loaded.source.container,
                        update_path=False,
                    )
                    session.commit()
                    cached_id = cached.id

            if cached_id is not None and fp is not None:
                fp, fp_path, art_warnings = _ensure_presentation_artifacts(
                    fingerprint=fp,
                    loaded=loaded,
                    config=config,
                    out_dir=out_dir,
                    plots=plots,
                    cached_artifact_paths=cached_arts,
                )
                pipeline_warnings.extend(art_warnings)
                arts = fp.artifacts
                with session_scope(engine) as session:
                    update_analysis_artifact_paths(
                        session,
                        cached_id,
                        fingerprint_json=arts.fingerprint_json,
                        waveform_png=arts.waveform_png,
                        fft_png=arts.fft_png,
                        fft_log_png=arts.fft_log_png,
                        spectrogram_png=arts.spectrogram_png,
                    )
                logger.info(
                    "Reusing completed analysis id=%s config_hash=%s…",
                    cached_id,
                    config_hash[:12],
                )
                return PipelineResult(
                    fingerprint=fp,
                    fingerprint_path=fp_path,
                    loaded=loaded,
                    disposition="reused",
                    analysis_id=cached_id,
                    config_hash=config_hash,
                    reused=True,
                    warnings=pipeline_warnings,
                )

        try:
            fingerprint, fp_path = _run_analysis(
                loaded, config, out_dir=out_dir, plots=plots
            )
        except Exception as exc:
            # Never overwrite a completed row with a failed forced attempt.
            try:
                with session_scope(engine) as session:
                    record_failed_analysis(
                        session,
                        content_hash=loaded.source.content_hash,
                        analysis_version=analysis_version,
                        schema_version=config.project.schema_version,
                        vector_version=config.analysis.vector_version,
                        config_hash=config_hash,
                        error_message=str(exc),
                        source_path=loaded.source.path,
                        filename=loaded.source.filename,
                        file_size_bytes=loaded.source.file_size_bytes,
                        duration_seconds=loaded.source.duration_seconds,
                        native_sample_rate=loaded.source.native_sample_rate,
                        channels=loaded.source.channels,
                        codec=loaded.source.codec,
                        container=loaded.source.container,
                    )
            except Exception as db_exc:  # noqa: BLE001
                logger.warning("Could not persist failed analysis: %s", db_exc)
            raise

        AcousticFingerprint.model_validate(fingerprint.to_json_dict())
        _assert_json_finite(fingerprint.to_json_dict())
        if fingerprint.vector_meta is not None:
            assert len(fingerprint.vector) == len(fingerprint.vector_meta.labels)

        try:
            with session_scope(engine) as session:
                row = persist_completed_analysis(
                    session,
                    fingerprint=fingerprint,
                    config_hash=config_hash,
                    fingerprint_path=fp_path,
                    replace_existing=force,
                )
                analysis_id = row.id
        except IntegrityError:
            # Concurrent insert won — only reuse a *completed* winner.
            with Session(engine) as session:
                cached = get_completed_analysis(
                    session,
                    content_hash=loaded.source.content_hash,
                    analysis_version=analysis_version,
                    config_hash=config_hash,
                )
                if cached is None:
                    raise
                fingerprint = fingerprint_from_analysis(cached)
                analysis_id = cached.id
                return PipelineResult(
                    fingerprint=fingerprint,
                    fingerprint_path=fp_path,
                    loaded=loaded,
                    disposition="reused",
                    analysis_id=analysis_id,
                    config_hash=config_hash,
                    reused=True,
                    warnings=pipeline_warnings
                    + ["concurrent_insert_reused_completed_row"],
                )

        disposition: PersistDisposition = "forced" if force else "analyzed"
        return PipelineResult(
            fingerprint=fingerprint,
            fingerprint_path=fp_path,
            loaded=loaded,
            disposition=disposition,
            analysis_id=analysis_id,
            config_hash=config_hash,
            reused=False,
            warnings=pipeline_warnings,
        )

    fingerprint, fp_path = _run_analysis(loaded, config, out_dir=out_dir, plots=plots)
    return PipelineResult(
        fingerprint=fingerprint,
        fingerprint_path=fp_path,
        loaded=loaded,
        disposition=None,
        analysis_id=None,
        config_hash=config_hash,
        reused=False,
        warnings=pipeline_warnings,
    )
