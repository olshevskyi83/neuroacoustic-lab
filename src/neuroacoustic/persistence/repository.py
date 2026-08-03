"""Repository helpers for tracks and analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from neuroacoustic.exceptions import AnalysisNotFoundError
from neuroacoustic.fingerprint.models import AcousticFingerprint
from neuroacoustic.logging import get_logger
from neuroacoustic.persistence.models import Analysis, Track

logger = get_logger(__name__)

AnalysisStatus = Literal["completed", "failed", "in_progress"]
PersistDisposition = Literal["analyzed", "reused", "forced"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _abs_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


@dataclass(slots=True)
class DbStats:
    database_path: str
    database_size_bytes: int | None
    total_analyses: int
    completed: int
    failed: int
    in_progress: int
    unique_content_hashes: int
    schema_versions: dict[str, int]
    analysis_versions: dict[str, int]


def get_completed_analysis(
    session: Session,
    *,
    content_hash: str,
    analysis_version: str,
    config_hash: str,
) -> Analysis | None:
    """Return a completed analysis for the cache identity, else None.

    Failed / in_progress rows are never returned for reuse.
    """
    stmt = (
        select(Analysis)
        .options(joinedload(Analysis.track))
        .where(
            Analysis.content_hash == content_hash,
            Analysis.analysis_version == analysis_version,
            Analysis.config_hash == config_hash,
            Analysis.status == "completed",
            Analysis.fingerprint_json.is_not(None),
        )
    )
    return session.scalar(stmt)


def upsert_track(
    session: Session,
    *,
    content_hash: str,
    source_path: str | None,
    filename: str | None,
    file_size_bytes: int | None,
    duration_seconds: float | None,
    native_sample_rate: int | None,
    channels: int | None,
    codec: str | None,
    container: str | None,
    update_path: bool = False,
) -> Track:
    """Insert or refresh track metadata for a content hash.

    Path / filename semantics (MVP)
    -------------------------------
    **First-seen** ``source_path`` and ``filename`` are retained by default.
    Re-analyzing identical bytes under another name updates ``updated_at``
    (last-seen activity) but does **not** overwrite the canonical first-seen
    path unless ``update_path=True``. Only one path is stored in this MVP.
    """
    track = session.scalar(select(Track).where(Track.content_hash == content_hash))
    now = _utc_now()
    if track is None:
        track = Track(
            content_hash=content_hash,
            source_path=source_path,
            filename=filename,
            file_size_bytes=file_size_bytes,
            duration_seconds=duration_seconds,
            native_sample_rate=native_sample_rate,
            channels=channels,
            codec=codec,
            container=container,
            created_at=now,
            updated_at=now,
        )
        try:
            with session.begin_nested():
                session.add(track)
                session.flush()
        except IntegrityError:
            # Concurrent insert of the same content hash
            track = session.scalar(
                select(Track).where(Track.content_hash == content_hash)
            )
            if track is None:
                raise
            track.updated_at = now
            if update_path:
                track.source_path = source_path
                track.filename = filename
            session.flush()
            return track
    else:
        # Always refresh technical identity fields; path is first-seen by default.
        if update_path:
            track.source_path = source_path
            track.filename = filename
        if track.file_size_bytes is None and file_size_bytes is not None:
            track.file_size_bytes = file_size_bytes
        if track.duration_seconds is None and duration_seconds is not None:
            track.duration_seconds = duration_seconds
        if track.native_sample_rate is None and native_sample_rate is not None:
            track.native_sample_rate = native_sample_rate
        if track.channels is None and channels is not None:
            track.channels = channels
        if track.codec is None and codec is not None:
            track.codec = codec
        if track.container is None and container is not None:
            track.container = container
        track.updated_at = now
        session.flush()
    return track


def update_analysis_artifact_paths(
    session: Session,
    analysis_id: int,
    *,
    fingerprint_json: str | None = None,
    waveform_png: str | None = None,
    fft_png: str | None = None,
    fft_log_png: str | None = None,
    spectrogram_png: str | None = None,
) -> Analysis | None:
    """Update presentation artifact paths on an existing analysis (no identity change)."""
    row = session.get(Analysis, analysis_id)
    if row is None:
        return None
    if fingerprint_json is not None:
        row.artifact_fingerprint_json = _abs_path(fingerprint_json)
    if waveform_png is not None:
        row.artifact_waveform_png = _abs_path(waveform_png)
    if fft_png is not None:
        row.artifact_fft_png = _abs_path(fft_png)
    if fft_log_png is not None:
        row.artifact_fft_log_png = _abs_path(fft_log_png)
    if spectrogram_png is not None:
        row.artifact_spectrogram_png = _abs_path(spectrogram_png)
    row.updated_at = _utc_now()
    session.flush()
    return row


def compute_stats(session: Session, database_path: Path | str) -> DbStats:
    path = Path(database_path).expanduser().resolve()
    # Include WAL/SHM sidecars so reported size matches on-disk footprint in WAL mode.
    size = 0
    present = False
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            present = True
            size += candidate.stat().st_size
    size_out = size if present else None
    total = session.scalar(select(func.count()).select_from(Analysis)) or 0
    completed = (
        session.scalar(
            select(func.count()).select_from(Analysis).where(Analysis.status == "completed")
        )
        or 0
    )
    failed = (
        session.scalar(
            select(func.count()).select_from(Analysis).where(Analysis.status == "failed")
        )
        or 0
    )
    in_progress = (
        session.scalar(
            select(func.count())
            .select_from(Analysis)
            .where(Analysis.status == "in_progress")
        )
        or 0
    )
    unique_hashes = (
        session.scalar(select(func.count()).select_from(Track)) or 0
    )

    schema_versions: dict[str, int] = {}
    for ver, cnt in session.execute(
        select(Analysis.schema_version, func.count()).group_by(Analysis.schema_version)
    ):
        schema_versions[str(ver)] = int(cnt)

    analysis_versions: dict[str, int] = {}
    for ver, cnt in session.execute(
        select(Analysis.analysis_version, func.count()).group_by(Analysis.analysis_version)
    ):
        analysis_versions[str(ver)] = int(cnt)

    return DbStats(
        database_path=str(path),
        database_size_bytes=size_out,
        total_analyses=int(total),
        completed=int(completed),
        failed=int(failed),
        in_progress=int(in_progress),
        unique_content_hashes=int(unique_hashes),
        schema_versions=schema_versions,
        analysis_versions=analysis_versions,
    )


def _scalar_fields(fp: AcousticFingerprint) -> dict[str, Any]:
    return {
        "duration_seconds": fp.source.duration_seconds,
        "analysis_sample_rate": fp.source.analysis_sample_rate,
        "rms": fp.energy.rms,
        "peak_amplitude": fp.energy.peak_amplitude,
        "centroid_mean_hz": fp.spectral.centroid_hz.mean,
        "flatness_mean": fp.spectral.flatness.mean,
        "entropy_mean": fp.spectral.entropy.mean,
        "f0_median_hz": fp.pitch.f0_median_hz,
        "voiced_ratio": fp.pitch.voiced_ratio,
        "tempo_bpm": fp.rhythm.tempo_bpm,
        "stereo_width_estimate": fp.stereo.stereo_width_estimate,
        "attack_time_s": fp.envelope.attack_time_s,
        "tail_decay_t60_s": fp.reverberation.tail_decay_t60_estimate_seconds,
    }


def persist_completed_analysis(
    session: Session,
    *,
    fingerprint: AcousticFingerprint,
    config_hash: str,
    fingerprint_path: Path | str,
    replace_existing: bool = False,
) -> Analysis:
    """Insert or replace a completed analysis row transactionally.

    Uniqueness is enforced by ``uq_analysis_identity``. On race
    ``IntegrityError``, the session must be rolled back by the caller and
    the completed row re-fetched.

    Parameters
    ----------
    replace_existing:
        When True (``--force``), update the existing row for the same
        identity in place. When False, insert a new row (fails if completed
        identity already exists — caller should have checked cache).
    """
    src = fingerprint.source
    track = upsert_track(
        session,
        content_hash=src.content_hash,
        source_path=src.path,
        filename=src.filename,
        file_size_bytes=src.file_size_bytes,
        duration_seconds=src.duration_seconds,
        native_sample_rate=src.native_sample_rate,
        channels=src.channels,
        codec=src.codec,
        container=src.container,
    )

    payload = fingerprint.to_json_dict()
    fingerprint_json = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    vector_json = json.dumps(list(fingerprint.vector), ensure_ascii=True, allow_nan=False)
    arts = fingerprint.artifacts
    now = _utc_now()
    scalars = _scalar_fields(fingerprint)

    existing = session.scalar(
        select(Analysis).where(
            Analysis.content_hash == src.content_hash,
            Analysis.analysis_version == fingerprint.analysis_version,
            Analysis.config_hash == config_hash,
        )
    )

    if existing is not None:
        if existing.status == "completed" and not replace_existing:
            return existing
        # Replace failed / in_progress, or force-replace completed
        existing.track_id = track.id
        existing.schema_version = fingerprint.schema_version
        existing.vector_version = (
            fingerprint.vector_meta.version if fingerprint.vector_meta else None
        )
        existing.status = "completed"
        existing.error_message = None
        existing.fingerprint_json = fingerprint_json
        existing.vector_json = vector_json
        existing.artifact_fingerprint_json = _abs_path(str(fingerprint_path))
        existing.artifact_waveform_png = _abs_path(arts.waveform_png)
        existing.artifact_fft_png = _abs_path(arts.fft_png)
        existing.artifact_fft_log_png = _abs_path(arts.fft_log_png)
        existing.artifact_spectrogram_png = _abs_path(arts.spectrogram_png)
        for key, value in scalars.items():
            setattr(existing, key, value)
        existing.updated_at = now
        session.flush()
        return existing

    row = Analysis(
        track_id=track.id,
        content_hash=src.content_hash,
        schema_version=fingerprint.schema_version,
        analysis_version=fingerprint.analysis_version,
        vector_version=fingerprint.vector_meta.version if fingerprint.vector_meta else None,
        config_hash=config_hash,
        status="completed",
        error_message=None,
        fingerprint_json=fingerprint_json,
        vector_json=vector_json,
        artifact_fingerprint_json=_abs_path(str(fingerprint_path)),
        artifact_waveform_png=_abs_path(arts.waveform_png),
        artifact_fft_png=_abs_path(arts.fft_png),
        artifact_fft_log_png=_abs_path(arts.fft_log_png),
        artifact_spectrogram_png=_abs_path(arts.spectrogram_png),
        created_at=now,
        updated_at=now,
        **scalars,
    )
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError:
        # Concurrent insert won the unique constraint race.
        winner = get_completed_analysis(
            session,
            content_hash=src.content_hash,
            analysis_version=fingerprint.analysis_version,
            config_hash=config_hash,
        )
        if winner is None:
            # Winner not completed (failed/in_progress) — do not reuse.
            raise
        return winner
    return row


def record_failed_analysis(
    session: Session,
    *,
    content_hash: str,
    analysis_version: str,
    schema_version: str,
    vector_version: str | None,
    config_hash: str,
    error_message: str,
    source_path: str | None = None,
    filename: str | None = None,
    file_size_bytes: int | None = None,
    duration_seconds: float | None = None,
    native_sample_rate: int | None = None,
    channels: int | None = None,
    codec: str | None = None,
    container: str | None = None,
) -> Analysis:
    """Upsert a failed analysis row (never status=completed)."""
    track = upsert_track(
        session,
        content_hash=content_hash,
        source_path=source_path,
        filename=filename,
        file_size_bytes=file_size_bytes,
        duration_seconds=duration_seconds,
        native_sample_rate=native_sample_rate,
        channels=channels,
        codec=codec,
        container=container,
    )
    now = _utc_now()
    existing = session.scalar(
        select(Analysis).where(
            Analysis.content_hash == content_hash,
            Analysis.analysis_version == analysis_version,
            Analysis.config_hash == config_hash,
        )
    )
    msg = error_message[:4000]
    if existing is not None:
        # Do not overwrite a completed success with a later failure unless forced —
        # caller should only invoke this when analyzing; if completed exists and
        # force wasn't used we wouldn't be here. On force failure after wipe risk:
        # mark failed only if not completed, or always mark failed when recording.
        if existing.status == "completed":
            # Leave completed intact; attach note via log only
            logger.warning(
                "Skipping failed-record overwrite of completed analysis id=%s",
                existing.id,
            )
            return existing
        existing.status = "failed"
        existing.error_message = msg
        existing.fingerprint_json = None
        existing.vector_json = None
        existing.updated_at = now
        session.flush()
        return existing

    row = Analysis(
        track_id=track.id,
        content_hash=content_hash,
        schema_version=schema_version,
        analysis_version=analysis_version,
        vector_version=vector_version,
        config_hash=config_hash,
        status="failed",
        error_message=msg,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.flush()
    return row


def list_analyses(
    session: Session,
    *,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
) -> list[Analysis]:
    stmt = (
        select(Analysis)
        .options(joinedload(Analysis.track))
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
    )
    if status:
        stmt = stmt.where(Analysis.status == status)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.scalars(stmt).unique())


def get_analysis_by_id(session: Session, analysis_id: int) -> Analysis:
    row = session.scalar(
        select(Analysis)
        .options(joinedload(Analysis.track))
        .where(Analysis.id == analysis_id)
    )
    if row is None:
        raise AnalysisNotFoundError(f"Analysis id {analysis_id} not found")
    return row


def resolve_analysis_or_track_id(session: Session, raw_id: str) -> Analysis:
    """Resolve an analysis id, or the newest analysis for a track id."""
    try:
        numeric = int(raw_id)
    except ValueError as exc:
        raise AnalysisNotFoundError(f"Invalid id {raw_id!r}; expected integer") from exc

    row = session.scalar(
        select(Analysis)
        .options(joinedload(Analysis.track))
        .where(Analysis.id == numeric)
    )
    if row is not None:
        return row

    track = session.get(Track, numeric)
    if track is None:
        raise AnalysisNotFoundError(f"No analysis or track with id {numeric}")
    newest = session.scalar(
        select(Analysis)
        .options(joinedload(Analysis.track))
        .where(Analysis.track_id == track.id)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(1)
    )
    if newest is None:
        raise AnalysisNotFoundError(f"Track {numeric} has no analyses")
    return newest


def fingerprint_from_analysis(row: Analysis) -> AcousticFingerprint:
    """Deserialize and Pydantic-validate stored fingerprint JSON."""
    if not row.fingerprint_json:
        raise AnalysisNotFoundError(
            f"Analysis {row.id} has no fingerprint JSON (status={row.status})"
        )
    data = json.loads(row.fingerprint_json)
    return AcousticFingerprint.model_validate(data)


def analysis_to_summary_dict(row: Analysis) -> dict[str, Any]:
    track = row.track
    artifacts = {
        "fingerprint_json": row.artifact_fingerprint_json,
        "waveform_png": row.artifact_waveform_png,
        "fft_png": row.artifact_fft_png,
        "fft_log_png": row.artifact_fft_log_png,
        "spectrogram_png": row.artifact_spectrogram_png,
    }
    missing = [
        key for key, path in artifacts.items() if path and not Path(path).exists()
    ]
    return {
        "analysis_id": row.id,
        "track_id": row.track_id,
        "status": row.status,
        "content_hash": row.content_hash,
        "filename": track.filename if track else None,
        "source_path": track.source_path if track else None,
        "file_size_bytes": track.file_size_bytes if track else None,
        "duration_seconds": row.duration_seconds,
        "native_sample_rate": track.native_sample_rate if track else None,
        "channels": track.channels if track else None,
        "codec": track.codec if track else None,
        "container": track.container if track else None,
        "schema_version": row.schema_version,
        "analysis_version": row.analysis_version,
        "vector_version": row.vector_version,
        "config_hash": row.config_hash,
        "error_message": row.error_message,
        "rms": row.rms,
        "peak_amplitude": row.peak_amplitude,
        "centroid_mean_hz": row.centroid_mean_hz,
        "flatness_mean": row.flatness_mean,
        "f0_median_hz": row.f0_median_hz,
        "tempo_bpm": row.tempo_bpm,
        "stereo_width_estimate": row.stereo_width_estimate,
        "attack_time_s": row.attack_time_s,
        "tail_decay_t60_s": row.tail_decay_t60_s,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


__all__ = [
    "AnalysisStatus",
    "DbStats",
    "IntegrityError",
    "PersistDisposition",
    "analysis_to_summary_dict",
    "compute_stats",
    "fingerprint_from_analysis",
    "get_analysis_by_id",
    "get_completed_analysis",
    "list_analyses",
    "persist_completed_analysis",
    "record_failed_analysis",
    "resolve_analysis_or_track_id",
    "update_analysis_artifact_paths",
    "upsert_track",
]
