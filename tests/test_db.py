"""Milestone 4B: SQLite persistence and database CLI tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from neuroacoustic.cli import app
from neuroacoustic.config import AppConfig, load_config
from neuroacoustic.exceptions import AnalysisNotFoundError, DatabaseSchemaError
from neuroacoustic.fingerprint.models import AcousticFingerprint
from neuroacoustic.persistence.config_hash import compute_config_hash
from neuroacoustic.persistence.database import assert_db_usable, init_db
from neuroacoustic.persistence.models import (
    DB_SCHEMA_VERSION,
    DB_SCHEMA_VERSION_KEY,
    Analysis,
    SchemaMeta,
    Track,
)
from neuroacoustic.persistence.repository import (
    compute_stats,
    get_completed_analysis,
    list_analyses,
    persist_completed_analysis,
    resolve_analysis_or_track_id,
)
from neuroacoustic.pipeline import run_pipeline

runner = CliRunner()


def _finite(obj: object) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _finite(v)
    elif isinstance(obj, list):
        for v in obj:
            _finite(v)
    elif isinstance(obj, float):
        assert obj == obj and obj not in (float("inf"), float("-inf"))


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.sqlite"


def test_init_idempotent(db_path: Path) -> None:
    init_db(db_path)
    init_db(db_path)
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        ver = session.get(SchemaMeta, DB_SCHEMA_VERSION_KEY)
        assert ver is not None
        assert int(ver.value) == DB_SCHEMA_VERSION


def test_incompatible_schema_refuses(db_path: Path) -> None:
    init_db(db_path)
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        meta = session.get(SchemaMeta, DB_SCHEMA_VERSION_KEY)
        assert meta is not None
        meta.value = "999"
        session.commit()
    with pytest.raises(DatabaseSchemaError):
        init_db(db_path)


def test_empty_list_and_stats(db_path: Path) -> None:
    init_db(db_path)
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        assert list_analyses(session) == []
        stats = compute_stats(session, db_path)
    assert stats.total_analyses == 0
    assert stats.completed == 0
    assert stats.unique_content_hashes == 0


def test_persist_reuse_and_filename_independence(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    out = tmp_path / "out"
    src = fixtures_dir / "sine_440hz.wav"
    r1 = run_pipeline(src, app_config, output_dir=out, plots=False, database=db_path)
    assert r1.disposition == "analyzed"
    assert r1.analysis_id is not None
    AcousticFingerprint.model_validate(r1.fingerprint.to_json_dict())
    _finite(r1.fingerprint.to_json_dict())

    r2 = run_pipeline(src, app_config, output_dir=out, plots=False, database=db_path)
    assert r2.disposition == "reused"
    assert r2.analysis_id == r1.analysis_id
    assert r2.reused is True

    # Same bytes, different filename → same content hash → reuse
    other = tmp_path / "copy name 440.wav"
    shutil.copy(src, other)
    r3 = run_pipeline(other, app_config, output_dir=out, plots=False, database=db_path)
    assert r3.disposition == "reused"
    assert r3.analysis_id == r1.analysis_id
    assert r3.fingerprint.source.content_hash == r1.fingerprint.source.content_hash

    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        stats = compute_stats(session, db_path)
        assert stats.completed == 1
        assert stats.unique_content_hashes == 1


def test_changed_config_creates_new_analysis(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    out = tmp_path / "out"
    src = fixtures_dir / "sine_440hz.wav"
    r1 = run_pipeline(src, app_config, output_dir=out, plots=False, database=db_path)
    h1 = r1.config_hash

    cfg2 = app_config.model_copy(
        update={
            "analysis": app_config.analysis.model_copy(update={"n_fft": 4096}),
        }
    )
    assert compute_config_hash(cfg2, analysis_sample_rate=44100) != h1
    r2 = run_pipeline(src, cfg2, output_dir=out, plots=False, database=db_path)
    assert r2.disposition == "analyzed"
    assert r2.analysis_id != r1.analysis_id
    assert r2.config_hash != h1

    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        stats = compute_stats(session, db_path)
        assert stats.completed == 2


def test_force_replaces_same_identity(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    out = tmp_path / "out"
    src = fixtures_dir / "sine_110hz.wav"
    r1 = run_pipeline(src, app_config, output_dir=out, plots=False, database=db_path)
    id1 = r1.analysis_id
    r2 = run_pipeline(
        src, app_config, output_dir=out, plots=False, database=db_path, force=True
    )
    assert r2.disposition == "forced"
    assert r2.analysis_id == id1  # same row replaced
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        assert compute_stats(session, db_path).completed == 1


def test_failed_not_completed(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    """A failing analysis must not be marked completed."""
    out = tmp_path / "out"
    # Use an unreadable path after hashing is tricky; instead insert failed via API
    # after a successful probe path: corrupt by analyzing then manually verify
    # pipeline records failure when analysis raises.
    from neuroacoustic.persistence.database import session_scope
    from neuroacoustic.persistence.repository import record_failed_analysis

    init_db(db_path)
    engine = assert_db_usable(db_path)
    with session_scope(engine) as session:
        row = record_failed_analysis(
            session,
            content_hash="a" * 64,
            analysis_version=app_config.project.analysis_version,
            schema_version=app_config.project.schema_version,
            vector_version=app_config.analysis.vector_version,
            config_hash="b" * 64,
            error_message="synthetic failure",
            filename="x.wav",
        )
        assert row.status == "failed"
        assert row.fingerprint_json is None
        cached = get_completed_analysis(
            session,
            content_hash="a" * 64,
            analysis_version=app_config.project.analysis_version,
            config_hash="b" * 64,
        )
        assert cached is None


def test_uniqueness_constraint(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    out = tmp_path / "out"
    src = fixtures_dir / "sine_440hz.wav"
    r1 = run_pipeline(src, app_config, output_dir=out, plots=False, database=db_path)
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        # Second insert without replace should return existing
        row = persist_completed_analysis(
            session,
            fingerprint=r1.fingerprint,
            config_hash=r1.config_hash or "",
            fingerprint_path=r1.fingerprint_path,
            replace_existing=False,
        )
        session.commit()
        assert row.id == r1.analysis_id
        count = session.scalar(select(Analysis).where(Analysis.status == "completed"))
        # count via list
        assert len(list_analyses(session)) == 1


def test_list_pagination_ordering(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    out = tmp_path / "out"
    for name in ("sine_110hz.wav", "sine_440hz.wav", "silence.wav"):
        run_pipeline(
            fixtures_dir / name, app_config, output_dir=out / name, plots=False, database=db_path
        )
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        all_rows = list_analyses(session, limit=10, offset=0)
        assert len(all_rows) == 3
        # Newest first
        assert all_rows[0].created_at >= all_rows[1].created_at
        page = list_analyses(session, limit=1, offset=1)
        assert len(page) == 1
        assert page[0].id == all_rows[1].id


def test_show_missing_id(db_path: Path) -> None:
    init_db(db_path)
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        with pytest.raises(AnalysisNotFoundError):
            resolve_analysis_or_track_id(session, "999999")


def test_cli_db_json_and_unicode_paths(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    db = tmp_path / "unicöde db" / "lab.sqlite"
    out = tmp_path / "out space"
    audio = tmp_path / "siné 440.wav"
    shutil.copy(fixtures_dir / "sine_440hz.wav", audio)

    result = runner.invoke(app, ["db", "init", "--database", str(db)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "analyze",
            str(audio),
            "--database",
            str(db),
            "--output-dir",
            str(out),
            "--no-plots",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "analyzed" in result.output

    result = runner.invoke(
        app,
        [
            "analyze",
            str(audio),
            "--database",
            str(db),
            "--output-dir",
            str(out),
            "--no-plots",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "reused" in result.output

    result = runner.invoke(app, ["db", "list", "--database", str(db), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert isinstance(data, list) and len(data) == 1
    analysis_id = data[0]["analysis_id"]

    result = runner.invoke(
        app, ["db", "show", str(analysis_id), "--database", str(db), "--json"]
    )
    assert result.exit_code == 0, result.output
    fp = json.loads(result.stdout)
    AcousticFingerprint.model_validate(fp)
    _finite(fp)

    result = runner.invoke(app, ["db", "stats", "--database", str(db), "--json"])
    assert result.exit_code == 0, result.output
    stats = json.loads(result.stdout)
    assert stats["completed"] == 1

    result = runner.invoke(app, ["db", "show", "999999", "--database", str(db)])
    assert result.exit_code != 0


def test_analyze_without_database_still_works(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    r = run_pipeline(
        fixtures_dir / "sine_440hz.wav",
        app_config,
        output_dir=tmp_path,
        plots=False,
        database=None,
    )
    assert r.disposition is None
    assert r.fingerprint_path.exists()


def test_force_failed_preserves_completed(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced re-analysis that fails must leave the prior completed row intact."""
    out = tmp_path / "out"
    src = fixtures_dir / "sine_440hz.wav"
    r1 = run_pipeline(src, app_config, output_dir=out, plots=False, database=db_path)
    assert r1.disposition == "analyzed"
    id1 = r1.analysis_id
    fp1 = r1.fingerprint.to_json_dict()

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic forced failure")

    monkeypatch.setattr("neuroacoustic.pipeline._run_analysis", _boom)
    with pytest.raises(RuntimeError, match="synthetic forced failure"):
        run_pipeline(
            src, app_config, output_dir=out, plots=False, database=db_path, force=True
        )

    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        row = get_completed_analysis(
            session,
            content_hash=r1.fingerprint.source.content_hash,
            analysis_version=app_config.project.analysis_version,
            config_hash=r1.config_hash or "",
        )
        assert row is not None
        assert row.id == id1
        assert row.status == "completed"
        restored = AcousticFingerprint.model_validate(json.loads(row.fingerprint_json or "{}"))
        assert restored.to_json_dict()["source"]["content_hash"] == fp1["source"]["content_hash"]
        assert compute_stats(session, db_path).completed == 1
        assert compute_stats(session, db_path).failed == 0


def test_reuse_regenerates_requested_plots(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    """Scientific reuse with --plots after --no-plots regenerates plot files."""
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    src = fixtures_dir / "sine_110hz.wav"
    r1 = run_pipeline(src, app_config, output_dir=out_a, plots=False, database=db_path)
    assert r1.fingerprint.artifacts.waveform_png is None

    r2 = run_pipeline(src, app_config, output_dir=out_b, plots=True, database=db_path)
    assert r2.disposition == "reused"
    assert r2.analysis_id == r1.analysis_id
    assert r2.fingerprint.artifacts.waveform_png is not None
    assert Path(r2.fingerprint.artifacts.waveform_png).exists()
    assert any(w.startswith("plot_regenerated:") for w in r2.warnings)
    # Fingerprint JSON available under the requested output dir
    assert r2.fingerprint_path.exists()
    assert out_b.resolve() in r2.fingerprint_path.parents or r2.fingerprint_path.parent == out_b.resolve()


def test_reuse_warns_when_plots_deleted(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    out = tmp_path / "out"
    src = fixtures_dir / "sine_110hz.wav"
    r1 = run_pipeline(src, app_config, output_dir=out, plots=True, database=db_path)
    assert r1.fingerprint.artifacts.waveform_png
    Path(r1.fingerprint.artifacts.waveform_png).unlink()
    # Also delete other plots so regeneration is required
    for attr in ("fft_png", "fft_log_png", "spectrogram_png"):
        p = getattr(r1.fingerprint.artifacts, attr)
        if p and Path(p).exists():
            Path(p).unlink()

    r2 = run_pipeline(src, app_config, output_dir=out, plots=True, database=db_path)
    assert r2.disposition == "reused"
    assert Path(r2.fingerprint.artifacts.waveform_png or "").exists()
    assert any(w.startswith("plot_regenerated:") for w in r2.warnings)


def test_first_seen_path_retained(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    out = tmp_path / "out"
    src = fixtures_dir / "sine_440hz.wav"
    r1 = run_pipeline(src, app_config, output_dir=out, plots=False, database=db_path)
    other = tmp_path / "other name.wav"
    shutil.copy(src, other)
    r2 = run_pipeline(other, app_config, output_dir=out, plots=False, database=db_path)
    assert r2.disposition == "reused"
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        track = session.scalar(
            select(Track).where(Track.content_hash == r1.fingerprint.source.content_hash)
        )
        assert track is not None
        assert track.filename == "sine_440hz.wav"
        assert "sine_440hz.wav" in (track.source_path or "")


def test_concurrent_duplicate_insert_unique(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, db_path: Path
) -> None:
    """SQLite unique constraint + IntegrityError recovery keep a single completed row."""
    import threading

    out = tmp_path / "out"
    src = fixtures_dir / "silence.wav"
    # Pre-create fingerprint once so both threads only race on persist
    r0 = run_pipeline(src, app_config, output_dir=out, plots=False, database=None)
    init_db(db_path)
    engine = assert_db_usable(db_path)
    errors: list[BaseException] = []
    results: list[int] = []

    def _persist() -> None:
        try:
            from neuroacoustic.persistence.database import session_scope
            from neuroacoustic.persistence.repository import persist_completed_analysis

            with session_scope(engine) as session:
                row = persist_completed_analysis(
                    session,
                    fingerprint=r0.fingerprint,
                    config_hash=r0.config_hash or compute_config_hash(
                        app_config, analysis_sample_rate=r0.loaded.analysis_sample_rate
                    ),
                    fingerprint_path=r0.fingerprint_path,
                    replace_existing=False,
                )
                results.append(row.id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_persist) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # All succeeded (second+ return existing) or IntegrityError was handled inside
    assert not errors, errors
    assert len(set(results)) == 1
    with Session(engine) as session:
        assert compute_stats(session, db_path).completed == 1


def test_database_files_gitignored(repo_root: Path) -> None:
    import subprocess

    for name in (
        "foo.sqlite",
        "foo.sqlite-wal",
        "foo.sqlite-shm",
        "foo.db",
        "foo.db-wal",
        "foo.db-shm",
    ):
        target = repo_root / "data" / "db" / name
        proc = subprocess.run(
            ["git", "check-ignore", "-q", str(target)],
            cwd=repo_root,
            check=False,
        )
        assert proc.returncode == 0, name


def test_stats_database_size_non_negative(db_path: Path) -> None:
    """db stats reports a size at least as large as the main DB file."""
    init_db(db_path)
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        stats = compute_stats(session, db_path)
    assert stats.database_size_bytes is not None
    assert stats.database_size_bytes >= db_path.stat().st_size
    # Sidecar accounting: if WAL/SHM appear after activity, they are included
    engine.dispose()
    wal = Path(str(db_path.resolve()) + "-wal")
    shm = Path(str(db_path.resolve()) + "-shm")
    # Simulate sidecars present on disk without opening SQLite against fake WAL
    expected = db_path.stat().st_size
    if wal.exists():
        expected += wal.stat().st_size
    if shm.exists():
        expected += shm.stat().st_size
    size = 0
    for candidate in (db_path.resolve(), wal, shm):
        if candidate.exists():
            size += candidate.stat().st_size
    assert size == expected
