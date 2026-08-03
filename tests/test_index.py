"""Batch directory indexing: discovery, resume, fail-fast, reports, integrity."""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from neuroacoustic.cli import app
from neuroacoustic.config import AppConfig
from neuroacoustic.exceptions import AudioValidationError
from neuroacoustic.fingerprint.models import AcousticFingerprint
from neuroacoustic.indexing.discovery import discover_audio_files
from neuroacoustic.indexing.indexer import IndexOptions, run_index, validate_workers
from neuroacoustic.indexing.report import IndexReport, truncate_error, write_report_atomic
from neuroacoustic.persistence.database import assert_db_usable
from neuroacoustic.persistence.repository import (
    fingerprint_from_analysis,
    get_completed_analysis,
    list_analyses,
)
from neuroacoustic.pipeline import run_pipeline
from neuroacoustic.persistence.config_hash import compute_config_hash

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


def _write_tone(
    path: Path,
    *,
    freq: float = 440.0,
    sr: int = 22050,
    seconds: float = 0.25,
    format: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float64)
    y = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    kwargs: dict = {"subtype": "PCM_16"}
    if format is not None:
        kwargs["format"] = format
    sf.write(str(path), y, sr, **kwargs)


def _ensure_lib(tmp_path: Path) -> Path:
    root = tmp_path / "lib"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovery_recursive_and_non_recursive(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    _write_tone(root / "a.wav")
    _write_tone(root / "sub" / "b.wav")
    (root / "notes.txt").write_text("nope", encoding="utf-8")

    nested = discover_audio_files(root, recursive=True)
    flat = discover_audio_files(root, recursive=False)
    assert nested.discovered == 2
    assert nested.supported == 2
    assert flat.discovered == 1
    assert flat.supported == 1
    assert list(flat)[0].name == "a.wav"


def test_discovery_case_insensitive_extensions(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    _write_tone(root / "Tone.WAV", format="WAV")
    _write_tone(root / "other.WaVe", format="WAV")
    found = discover_audio_files(root)
    names = {p.name for p in found}
    assert "Tone.WAV" in names
    assert "other.WaVe" in names


def test_discovery_unsupported_ignored(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    _write_tone(root / "ok.wav")
    (root / "readme.txt").write_text("x", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG")
    result = discover_audio_files(root)
    assert result.discovered == 1
    assert result.supported == 1


def test_discovery_spaces_and_unicode(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    path = root / "café sounds" / "tone 440.wav"
    _write_tone(path)
    found = discover_audio_files(root)
    assert found.supported == 1
    assert list(found)[0].name == "tone 440.wav"


def test_discovery_include_exclude_distinct_counters(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    _write_tone(root / "keep.wav")
    _write_tone(root / "drop.wav")
    _write_tone(root / "nested" / "keep.wav")
    only_keep = discover_audio_files(root, include=["keep.wav"])
    assert only_keep.discovered == 3
    assert only_keep.supported == 2
    assert {p.name for p in only_keep} == {"keep.wav"}
    no_drop = discover_audio_files(root, exclude=["drop.wav"])
    assert no_drop.discovered == 3
    assert no_drop.supported == 2
    assert all(p.name != "drop.wav" for p in no_drop)


def test_discovery_symlinks_disabled_by_default(tmp_path: Path) -> None:
    root = _ensure_lib(tmp_path)
    real = tmp_path / "outside" / "real.wav"
    _write_tone(real)
    (root / "link.wav").symlink_to(real)
    (root / "subdir_link").symlink_to(tmp_path / "outside")
    _write_tone(root / "local.wav")
    found = discover_audio_files(root, follow_symlinks=False)
    assert found.discovered == 1
    assert found.supported == 1
    assert list(found)[0].name == "local.wav"


def test_discovery_symlink_loop_safe(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    a = root / "a"
    b = root / "b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "to_b").symlink_to(b)
    (b / "to_a").symlink_to(a)
    _write_tone(a / "tone.wav")
    found = discover_audio_files(root, follow_symlinks=True)
    assert found.supported == 1


def test_discovery_deterministic_order(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    for name in ("c.wav", "a.wav", "B.wav"):
        _write_tone(root / name)
    first = [str(p) for p in discover_audio_files(root)]
    second = [str(p) for p in discover_audio_files(root)]
    assert first == second
    assert [Path(p).name for p in first] == sorted(
        [Path(p).name for p in first], key=str.casefold
    )


def test_discovery_skips_output_and_database(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    out = root / "output"
    db = root / "catalog.sqlite"
    _write_tone(root / "ok.wav")
    _write_tone(out / "artifact.wav")
    db.write_bytes(b"not really sqlite")
    found = discover_audio_files(
        root,
        skip_dirs=[out],
        skip_files=[db, f"{db}-wal", f"{db}-shm"],
    )
    assert found.discovered == 1
    assert found.supported == 1
    assert list(found)[0].name == "ok.wav"


# ---------------------------------------------------------------------------
# Indexing / resume / integrity
# ---------------------------------------------------------------------------


def test_index_first_analyzes_second_reuses(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    shutil.copy(fixtures_dir / "sine_440hz.wav", root / "a.wav")
    shutil.copy(fixtures_dir / "sine_110hz.wav", root / "b.wav")

    opts = IndexOptions(root=root, database=db, output_dir=out, plots=False, workers=1)
    r1 = run_index(app_config, opts)
    assert r1.report.analyzed == 2
    assert r1.report.reused == 0
    assert r1.report.failed == 0
    assert r1.report.discovered == r1.report.supported == 2
    assert r1.report.counts_consistent()

    r2 = run_index(app_config, opts)
    assert r2.report.analyzed == 0
    assert r2.report.reused == 2
    assert r2.report.failed == 0
    assert r2.exit_code == 0


def test_index_identical_bytes_different_names_reuse_identity(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    src = fixtures_dir / "sine_440hz.wav"
    shutil.copy(src, root / "first.wav")
    shutil.copy(src, root / "second.wav")

    opts = IndexOptions(root=root, database=db, output_dir=out, plots=False)
    r1 = run_index(app_config, opts)
    assert r1.report.analyzed == 1
    assert r1.report.reused == 1
    engine = assert_db_usable(db)
    with Session(engine) as session:
        rows = list_analyses(session)
        assert len(rows) == 1


def test_same_basename_different_bytes_no_artifact_collision(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    a = root / "A" / "track.wav"
    b = root / "B" / "track.wav"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    shutil.copy(fixtures_dir / "sine_440hz.wav", a)
    shutil.copy(fixtures_dir / "sine_110hz.wav", b)

    unicode_path = root / "café dir" / "track with spaces.wav"
    unicode_path.parent.mkdir(parents=True)
    shutil.copy(fixtures_dir / "silence.wav", unicode_path)

    # Plots with workers=1 (matplotlib Agg is not guaranteed thread-safe).
    result = run_index(
        app_config,
        IndexOptions(root=root, database=db, output_dir=out, plots=True, workers=1),
    )
    assert result.report.analyzed == 3, [f.error for f in result.report.files]
    assert result.report.failed == 0
    hashes = {item.content_hash for item in result.report.files}
    assert len(hashes) == 3
    json_paths = list(out.rglob("*_fingerprint.json"))
    pngs = list(out.rglob("*_waveform.png"))
    assert len(json_paths) == 3
    assert len(pngs) == 3
    assert len({p.resolve() for p in json_paths}) == 3
    assert len({p.resolve() for p in pngs}) == 3
    for p in json_paths:
        assert "by_hash" in p.parts
        assert any(h and h in p.name for h in hashes)


def test_same_basename_concurrent_workers_no_identity_collision(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    for sub, name in (("A", "sine_440hz.wav"), ("B", "sine_110hz.wav")):
        dest = root / sub / "track.wav"
        dest.parent.mkdir(parents=True)
        shutil.copy(fixtures_dir / name, dest)
    # Duplicate of A under another name — should reuse, not collide.
    shutil.copy(fixtures_dir / "sine_440hz.wav", root / "A" / "alias.wav")

    result = run_index(
        app_config,
        IndexOptions(root=root, database=db, output_dir=out, plots=False, workers=2),
    )
    assert result.report.failed == 0, [f.error for f in result.report.files]
    assert result.report.analyzed + result.report.reused == 3
    engine = assert_db_usable(db)
    with Session(engine) as session:
        assert len(list_analyses(session)) == 2


def test_index_corrupt_continues(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    shutil.copy(fixtures_dir / "sine_440hz.wav", root / "good.wav")
    (root / "bad.wav").write_bytes(b"RIFF....not audio")

    opts = IndexOptions(
        root=root,
        database=db,
        output_dir=out,
        plots=False,
        continue_on_error=True,
    )
    result = run_index(app_config, opts)
    assert result.report.analyzed == 1
    assert result.report.failed == 1
    assert result.exit_code == 2
    assert result.report.counts_consistent()


def test_index_fail_fast(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    (root / "aaa_bad.wav").write_bytes(b"not-a-wav")
    shutil.copy(fixtures_dir / "sine_440hz.wav", root / "zzz_good.wav")

    opts = IndexOptions(
        root=root,
        database=db,
        output_dir=out,
        plots=False,
        continue_on_error=False,
        workers=1,
    )
    result = run_index(app_config, opts)
    assert result.report.failed == 1
    assert result.report.skipped >= 1
    assert result.report.analyzed == 0
    assert result.exit_code == 1
    assert result.report.counts_consistent()


def test_index_interrupt_exit_130(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    report_path = tmp_path / "partial.json"
    shutil.copy(fixtures_dir / "sine_440hz.wav", root / "a.wav")
    shutil.copy(fixtures_dir / "sine_110hz.wav", root / "b.wav")
    shutil.copy(fixtures_dir / "silence.wav", root / "c.wav")

    handler_box: dict = {}

    def capture(sig, handler):  # type: ignore[no-untyped-def]
        handler_box["handler"] = handler
        return signal.SIG_DFL

    import neuroacoustic.indexing.indexer as indexer_mod

    monkey_signal = pytest.MonkeyPatch()
    monkey_signal.setattr(indexer_mod.signal, "signal", capture)

    def on_progress(completed: int, total: int, item) -> None:  # type: ignore[no-untyped-def]
        if completed >= 1 and "handler" in handler_box:
            handler_box["handler"](signal.SIGINT, None)

    try:
        result = run_index(
            app_config,
            IndexOptions(
                root=root,
                database=db,
                output_dir=out,
                plots=False,
                workers=1,
                report_path=report_path,
            ),
            progress_callback=on_progress,
        )
    finally:
        monkey_signal.undo()

    assert result.exit_code == 130
    assert result.interrupted is True
    assert result.report.interrupted is True
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert IndexReport.model_validate(payload).counts_consistent()
    assert payload["skipped"] + payload["analyzed"] + payload["reused"] + payload[
        "failed"
    ] + payload["forced"] == payload["supported"]


def test_file_changed_during_analysis_refuses_persist(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If bytes change after load, do not persist a mismatched identity."""
    import neuroacoustic.pipeline as pipeline_mod

    src = tmp_path / "mutable.wav"
    shutil.copy(fixtures_dir / "sine_440hz.wav", src)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    other = fixtures_dir / "sine_110hz.wav"

    real_run = pipeline_mod._run_analysis

    def corrupt_then_analyze(loaded, config, *, out_dir, plots):  # type: ignore[no-untyped-def]
        # Replace on-disk bytes with different audio after load/hash.
        shutil.copy(other, src)
        return real_run(loaded, config, out_dir=out_dir, plots=plots)

    monkeypatch.setattr(pipeline_mod, "_run_analysis", corrupt_then_analyze)

    with pytest.raises(AudioValidationError, match="file_changed_during_analysis"):
        run_pipeline(src, app_config, output_dir=out, plots=False, database=db)

    engine = assert_db_usable(db)
    cfg_hash = compute_config_hash(app_config)
    from neuroacoustic.audio.probe import sha256_file

    # Original identity must not be completed (hash was of pre-corrupt bytes).
    original_hash = sha256_file(fixtures_dir / "sine_440hz.wav")
    with Session(engine) as session:
        completed = get_completed_analysis(
            session,
            content_hash=original_hash,
            analysis_version=app_config.project.analysis_version,
            config_hash=cfg_hash,
        )
        assert completed is None


def test_report_atomic_and_counts(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    report_path = tmp_path / "report.json"
    shutil.copy(fixtures_dir / "silence.wav", root / "s.wav")

    opts = IndexOptions(
        root=root,
        database=db,
        output_dir=out,
        plots=False,
        report_path=report_path,
    )
    result = run_index(app_config, opts)
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report = IndexReport.model_validate(payload)
    assert report.counts_consistent()
    _finite(payload)
    assert result.report_path == report_path.resolve()


def test_write_report_atomic_replace(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    report = IndexReport(
        root_directory="/x",
        database_path="/y.sqlite",
        started_at="2020-01-01T00:00:00+00:00",
        finished_at="2020-01-01T00:00:01+00:00",
        elapsed_seconds=1.0,
        recursive=True,
        follow_symlinks=False,
        workers=1,
        force=False,
        plots=False,
        continue_on_error=True,
        discovered=0,
        supported=0,
        analyzed=0,
        reused=0,
        forced=0,
        failed=0,
        skipped=0,
        schema_version="0.4.0",
        analysis_version="0.4.0",
        vector_version="0.4.0-preliminary",
        config_hash="abc",
    )
    write_report_atomic(report, path)
    write_report_atomic(report.model_copy(update={"analyzed": 1, "supported": 1}), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["analyzed"] == 1


def test_truncate_error_bounded() -> None:
    assert len(truncate_error("x" * 2000)) <= 500


def test_cli_json_stdout_only(fixtures_dir: Path, tmp_path: Path) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    shutil.copy(fixtures_dir / "sine_110hz.wav", root / "x.wav")
    result = runner.invoke(
        app,
        [
            "index",
            str(root),
            "--database",
            str(db),
            "--output-dir",
            str(out),
            "--no-plots",
            "--workers",
            "1",
            "--json",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["analyzed"] == 1
    _finite(payload)
    IndexReport.model_validate(payload)
    assert "Traceback" not in result.stdout


def test_cli_invalid_directory_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "index",
            str(tmp_path / "missing"),
            "--database",
            str(tmp_path / "db.sqlite"),
            "--output-dir",
            str(tmp_path / "out"),
            "--quiet",
        ],
    )
    assert result.exit_code != 0


def test_cli_workers_validation(tmp_path: Path, fixtures_dir: Path) -> None:
    root = _ensure_lib(tmp_path)
    shutil.copy(fixtures_dir / "sine_440hz.wav", root / "a.wav")
    result = runner.invoke(
        app,
        [
            "index",
            str(root),
            "--database",
            str(tmp_path / "db.sqlite"),
            "--output-dir",
            str(tmp_path / "out"),
            "--workers",
            "0",
            "--quiet",
        ],
    )
    assert result.exit_code != 0
    with pytest.raises(ValueError):
        validate_workers(0)
    with pytest.raises(ValueError):
        validate_workers(99)


def test_index_no_self_discovery_after_artifacts(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    out = root / "output"
    db = root / "neuroacoustic.sqlite"
    report = root / "output" / "index-report.json"
    shutil.copy(fixtures_dir / "sine_440hz.wav", root / "keep.wav")
    _write_tone(out / "should_skip.wav")

    opts = IndexOptions(
        root=root,
        database=db,
        output_dir=out,
        plots=True,
        report_path=report,
    )
    first = run_index(app_config, opts)
    assert first.report.supported == 1
    assert first.report.analyzed == 1
    assert (out / "by_hash").exists()

    second = run_index(app_config, opts)
    assert second.report.discovered == 1
    assert second.report.supported == 1
    assert second.report.reused == 1
    assert second.report.analyzed == 0


def test_index_validates_db_fingerprints(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    root = _ensure_lib(tmp_path)
    db = tmp_path / "db.sqlite"
    out = tmp_path / "out"
    shutil.copy(fixtures_dir / "sine_440hz.wav", root / "a.wav")
    run_index(
        app_config,
        IndexOptions(root=root, database=db, output_dir=out, plots=False),
    )
    engine = assert_db_usable(db)
    with Session(engine) as session:
        for row in list_analyses(session):
            fp = fingerprint_from_analysis(row)
            assert isinstance(fp, AcousticFingerprint)
            AcousticFingerprint.model_validate(fp.model_dump())
            _finite(fp.model_dump(mode="json"))
