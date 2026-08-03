"""Batch directory indexing over the single-file analysis pipeline."""

from __future__ import annotations

import os
import signal
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from neuroacoustic.config import AppConfig
from neuroacoustic.exceptions import NeuroAcousticError
from neuroacoustic.indexing.discovery import DEFAULT_INDEX_EXTENSIONS, discover_audio_files
from neuroacoustic.indexing.report import (
    FileIndexResult,
    IndexReport,
    assert_json_finite,
    truncate_error,
    write_report_atomic,
    _utc_now_iso,
)
from neuroacoustic.logging import get_logger
from neuroacoustic.persistence.config_hash import compute_config_hash
from neuroacoustic.persistence.database import init_db
from neuroacoustic.pipeline import run_pipeline

logger = get_logger(__name__)

_MAX_WORKERS = 8


@dataclass
class IndexOptions:
    root: Path
    database: Path
    output_dir: Path
    recursive: bool = True
    follow_symlinks: bool = False
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    plots: bool = False
    force: bool = False
    continue_on_error: bool = True
    workers: int = 1
    report_path: Path | None = None
    quiet: bool = False


@dataclass
class IndexRunResult:
    report: IndexReport
    report_path: Path | None
    interrupted: bool
    exit_code: int


def validate_workers(workers: int) -> int:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if workers > _MAX_WORKERS:
        raise ValueError(f"--workers must be <= {_MAX_WORKERS}")
    return int(workers)


def _limit_blas_threads() -> None:
    """Best-effort BLAS/OpenMP caps when workers > 1.

    These environment variables are only reliable if set **before** NumPy/BLAS
    libraries are imported and initialized. This helper uses ``setdefault`` and
    may have no effect when NeuroAcoustic (and NumPy) are already loaded.
    Prefer ``--workers 1`` for initial production runs.
    """
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _analyze_one(
    path: Path,
    *,
    config: AppConfig,
    options: IndexOptions,
    stop_event: threading.Event,
) -> FileIndexResult:
    if stop_event.is_set():
        return FileIndexResult(
            path=str(path),
            disposition="skipped",
            error="interrupted_before_start",
        )

    started = time.perf_counter()
    if not path.exists():
        return FileIndexResult(
            path=str(path),
            disposition="failed",
            error=truncate_error("file disappeared or unreadable before analysis"),
            elapsed_seconds=time.perf_counter() - started,
        )

    try:
        result = run_pipeline(
            path,
            config,
            output_dir=options.output_dir,
            plots=options.plots,
            force=options.force,
            database=options.database,
        )
    except NeuroAcousticError as exc:
        return FileIndexResult(
            path=str(path),
            disposition="failed",
            error=truncate_error(str(exc)),
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return FileIndexResult(
            path=str(path),
            disposition="failed",
            error=truncate_error(f"{type(exc).__name__}: {exc}"),
            elapsed_seconds=time.perf_counter() - started,
        )

    disposition = result.disposition or "analyzed"
    if disposition not in {"analyzed", "reused", "forced"}:
        disposition = "analyzed"

    fp = result.fingerprint
    return FileIndexResult(
        path=str(path),
        disposition=disposition,  # type: ignore[arg-type]
        analysis_id=result.analysis_id,
        content_hash=fp.source.content_hash,
        duration_seconds=fp.source.duration_seconds,
        elapsed_seconds=time.perf_counter() - started,
        warnings=list(result.warnings),
    )


ProgressCallback = Callable[[int, int, FileIndexResult], None]


def run_index(
    config: AppConfig,
    options: IndexOptions,
    *,
    progress_callback: ProgressCallback | None = None,
    on_discovered: Callable[[int], None] | None = None,
) -> IndexRunResult:
    """Discover and analyze audio under ``options.root``.

    SQLite is the resume/cache state. Default ``workers=1``. When ``workers>1``,
    each task uses its own pipeline/DB sessions (no shared Session objects).

    Exit codes
    ----------
    - ``0`` — completed with no per-file failures
    - ``2`` — completed with one or more per-file failures (continue-on-error)
    - ``1`` — fail-fast stopped after a failure, or other batch failure
    - ``130`` — interrupted (SIGINT); partial report is still written
    """
    workers = validate_workers(int(options.workers))
    if workers > 1:
        _limit_blas_threads()

    root = Path(options.root).expanduser()
    database = Path(options.database).expanduser()
    output_dir = Path(options.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    init_db(database)

    started_mono = time.perf_counter()
    started_at = _utc_now_iso()
    config_hash = compute_config_hash(config)

    skip_dirs = [output_dir]
    skip_files: list[Path | str] = [database, f"{database}-wal", f"{database}-shm"]
    if options.report_path is not None:
        skip_files.append(options.report_path)

    extensions = set(config.audio.supported_extensions) | set(DEFAULT_INDEX_EXTENSIONS)

    discovery = discover_audio_files(
        root,
        recursive=options.recursive,
        follow_symlinks=options.follow_symlinks,
        include=options.include,
        exclude=options.exclude,
        extensions=extensions,
        skip_dirs=skip_dirs,
        skip_files=skip_files,
    )
    files = list(discovery.files)
    supported = discovery.supported
    discovered = discovery.discovered
    if on_discovered is not None:
        on_discovered(supported)

    stop_event = threading.Event()
    interrupted = False
    results: list[FileIndexResult] = []
    fail_fast_triggered = False

    previous_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum, frame):  # type: ignore[no-untyped-def]
        nonlocal interrupted
        interrupted = True
        stop_event.set()
        logger.warning("Interrupt received — finishing in-flight files, then stopping")

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except ValueError:
        pass

    def _record(item: FileIndexResult) -> None:
        nonlocal fail_fast_triggered
        results.append(item)
        if progress_callback is not None:
            progress_callback(len(results), supported, item)
        if item.disposition == "failed" and not options.continue_on_error:
            fail_fast_triggered = True
            stop_event.set()

    try:
        if workers == 1:
            for idx, path in enumerate(files):
                if stop_event.is_set():
                    reason = "interrupted" if interrupted else "fail_fast_stopped"
                    for rest in files[idx:]:
                        results.append(
                            FileIndexResult(
                                path=str(rest),
                                disposition="skipped",
                                error=reason,
                            )
                        )
                    break
                item = _analyze_one(
                    path, config=config, options=options, stop_event=stop_event
                )
                _record(item)
                if fail_fast_triggered:
                    for rest in files[idx + 1 :]:
                        results.append(
                            FileIndexResult(
                                path=str(rest),
                                disposition="skipped",
                                error="fail_fast_stopped",
                            )
                        )
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map: dict = {}
                for path in files:
                    if stop_event.is_set():
                        results.append(
                            FileIndexResult(
                                path=str(path),
                                disposition="skipped",
                                error="interrupted" if interrupted else "fail_fast_stopped",
                            )
                        )
                        continue
                    fut = pool.submit(
                        _analyze_one,
                        path,
                        config=config,
                        options=options,
                        stop_event=stop_event,
                    )
                    future_map[fut] = path

                for fut in as_completed(list(future_map.keys())):
                    path = future_map[fut]
                    try:
                        item = fut.result()
                    except CancelledError:
                        item = FileIndexResult(
                            path=str(path),
                            disposition="skipped",
                            error="interrupted" if interrupted else "fail_fast_stopped",
                        )
                    except Exception as exc:  # noqa: BLE001
                        item = FileIndexResult(
                            path=str(path),
                            disposition="failed",
                            error=truncate_error(f"{type(exc).__name__}: {exc}"),
                        )
                    _record(item)
                    if fail_fast_triggered or interrupted:
                        for pending in future_map:
                            pending.cancel()
    finally:
        try:
            signal.signal(signal.SIGINT, previous_sigint)
        except ValueError:
            pass

    recorded = {r.path for r in results}
    skip_reason = "interrupted" if interrupted else "fail_fast_stopped"
    for path in files:
        key = str(path)
        if key not in recorded:
            results.append(
                FileIndexResult(
                    path=key,
                    disposition="skipped",
                    error=skip_reason,
                )
            )
            recorded.add(key)

    by_path = {r.path: r for r in results}
    ordered = [by_path[str(p)] for p in files if str(p) in by_path]
    for r in results:
        if r.path not in {x.path for x in ordered}:
            ordered.append(r)

    tallies = {"analyzed": 0, "reused": 0, "forced": 0, "failed": 0, "skipped": 0}
    for item in ordered:
        tallies[item.disposition] += 1

    report = IndexReport(
        root_directory=str(root.resolve()) if root.exists() else str(root),
        database_path=str(database.resolve()) if database.parent.exists() else str(database),
        started_at=started_at,
        finished_at=_utc_now_iso(),
        elapsed_seconds=round(time.perf_counter() - started_mono, 3),
        recursive=options.recursive,
        follow_symlinks=options.follow_symlinks,
        workers=workers,
        force=options.force,
        plots=options.plots,
        continue_on_error=options.continue_on_error,
        interrupted=interrupted,
        discovered=discovered,
        supported=supported,
        analyzed=tallies["analyzed"],
        reused=tallies["reused"],
        forced=tallies["forced"],
        failed=tallies["failed"],
        skipped=tallies["skipped"],
        schema_version=config.project.schema_version,
        analysis_version=config.project.analysis_version,
        vector_version=config.analysis.vector_version,
        config_hash=config_hash,
        include=list(options.include),
        exclude=list(options.exclude),
        files=ordered,
    )

    assert_json_finite(report.model_dump(mode="json"))

    report_path = None
    if options.report_path is not None:
        report_path = write_report_atomic(report, options.report_path)

    exit_code = 0
    if interrupted:
        exit_code = 130
    elif tallies["failed"] and not options.continue_on_error:
        exit_code = 1
    elif tallies["failed"]:
        exit_code = 2

    return IndexRunResult(
        report=report,
        report_path=report_path,
        interrupted=interrupted,
        exit_code=exit_code,
    )


__all__ = [
    "IndexOptions",
    "IndexRunResult",
    "run_index",
    "validate_workers",
]
