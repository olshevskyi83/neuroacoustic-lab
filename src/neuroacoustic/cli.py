"""CLI entry points for NeuroAcoustic Lab."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import Session

from neuroacoustic import __analysis_version__, __schema_version__, __version__
from neuroacoustic.audio.probe import ffmpeg_available, ffprobe_available, probe_audio
from neuroacoustic.config import load_config
from neuroacoustic.exceptions import (
    AnalysisNotFoundError,
    DatabaseError,
    NeuroAcousticError,
)
from neuroacoustic.logging import setup_logging
from neuroacoustic.persistence.database import assert_db_usable, init_db
from neuroacoustic.persistence.models import DB_SCHEMA_VERSION
from neuroacoustic.persistence.repository import (
    analysis_to_summary_dict,
    compute_stats,
    fingerprint_from_analysis,
    list_analyses,
    resolve_analysis_or_track_id,
)
from neuroacoustic.qdrant import (
    QdrantClient,
    QdrantSettings,
    metric_explanation,
    sync_completed,
    sync_calibrated,
    validated_analysis_vector,
)
from neuroacoustic.calibration import (
    DEFAULT_CALIBRATED_COLLECTION, build_profile, save_profile,
)

app = typer.Typer(
    name="neuroacoustic",
    help="Local-first physical acoustic analysis of sound.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="SQLite database commands.")
app.add_typer(db_app, name="db")
qdrant_app = typer.Typer(help="Rebuildable Qdrant projection commands.")
app.add_typer(qdrant_app, name="qdrant")

console = Console(stderr=False)
err_console = Console(stderr=True)


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def _package_version(module: str) -> str:
    """Best-effort version string for an importable module."""
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", None)
        if version:
            return str(version)
    except Exception:  # noqa: BLE001
        pass
    try:
        return importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _check_python_import(module: str) -> tuple[bool, str]:
    try:
        __import__(module)
        return True, _package_version(module)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _resolve_database_option(
    database: Path | None,
    config_path: Path | None,
    *,
    required: bool = True,
) -> Path:
    cfg = load_config(config_path)
    path = database if database is not None else cfg.paths.database
    if path is None and required:
        _fail("No database path: pass --database or set paths.database in config.")
    return Path(path)


def _qdrant_client(*, require_enabled: bool = True) -> QdrantClient:
    settings = QdrantSettings.from_environment()
    if require_enabled and not settings.enabled:
        _fail("Qdrant is disabled. Set QDRANT_ENABLED=true to allow this command.")
    return QdrantClient(settings)


@qdrant_app.command("doctor")
def qdrant_doctor() -> None:
    """Check Qdrant reachability without creating or changing a collection."""
    try:
        settings = QdrantSettings.from_environment()
        table = Table(title="NeuroAcoustic — Qdrant doctor")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail")
        table.add_row("enabled", "ok" if settings.enabled else "disabled", str(settings.enabled).lower())
        table.add_row("url", "ok", settings.url)
        table.add_row("collection", "ok", settings.collection)
        table.add_row("vector", "ok", "36 dimensions / Cosine / 0.4.0-preliminary")
        if settings.enabled:
            QdrantClient(settings).health()
            table.add_row("health", "ok", "reachable")
        else:
            table.add_row("health", "skipped", "set QDRANT_ENABLED=true to test")
        console.print(table)
    except NeuroAcousticError as exc:
        _fail(str(exc))


@qdrant_app.command("init")
def qdrant_init() -> None:
    """Idempotently create or verify only the configured dedicated collection."""
    try:
        client = _qdrant_client()
        created = client.initialize_collection()
        console.print(
            f"Qdrant collection {client.settings.collection!r} "
            f"{'created' if created else 'already compatible'} (36 / Cosine)."
        )
    except NeuroAcousticError as exc:
        _fail(str(exc))


@qdrant_app.command("status")
def qdrant_status() -> None:
    """Show projection collection state without modifying it."""
    try:
        settings = QdrantSettings.from_environment()
        if not settings.enabled:
            console.print("Qdrant disabled (QDRANT_ENABLED=false); no request made.")
            return
        info = QdrantClient(settings).collection_info()
        if info is None:
            console.print(f"Collection {settings.collection!r} does not exist.")
            return
        vectors = info.get("config", {}).get("params", {}).get("vectors", {})
        console.print(
            f"collection={settings.collection} points={info.get('points_count', 0)} "
            f"size={vectors.get('size')} distance={vectors.get('distance')}"
        )
    except NeuroAcousticError as exc:
        _fail(str(exc))


@qdrant_app.command("sync")
def qdrant_sync(
    database: Path = typer.Option(..., "--database", help="SQLite source database path"),
) -> None:
    """Idempotently project completed SQLite analyses; never deletes stale points."""
    try:
        engine = assert_db_usable(database)
        client = _qdrant_client()
        with Session(engine) as session:
            report = sync_completed(session, client)
        console.print(
            f"eligible={report.eligible} synced={report.synced} invalid={report.skipped_invalid} "
            f"missing_before_sync={report.missing_before_sync} stale_not_deleted={report.stale_points}"
        )
    except NeuroAcousticError as exc:
        _fail(str(exc))


def _calibration_profile(session: Session):
    from neuroacoustic.qdrant import completed_rows
    rows = completed_rows(session)
    vectors = [validated_analysis_vector(row) for row in rows]
    return build_profile(vectors, [row.id for row in rows])


@qdrant_app.command("calibration-report")
def qdrant_calibration_report(
    database: Path = typer.Option(..., "--database"),
    profile: Optional[Path] = typer.Option(None, "--profile", help="Optional JSON profile output"),
) -> None:
    """Report robust corpus calibration; optionally save its reproducible JSON profile."""
    try:
        engine = assert_db_usable(database)
        with Session(engine) as session:
            value = _calibration_profile(session)
        if profile:
            save_profile(value, profile)
        console.print(f"version={value.version} samples={value.sample_count} inactive_dimensions={value.inactive_dimensions}")
        console.print("profile=" + (str(profile) if profile else "not written (pass --profile PATH)"))
    except (NeuroAcousticError, ValueError) as exc:
        _fail(str(exc))


@qdrant_app.command("sync-calibrated")
def qdrant_sync_calibrated(
    database: Path = typer.Option(..., "--database"),
    profile: Path = typer.Option(..., "--profile", exists=True),
) -> None:
    """Build the separate calibrated projection from a saved profile."""
    try:
        from neuroacoustic.calibration import load_profile
        engine = assert_db_usable(database)
        settings = QdrantSettings.from_environment()
        if not settings.enabled: _fail("Qdrant is disabled. Set QDRANT_ENABLED=true to allow this command.")
        if settings.collection != DEFAULT_CALIBRATED_COLLECTION:
            _fail(f"Set QDRANT_COLLECTION={DEFAULT_CALIBRATED_COLLECTION!r} for calibrated sync.")
        with Session(engine) as session:
            report = sync_calibrated(session, QdrantClient(settings), load_profile(profile))
        console.print(f"eligible={report.eligible} synced={report.synced} invalid={report.skipped_invalid} missing_before_sync={report.missing_before_sync}")
    except (NeuroAcousticError, ValueError) as exc:
        _fail(str(exc))


@app.command("similar")
def similar(
    analysis_id: str = typer.Argument(..., metavar="ANALYSIS_ID"),
    top: int = typer.Option(5, "--top", min=1, max=100),
    database: Optional[Path] = typer.Option(None, "--database"),
    config: Optional[Path] = typer.Option(None, "--config"),
    space: str = typer.Option("preliminary", "--space", case_sensitive=False),
    profile: Optional[Path] = typer.Option(None, "--profile", help="Required for calibrated space"),
) -> None:
    """Find vector neighbours and display their numeric acoustic deltas."""
    try:
        db_path = _resolve_database_option(database, config)
        engine = assert_db_usable(db_path)
        with Session(engine) as session:
            query = resolve_analysis_or_track_id(session, analysis_id)
            if query.status != "completed":
                _fail(f"Analysis {query.id} is not completed.")
            vector = validated_analysis_vector(query)
            version = query.vector_version
            if space == "calibrated":
                if profile is None: _fail("--profile is required with --space calibrated.")
                from neuroacoustic.calibration import CALIBRATED_VECTOR_VERSION, load_profile, transform
                calibration = load_profile(profile); raw_vector = vector
                vector = transform(vector, calibration); version = CALIBRATED_VECTOR_VERSION
            elif space != "preliminary":
                _fail("--space must be preliminary or calibrated.")
            matches = _qdrant_client().search(vector, version, query.id, top)
        table = Table(title=f"Similar to analysis {query.id} (metric neighbours, not corpus-calibrated)")
        table.add_column("Score", justify="right")
        table.add_column("Analysis ID", justify="right")
        table.add_column("Filename")
        table.add_column("Vector metric explanation")
        for match in matches:
            payload = match.get("payload", {})
            candidate_vector = match.get("vector")
            explanation = "group metrics unavailable (Qdrant did not return vector)"
            if isinstance(candidate_vector, list) and all(
                isinstance(value, (int, float)) and math.isfinite(float(value)) for value in candidate_vector
            ):
                explanation = metric_explanation(vector, [float(value) for value in candidate_vector])
            if space == "calibrated":
                from neuroacoustic.calibration import explanation as calibrated_explanation
                raw_candidate = payload.get("preliminary_vector")
                if payload.get("calibration_profile_sha256") != calibration.profile_sha256:
                    _fail("calibration profile does not match calibrated Qdrant payload.")
                if not isinstance(raw_candidate, list) or len(raw_candidate) != 36:
                    _fail("calibrated Qdrant result lacks its raw-vector snapshot.")
                detail = calibrated_explanation(raw_vector, [float(v) for v in raw_candidate], calibration)
                best = ", ".join(f"{x['name']} Δz={x['standardized_difference']:.2f} w={x['weight']:.2f}" for x in detail["positive"][:3])
                worst = ", ".join(f"{x['name']} Δz={x['standardized_difference']:.2f} w={x['weight']:.2f}" for x in detail["negative"][:3])
                explanation = f"weighted standardized distance: {detail['groups']}; closest: {best}; differs: {worst}"
            table.add_row(
                f"{float(match.get('score', 0)):.6f}", str(payload.get("analysis_id", "—")),
                str(payload.get("filename") or "—"), explanation,
            )
        console.print(table)
    except NeuroAcousticError as exc:
        _fail(str(exc))


@app.command("doctor")
def doctor(
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to TOML config file"
    ),
) -> None:
    """Print environment and dependency status."""
    try:
        cfg = load_config(config)
    except NeuroAcousticError as exc:
        _fail(str(exc))

    setup_logging(cfg.logging.level, json_logs=cfg.logging.json_logs)

    table = Table(title="NeuroAcoustic Lab — doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")

    failures = 0
    warnings = 0

    def add_row(name: str, status: str, detail: str) -> None:
        nonlocal failures, warnings
        table.add_row(name, status, detail)
        if status == "fail":
            failures += 1
        elif status in {"missing", "warn"}:
            warnings += 1

    add_row("neuroacoustic", "ok", f"v{__version__}")
    add_row("schema_version", "ok", __schema_version__)
    add_row("analysis_version", "ok", __analysis_version__)
    add_row("db_schema_version", "ok", str(DB_SCHEMA_VERSION))
    add_row(
        "python",
        "ok" if sys.version_info >= (3, 11) else "fail",
        platform.python_version(),
    )
    add_row("platform", "ok", f"{platform.system()} {platform.machine()}")

    for mod in (
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "pydantic",
        "typer",
        "rich",
        "sqlalchemy",
    ):
        ok, detail = _check_python_import(mod)
        add_row(f"py:{mod}", "ok" if ok else "fail", detail)

    add_row(
        "ffmpeg",
        "ok" if ffmpeg_available() else "missing",
        shutil.which("ffmpeg") or "not on PATH — required for MP3 decode",
    )
    add_row(
        "ffprobe",
        "ok" if ffprobe_available() else "missing",
        shutil.which("ffprobe") or "not on PATH — required for MP3 probe",
    )

    cfg_path = cfg.config_path
    add_row(
        "config",
        "ok" if cfg_path and Path(cfg_path).exists() else "warn",
        str(cfg_path),
    )
    add_row("output_dir", "ok", str(cfg.paths.output_dir))
    add_row("database", "ok", str(cfg.paths.database))

    console.print(table)

    if not ffmpeg_available() or not ffprobe_available():
        err_console.print(
            "[yellow]warning:[/yellow] FFmpeg/ffprobe missing. "
            "WAV/FLAC/AIFF work via soundfile; MP3 requires FFmpeg."
        )

    if failures:
        err_console.print(
            f"[red]doctor failed:[/red] {failures} required check(s) failed."
        )
        raise typer.Exit(code=1)

    if warnings:
        err_console.print(
            f"[yellow]doctor completed with {warnings} warning(s).[/yellow]"
        )
        raise typer.Exit(code=0)

    console.print("[green]All core checks passed.[/green]")


@app.command("probe")
def probe_cmd(
    audio_file: Path = typer.Argument(..., exists=False, help="Path to audio file"),
    config: Optional[Path] = typer.Option(None, "--config", help="TOML config path"),
    analysis_sample_rate: Optional[int] = typer.Option(
        None,
        "--analysis-sample-rate",
        help="Override analysis sample rate (Hz); default preserves native rate",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON to stdout"),
) -> None:
    """Print source metadata without performing full analysis."""
    try:
        cfg = load_config(config)
        if analysis_sample_rate is not None:
            cfg = cfg.model_copy(
                update={
                    "audio": cfg.audio.model_copy(
                        update={"analysis_sample_rate": analysis_sample_rate}
                    )
                }
            )
        setup_logging(cfg.logging.level, json_logs=cfg.logging.json_logs)
        result = probe_audio(audio_file, cfg)
    except NeuroAcousticError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error: {exc}")

    if as_json:
        sys.stdout.write(json.dumps(result.to_printable_dict(), allow_nan=False) + "\n")
        return

    source = result.source
    table = Table(title=f"Probe: {source.filename}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    rows = [
        ("path", source.path),
        ("content_hash", source.content_hash),
        ("file_size_bytes", str(source.file_size_bytes)),
        ("container", source.container or "—"),
        ("codec", source.codec or "—"),
        ("duration_seconds", f"{source.duration_seconds:.6f}" if source.duration_seconds else "—"),
        ("native_sample_rate", str(source.native_sample_rate or "—")),
        ("analysis_sample_rate", str(source.analysis_sample_rate or "—")),
        ("channels", str(source.channels or "—")),
        ("bit_depth", str(source.bit_depth or "—")),
        ("bitrate", str(source.bitrate or "—")),
        ("probe_backend", source.probe_backend.value),
        ("schema_version", result.analysis_config.schema_version),
        ("analysis_version", result.analysis_config.analysis_version),
        ("created_at", result.created_at.isoformat()),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)
    for warning in source.warnings:
        err_console.print(f"[yellow]warning:[/yellow] {warning}")


@app.command("analyze")
def analyze_cmd(
    audio_file: Path = typer.Argument(..., help="Path to audio file"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    database: Optional[Path] = typer.Option(
        None,
        "--database",
        help=(
            "SQLite path — enables persistence. When omitted, analysis writes "
            "JSON/plots only (no database)."
        ),
    ),
    config: Optional[Path] = typer.Option(None, "--config"),
    analysis_sample_rate: Optional[int] = typer.Option(None, "--analysis-sample-rate"),
    plots: bool = typer.Option(
        True,
        "--plots/--no-plots",
        help="Generate waveform / FFT / spectrogram PNG plots",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Bypass max-duration guard; when --database is set, also re-analyze "
            "and replace the completed row for the same "
            "(content_hash, analysis_version, config_hash) identity"
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Print fingerprint JSON to stdout"),
) -> None:
    """Run analysis, write fingerprint JSON/plots, optionally persist to SQLite."""
    from neuroacoustic.pipeline import run_pipeline

    try:
        cfg = load_config(config)
        if analysis_sample_rate is not None:
            cfg = cfg.model_copy(
                update={
                    "audio": cfg.audio.model_copy(
                        update={"analysis_sample_rate": analysis_sample_rate}
                    )
                }
            )
        if output_dir is not None:
            cfg = cfg.model_copy(
                update={"paths": cfg.paths.model_copy(update={"output_dir": output_dir})}
            )
        setup_logging(cfg.logging.level, json_logs=cfg.logging.json_logs)
        result = run_pipeline(
            audio_file,
            cfg,
            output_dir=cfg.paths.output_dir,
            plots=plots,
            force=force,
            database=database,
        )
    except NeuroAcousticError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error: {exc}")

    fp = result.fingerprint
    if as_json:
        # Machine-readable: raw JSON on stdout only (logs already on stderr)
        sys.stdout.write(json.dumps(fp.to_json_dict(), allow_nan=False) + "\n")
        return

    title = f"Analyze: {fp.source.filename}"
    if result.disposition:
        title = f"{title} [{result.disposition}]"
    table = Table(title=title)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    rows = [
        ("disposition", result.disposition or "no-database"),
        ("analysis_id", str(result.analysis_id) if result.analysis_id is not None else "—"),
        ("config_hash", (result.config_hash[:16] + "…") if result.config_hash else "—"),
        ("fingerprint", str(result.fingerprint_path)),
        ("schema_version", fp.schema_version),
        ("analysis_version", fp.analysis_version),
        ("content_hash", fp.source.content_hash),
        ("duration_seconds", str(fp.source.duration_seconds)),
        ("analysis_sample_rate", str(fp.source.analysis_sample_rate)),
        ("fft_peak_hz", str(fp.spectral.fft.peak_frequency_hz)),
        ("centroid_mean_hz", str(fp.spectral.centroid_hz.mean)),
        ("flatness_mean", str(fp.spectral.flatness.mean)),
        ("f0_median_hz", str(fp.pitch.f0_median_hz)),
        ("tempo_bpm", str(fp.rhythm.tempo_bpm)),
        ("rms", str(fp.energy.rms)),
        ("vector_len", str(len(fp.vector))),
        ("vector_version", str(fp.vector_meta.version if fp.vector_meta else "—")),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)
    for warning in fp.quality.warnings:
        err_console.print(f"[yellow]warning:[/yellow] {warning}")
    for warning in result.warnings:
        err_console.print(f"[yellow]artifact:[/yellow] {warning}")
    if fp.artifacts.waveform_png and Path(fp.artifacts.waveform_png).exists():
        console.print(f"plots: {fp.artifacts.waveform_png}")
    elif result.disposition == "reused" and not plots:
        pass
    elif plots and not (fp.artifacts.waveform_png and Path(fp.artifacts.waveform_png).exists()):
        err_console.print(
            "[yellow]warning:[/yellow] plots were requested but no waveform PNG is available"
        )


@db_app.command("init")
def db_init(
    database: Optional[Path] = typer.Option(None, "--database"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Create/verify the SQLite schema idempotently."""
    try:
        path = _resolve_database_option(database, config)
        setup_logging(load_config(config).logging.level)
        init_db(path)
    except NeuroAcousticError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error: {exc}")
    console.print(f"[green]Database ready[/green] schema_v{DB_SCHEMA_VERSION}: {path}")


@db_app.command("list")
def db_list(
    database: Optional[Path] = typer.Option(None, "--database"),
    config: Optional[Path] = typer.Option(None, "--config"),
    limit: int = typer.Option(50, "--limit", min=1),
    offset: int = typer.Option(0, "--offset", min=0),
    status: Optional[str] = typer.Option(
        None, "--status", help="Filter: completed|failed|in_progress"
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List analyses (newest first)."""
    try:
        path = _resolve_database_option(database, config)
        engine = assert_db_usable(path)
        with Session(engine) as session:
            rows = list_analyses(session, limit=limit, offset=offset, status=status)
            summaries = [analysis_to_summary_dict(r) for r in rows]
    except AnalysisNotFoundError as exc:
        _fail(str(exc), code=1)
    except DatabaseError as exc:
        _fail(str(exc), code=1)
    except NeuroAcousticError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error: {exc}")

    if as_json:
        sys.stdout.write(json.dumps(summaries, allow_nan=False) + "\n")
        return

    if not summaries:
        console.print("(empty database — no analyses)")
        return

    table = Table(title=f"Analyses ({path.name})")
    table.add_column("id", style="cyan")
    table.add_column("status")
    table.add_column("filename")
    table.add_column("hash12")
    table.add_column("analysis")
    table.add_column("created")
    for s in summaries:
        table.add_row(
            str(s["analysis_id"]),
            str(s["status"]),
            str(s["filename"] or "—"),
            str(s["content_hash"] or "")[:12],
            str(s["analysis_version"] or "—"),
            str(s["created_at"] or "—")[:19],
        )
    console.print(table)


@db_app.command("show")
def db_show(
    track_or_analysis_id: str = typer.Argument(..., help="Analysis id or track id"),
    database: Optional[Path] = typer.Option(None, "--database"),
    config: Optional[Path] = typer.Option(None, "--config"),
    as_json: bool = typer.Option(
        False, "--json", help="Emit full fingerprint JSON when available"
    ),
) -> None:
    """Show one analysis (or newest analysis for a track id)."""
    try:
        path = _resolve_database_option(database, config)
        engine = assert_db_usable(path)
        with Session(engine) as session:
            row = resolve_analysis_or_track_id(session, track_or_analysis_id)
            summary = analysis_to_summary_dict(row)
            fp = None
            if row.status == "completed" and row.fingerprint_json:
                fp = fingerprint_from_analysis(row)
    except AnalysisNotFoundError as exc:
        _fail(str(exc), code=1)
    except DatabaseError as exc:
        _fail(str(exc), code=1)
    except NeuroAcousticError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error: {exc}")

    if as_json:
        if fp is not None:
            sys.stdout.write(json.dumps(fp.to_json_dict(), allow_nan=False) + "\n")
        else:
            sys.stdout.write(json.dumps(summary, allow_nan=False) + "\n")
        return

    table = Table(title=f"Analysis {summary['analysis_id']}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key in (
        "analysis_id",
        "track_id",
        "status",
        "content_hash",
        "filename",
        "source_path",
        "duration_seconds",
        "schema_version",
        "analysis_version",
        "vector_version",
        "config_hash",
        "rms",
        "peak_amplitude",
        "centroid_mean_hz",
        "flatness_mean",
        "f0_median_hz",
        "tempo_bpm",
        "stereo_width_estimate",
        "attack_time_s",
        "tail_decay_t60_s",
        "created_at",
        "updated_at",
        "error_message",
    ):
        table.add_row(key, str(summary.get(key) if summary.get(key) is not None else "—"))
    console.print(table)

    arts = summary.get("artifacts") or {}
    art_table = Table(title="Artifacts")
    art_table.add_column("Kind")
    art_table.add_column("Path")
    for kind, apath in arts.items():
        art_table.add_row(kind, str(apath or "—"))
    console.print(art_table)

    missing = summary.get("missing_artifacts") or []
    for kind in missing:
        err_console.print(
            f"[yellow]warning:[/yellow] artifact path missing on disk: {kind}={arts.get(kind)}"
        )


@db_app.command("stats")
def db_stats(
    database: Optional[Path] = typer.Option(None, "--database"),
    config: Optional[Path] = typer.Option(None, "--config"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show database counts and version distribution."""
    try:
        path = _resolve_database_option(database, config)
        engine = assert_db_usable(path)
        with Session(engine) as session:
            stats = compute_stats(session, path)
    except DatabaseError as exc:
        _fail(str(exc), code=1)
    except NeuroAcousticError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error: {exc}")

    payload = {
        "database_path": stats.database_path,
        "database_size_bytes": stats.database_size_bytes,
        "db_schema_version": DB_SCHEMA_VERSION,
        "total_analyses": stats.total_analyses,
        "completed": stats.completed,
        "failed": stats.failed,
        "in_progress": stats.in_progress,
        "unique_content_hashes": stats.unique_content_hashes,
        "schema_versions": stats.schema_versions,
        "analysis_versions": stats.analysis_versions,
    }
    if as_json:
        sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
        return

    table = Table(title="Database stats")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command("index")
def index_cmd(
    directory: Path = typer.Argument(..., help="Root directory to index"),
    database: Optional[Path] = typer.Option(
        None, "--database", help="SQLite database path (required for persistence)"
    ),
    config: Optional[Path] = typer.Option(None, "--config"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    include: Optional[list[str]] = typer.Option(
        None, "--include", help="fnmatch include pattern (repeatable)"
    ),
    exclude: Optional[list[str]] = typer.Option(
        None, "--exclude", help="fnmatch exclude pattern (repeatable)"
    ),
    follow_symlinks: bool = typer.Option(
        False, "--follow-symlinks/--no-follow-symlinks"
    ),
    plots: bool = typer.Option(False, "--plots/--no-plots"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    force: bool = typer.Option(False, "--force"),
    continue_on_error: bool = typer.Option(
        True, "--continue-on-error/--fail-fast"
    ),
    workers: int = typer.Option(1, "--workers", help="Worker threads (1–8; default 1)"),
    report: Optional[Path] = typer.Option(None, "--report", help="Write JSON report path"),
    as_json: bool = typer.Option(False, "--json", help="Emit IndexReport JSON on stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress display"),
    analysis_sample_rate: Optional[int] = typer.Option(None, "--analysis-sample-rate"),
) -> None:
    """Recursively discover and analyze audio files into SQLite."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from neuroacoustic.indexing.indexer import IndexOptions, run_index, validate_workers
    from neuroacoustic.indexing.report import IndexReport

    try:
        validate_workers(workers)
    except ValueError as exc:
        _fail(str(exc))

    try:
        cfg = load_config(config)
        if analysis_sample_rate is not None:
            cfg = cfg.model_copy(
                update={
                    "audio": cfg.audio.model_copy(
                        update={"analysis_sample_rate": analysis_sample_rate}
                    )
                }
            )
        if output_dir is not None:
            cfg = cfg.model_copy(
                update={"paths": cfg.paths.model_copy(update={"output_dir": output_dir})}
            )
        db_path = database if database is not None else cfg.paths.database
        if db_path is None:
            _fail("--database is required (or set paths.database in config)")

        log_level = "WARNING" if quiet or as_json else cfg.logging.level
        setup_logging(log_level, json_logs=cfg.logging.json_logs)

        options = IndexOptions(
            root=directory,
            database=Path(db_path),
            output_dir=Path(output_dir or cfg.paths.output_dir),
            recursive=recursive,
            follow_symlinks=follow_symlinks,
            include=list(include or []),
            exclude=list(exclude or []),
            plots=plots,
            force=force,
            continue_on_error=continue_on_error,
            workers=workers,
            report_path=report,
            quiet=quiet or as_json,
        )

        progress_ctx = None
        task_id = None
        counts = {"analyzed": 0, "reused": 0, "forced": 0, "failed": 0, "skipped": 0}

        def on_progress(completed: int, total: int, item) -> None:  # type: ignore[no-untyped-def]
            counts[item.disposition] = counts.get(item.disposition, 0) + 1
            if progress_ctx is not None and task_id is not None:
                progress_ctx.update(
                    task_id,
                    completed=completed,
                    description=(
                        f"{Path(item.path).name[:40]}  "
                        f"a={counts['analyzed']} r={counts['reused']} "
                        f"f={counts['failed']} s={counts['skipped']}"
                    ),
                )

        if not as_json and not quiet:
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=err_console,
            )
            progress_ctx.start()
            task_id = progress_ctx.add_task("index", total=0)

        try:
            def on_discovered(n: int) -> None:
                if progress_ctx is not None and task_id is not None:
                    progress_ctx.update(task_id, total=n)

            result = run_index(
                cfg,
                options,
                progress_callback=None if (as_json or quiet) else on_progress,
                on_discovered=None if (as_json or quiet) else on_discovered,
            )
        finally:
            if progress_ctx is not None:
                progress_ctx.stop()

        report_obj: IndexReport = result.report
        if as_json:
            sys.stdout.write(
                json.dumps(report_obj.model_dump(mode="json"), allow_nan=False) + "\n"
            )
        else:
            table = Table(title=f"Index: {directory}")
            table.add_column("Field", style="cyan")
            table.add_column("Value")
            for key, value in (
                ("database", report_obj.database_path),
                ("discovered", str(report_obj.discovered)),
                ("supported", str(report_obj.supported)),
                ("analyzed", str(report_obj.analyzed)),
                ("reused", str(report_obj.reused)),
                ("forced", str(report_obj.forced)),
                ("failed", str(report_obj.failed)),
                ("skipped", str(report_obj.skipped)),
                ("elapsed_seconds", str(report_obj.elapsed_seconds)),
                ("config_hash", report_obj.config_hash[:16] + "…"),
                ("report", str(result.report_path or "—")),
                ("interrupted", str(report_obj.interrupted)),
            ):
                table.add_row(key, value)
            console.print(table)
            if report_obj.failed:
                err_console.print(
                    f"[yellow]warning:[/yellow] {report_obj.failed} file(s) failed"
                )

        if result.exit_code:
            raise typer.Exit(result.exit_code)
    except typer.Exit:
        raise
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        _fail(str(exc))
    except NeuroAcousticError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error: {exc}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
