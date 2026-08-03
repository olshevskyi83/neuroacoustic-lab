"""CLI entry points for NeuroAcoustic Lab."""

from __future__ import annotations

import importlib.metadata
import json
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

app = typer.Typer(
    name="neuroacoustic",
    help="Local-first physical acoustic analysis of sound.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="SQLite database commands.")
app.add_typer(db_app, name="db")

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
