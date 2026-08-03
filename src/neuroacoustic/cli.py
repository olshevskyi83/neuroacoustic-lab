"""CLI entry points for NeuroAcoustic Lab."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from neuroacoustic import __analysis_version__, __schema_version__, __version__
from neuroacoustic.audio.probe import ffmpeg_available, ffprobe_available, probe_audio
from neuroacoustic.config import load_config
from neuroacoustic.exceptions import NeuroAcousticError
from neuroacoustic.logging import setup_logging

app = typer.Typer(
    name="neuroacoustic",
    help="Local-first physical acoustic analysis of sound.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="SQLite database commands (Milestone 4).")
app.add_typer(db_app, name="db")

console = Console(stderr=False)
err_console = Console(stderr=True)


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def _check_python_import(module: str) -> tuple[bool, str]:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        return True, str(version)
    except Exception as exc:  # noqa: BLE001 — doctor must never crash on import probe
        return False, str(exc)


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

    table.add_row("neuroacoustic", "ok", f"v{__version__}")
    table.add_row("schema_version", "ok", __schema_version__)
    table.add_row("analysis_version", "ok", __analysis_version__)
    table.add_row(
        "python",
        "ok" if sys.version_info >= (3, 11) else "fail",
        platform.python_version(),
    )
    table.add_row("platform", "ok", f"{platform.system()} {platform.machine()}")

    for mod in ("numpy", "scipy", "librosa", "soundfile", "pydantic", "typer", "rich"):
        ok, detail = _check_python_import(mod)
        table.add_row(f"py:{mod}", "ok" if ok else "fail", detail)

    table.add_row(
        "ffmpeg",
        "ok" if ffmpeg_available() else "missing",
        shutil.which("ffmpeg") or "not on PATH — required for MP3 decode",
    )
    table.add_row(
        "ffprobe",
        "ok" if ffprobe_available() else "missing",
        shutil.which("ffprobe") or "not on PATH — required for MP3 probe",
    )

    cfg_path = cfg.config_path
    table.add_row(
        "config",
        "ok" if cfg_path and Path(cfg_path).exists() else "warn",
        str(cfg_path),
    )
    table.add_row("output_dir", "ok", str(cfg.paths.output_dir))
    table.add_row("database", "ok", str(cfg.paths.database))

    console.print(table)

    missing_ffmpeg = not ffmpeg_available() or not ffprobe_available()
    if missing_ffmpeg:
        err_console.print(
            "[yellow]warning:[/yellow] FFmpeg/ffprobe missing. "
            "WAV/FLAC/AIFF work via soundfile; MP3 requires FFmpeg."
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
        console.print_json(json.dumps(result.to_printable_dict()))
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
    database: Optional[Path] = typer.Option(None, "--database"),
    config: Optional[Path] = typer.Option(None, "--config"),
    analysis_sample_rate: Optional[int] = typer.Option(None, "--analysis-sample-rate"),
    plots: bool = typer.Option(False, "--plots/--no-plots", help="Generate plots (Milestone 4)"),
    force: bool = typer.Option(False, "--force"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Run full analysis (Milestone 2+). Not available in Milestone 1."""
    _ = (audio_file, output_dir, database, config, analysis_sample_rate, plots, force, as_json)
    _fail(
        "analyze is not implemented in Milestone 1. "
        "Use `neuroacoustic probe` for metadata. Full fingerprinting arrives in Milestone 2.",
        code=2,
    )


@db_app.command("list")
def db_list() -> None:
    """List analyzed tracks (Milestone 4)."""
    _fail("db list requires SQLite persistence (Milestone 4).", code=2)


@db_app.command("show")
def db_show(track_id: str = typer.Argument(..., help="Track id")) -> None:
    """Show one stored fingerprint (Milestone 4)."""
    _ = track_id
    _fail("db show requires SQLite persistence (Milestone 4).", code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
