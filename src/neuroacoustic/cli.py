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
    add_row(
        "python",
        "ok" if sys.version_info >= (3, 11) else "fail",
        platform.python_version(),
    )
    add_row("platform", "ok", f"{platform.system()} {platform.machine()}")

    for mod in ("numpy", "scipy", "librosa", "soundfile", "pydantic", "typer", "rich"):
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
    database: Optional[Path] = typer.Option(
        None,
        "--database",
        help="SQLite path (ignored in Milestone 2; persistence arrives in Milestone 4)",
    ),
    config: Optional[Path] = typer.Option(None, "--config"),
    analysis_sample_rate: Optional[int] = typer.Option(None, "--analysis-sample-rate"),
    plots: bool = typer.Option(
        True,
        "--plots/--no-plots",
        help="Generate waveform / FFT / spectrogram PNG plots",
    ),
    force: bool = typer.Option(False, "--force", help="Allow files longer than max_duration"),
    as_json: bool = typer.Option(False, "--json", help="Print fingerprint JSON to stdout"),
) -> None:
    """Run Milestone 2 analysis, write fingerprint JSON, and optional plots."""
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
        if database is not None:
            err_console.print(
                "[yellow]warning:[/yellow] --database is accepted but SQLite "
                "persistence is deferred to Milestone 4; writing JSON only."
            )
        setup_logging(cfg.logging.level, json_logs=cfg.logging.json_logs)
        result = run_pipeline(
            audio_file,
            cfg,
            output_dir=cfg.paths.output_dir,
            plots=plots,
            force=force,
        )
    except NeuroAcousticError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error: {exc}")

    fp = result.fingerprint
    if as_json:
        console.print_json(json.dumps(fp.to_json_dict()))
        return

    table = Table(title=f"Analyze: {fp.source.filename}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    rows = [
        ("fingerprint", str(result.fingerprint_path)),
        ("schema_version", fp.schema_version),
        ("analysis_version", fp.analysis_version),
        ("content_hash", fp.source.content_hash),
        ("duration_seconds", str(fp.source.duration_seconds)),
        ("analysis_sample_rate", str(fp.source.analysis_sample_rate)),
        ("fft_peak_hz", str(fp.spectral.fft.peak_frequency_hz)),
        ("centroid_mean_hz", str(fp.spectral.centroid_hz.mean)),
        ("flatness_mean", str(fp.spectral.flatness.mean)),
        ("entropy_mean", str(fp.spectral.entropy.mean)),
        ("rms", str(fp.energy.rms)),
        ("peak_amplitude", str(fp.energy.peak_amplitude)),
        ("crest_factor", str(fp.energy.crest_factor)),
        ("zcr", str(fp.energy.zero_crossing_rate)),
        ("estimated_dynamic_range_db", str(fp.energy.estimated_dynamic_range_db)),
        ("vector_len", str(len(fp.vector))),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)
    for warning in fp.quality.warnings:
        err_console.print(f"[yellow]warning:[/yellow] {warning}")
    if fp.artifacts.waveform_png:
        console.print(f"plots: {fp.artifacts.waveform_png}")


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
