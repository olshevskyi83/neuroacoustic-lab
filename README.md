# NeuroAcoustic Lab

Local-first research platform for analyzing sound by its **physical acoustic properties** — not genre, artist, album, or musical style.

The first milestone is a reliable **Audio Analysis Engine** that turns an audio file into a reproducible acoustic fingerprint.

> **Current status:** Milestone 2 — spectral + energy fingerprinting, JSON export, and PNG plots. Pitch/harmonics (M3) and SQLite/persistence extras (M4) are not implemented yet.

## Philosophy

- Measure what is physically present in the signal (spectrum, energy, pitch estimates, stereo geometry, …).
- Prefer deterministic, testable signal-processing algorithms.
- Name estimates as estimates; never present heuristics as exact physical measurements.
- Keep analysis parameters, schema versions, and content hashes so results are reproducible and comparable.

## Requirements

- **Python 3.11+** (developed against 3.12)
- **FFmpeg / ffprobe** (system packages) — required for MP3 and rich container metadata; WAV/FLAC/AIFF work without them via `soundfile`
- Local virtual environment (do not install packages globally)

### Install FFmpeg (system)

On Debian/Ubuntu (requires your permission / `sudo`):

```bash
sudo apt update && sudo apt install -y ffmpeg
```

Verify:

```bash
ffmpeg -version
ffprobe -version
```

## Setup

```bash
cd /home/homelabuser/projects/neuroacoustic-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # optional
```

## Example commands

```bash
# Environment and dependency status
neuroacoustic doctor

# Source metadata only (no full analysis)
neuroacoustic probe path/to/audio.wav

# Machine-readable probe output
neuroacoustic probe path/to/audio.wav --json

# Milestone 2 analysis → fingerprint JSON (+ plots by default)
neuroacoustic analyze path/to/audio.wav --output-dir data/output
neuroacoustic analyze path/to/audio.wav --no-plots --json
```

`db list` / `db show` remain stubbed until Milestone 4.

## Output locations

| Path | Purpose |
|------|---------|
| `data/input/` | Place source audio (not tracked by git) |
| `data/output/` | Fingerprint JSON and plots (Milestone 2+) |
| `data/db/` | SQLite database (Milestone 4; not tracked) |
| `config/default.toml` | Default analysis configuration |

## JSON fingerprint purpose

A versioned Pydantic document capturing source metadata, analysis config, quality warnings, acoustic feature summaries, and a preliminary comparison vector. Schema details: [`docs/fingerprint-schema.md`](docs/fingerprint-schema.md).

## Current limitations (Milestone 2)

- Pitch, harmonics, envelope, stereo, rhythm, and reverb are empty placeholders.
- No SQLite persistence yet (`--database` ignored with a warning).
- Preliminary comparison vector is heuristically scaled — not corpus-normalized.
- `estimated_dynamic_range_db` is a frame-RMS percentile heuristic, **not LUFS**.
- Spectral/energy features use a mono mixdown of multi-channel files.
- MP3 decoding requires FFmpeg.

## Tests

```bash
source .venv/bin/activate
# Generate synthetic fixtures used by tests
python scripts/generate_test_signals.py
pytest
```

## Documentation

- [`PROJECT_SPEC.md`](PROJECT_SPEC.md) — full project specification
- [`docs/architecture.md`](docs/architecture.md) — modules and pipeline
- [`docs/metrics.md`](docs/metrics.md) — metric definitions (filled as features land)
- [`docs/fingerprint-schema.md`](docs/fingerprint-schema.md) — schema versioning

## License

MIT (see `pyproject.toml`).
