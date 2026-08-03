# NeuroAcoustic Lab

Local-first research platform for analyzing sound by its **physical acoustic properties** — not genre, artist, album, or musical style.

> **Current status:** Milestone 4B — SQLite persistence, duplicate-analysis detection, and database CLI. Envelope/stereo/rhythm/decay (4A) are included. No web UI or vector search yet.

## Philosophy

- Measure what is physically present in the signal.
- Prefer deterministic, testable signal-processing algorithms.
- Name estimates as estimates; never present heuristics as exact physical measurements.
- Keep analysis parameters, schema versions, and content hashes so results are reproducible.

## Requirements

- **Python 3.11+** (developed against 3.12)
- **FFmpeg / ffprobe** for MP3; WAV/FLAC/AIFF work via `soundfile`
- Local virtual environment (do not install packages globally)

```bash
sudo apt update && sudo apt install -y ffmpeg   # if needed
cd /home/homelabuser/projects/neuroacoustic-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Example commands

```bash
neuroacoustic doctor
neuroacoustic probe path/to/audio.wav
neuroacoustic probe path/to/audio.wav --json

# Analyze without database (JSON + plots only)
neuroacoustic analyze path/to/audio.wav --output-dir data/output --no-plots

# Initialize SQLite (idempotent) and persist analyses
neuroacoustic db init --database data/db/neuroacoustic.sqlite
neuroacoustic analyze path/to/audio.wav --database data/db/neuroacoustic.sqlite --output-dir data/output

# Second run of the same bytes reuses the completed row
neuroacoustic analyze path/to/audio.wav --database data/db/neuroacoustic.sqlite --output-dir data/output

# Force re-analysis: replaces the same identity row in place
neuroacoustic analyze path/to/audio.wav --database data/db/neuroacoustic.sqlite --force --no-plots

neuroacoustic db list --database data/db/neuroacoustic.sqlite
neuroacoustic db show 1 --database data/db/neuroacoustic.sqlite
neuroacoustic db show 1 --database data/db/neuroacoustic.sqlite --json
neuroacoustic db stats --database data/db/neuroacoustic.sqlite
```

## Persistence and cache identity

Database location defaults to `data/db/neuroacoustic.sqlite` (config `paths.database`).
SQLite files, WAL, and SHM sidecars are **gitignored** — never commit them.

**Cache identity** (one completed analysis):

```
content SHA-256  +  analysis_version  +  config_hash
```

- Path/filename are informational only (same bytes under another name → reuse).
- `config_hash` is SHA-256 of a canonical JSON of **scientific** parameters
  (STFT/pitch/envelope/… settings + analysis sample rate). Output directories,
  logging, and CLI display flags are excluded. See `docs/architecture.md`.
- Failed / incomplete rows are **never** reused.
- **`--force`**: allows long files **and**, when `--database` is set, re-runs
  analysis. The completed row for that identity is **replaced only after a
  successful** re-analysis (same primary key). If the forced run fails, the
  previous completed result remains usable.
- **Track paths**: first-seen path/filename are retained for a content hash;
  later filenames update `updated_at` only (MVP limitation: one path stored).
- **Artifacts**: `--plots` / `--output-dir` do not change cache identity. On
  reuse, missing requested plots may be regenerated; otherwise warnings name
  absent or original-location artifacts.

SQLite is appropriate for the local MVP. Later vector search will use a
separate vector database; SQLite fingerprints remain the system of record
for metadata and full JSON.

### Backup

Copy the `.sqlite` file (and `-wal`/`-shm` if present, or checkpoint first) to
back up the catalog. Schema version is stored in `schema_meta`; incompatible
versions refuse to open rather than silently migrating or deleting data.

## Output locations

| Path | Purpose |
|------|---------|
| `data/input/` | Source audio (not tracked) |
| `data/output/` | Fingerprint JSON and plots |
| `data/db/` | SQLite database (not tracked) |
| `config/default.toml` | Default analysis configuration |

Artifact paths in the DB are stored as **absolute** resolved paths.

## Tests

```bash
source .venv/bin/activate
python scripts/generate_test_signals.py
pytest
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/metrics.md`](docs/metrics.md)
- [`docs/fingerprint-schema.md`](docs/fingerprint-schema.md)
- [`PROJECT_SPEC.md`](PROJECT_SPEC.md)

## Current limitations

- No web UI, similarity search, or Qdrant.
- No destructive automatic schema migrations (incompatible DB → clear error).
- Preliminary vector is heuristically scaled — not corpus-normalized.
- Pitch is single-F0; ADSR/tempo/T60 fields are estimates with documented limits.
- `estimated_dynamic_range_db` is **not LUFS**.
