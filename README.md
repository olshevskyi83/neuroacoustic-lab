# NeuroAcoustic Lab

Local-first research platform for analyzing sound by its **physical acoustic properties** — not genre, artist, album, or musical style.

> **Current status:** Milestone 4C — batch directory indexing into SQLite. Envelope/stereo/rhythm/decay (4A) and persistence (4B) included. No web UI or vector search yet.

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

### Batch indexing (recommended first production run)

Supported extensions (case-insensitive): `.wav`, `.wave`, `.flac`, `.aiff`, `.aif`, `.mp3`.

```bash
neuroacoustic index /path/to/audio \
  --database data/db/neuroacoustic.sqlite \
  --output-dir data/output \
  --recursive \
  --no-plots \
  --workers 1 \
  --report data/output/index-report.json
```

Useful options:

| Option | Notes |
|--------|--------|
| `--recursive` / `--no-recursive` | Default recursive |
| `--include` / `--exclude` | Repeatable fnmatch patterns |
| `--follow-symlinks` / `--no-follow-symlinks` | Default **no-follow** |
| `--plots` / `--no-plots` | Plots are presentation-only (not part of scientific identity) |
| `--force` | Re-analyze identities (same rules as `analyze --force`) |
| `--continue-on-error` / `--fail-fast` | Default continue; fail-fast stops scheduling after first failure |
| `--workers N` | Default **1**; max 8. Prefer 1 until you need throughput |
| `--report PATH` | Atomic JSON summary report |
| `--json` | Machine-readable `IndexReport` on stdout only (logs on stderr) |
| `--quiet` | Suppress Rich progress |

**Cache / resume:** SQLite is the resume state. A second identical index run reuses completed analyses (`reused`). Changed scientific config → new analysis identities. Changed plot/output settings do not. Missing requested plots follow the same artifact-regeneration rules as `analyze`.

**Failures:** One corrupt or unreadable file is recorded as `failed` and does not roll back other completed rows (`--continue-on-error`). With `--fail-fast`, remaining files are `skipped` and the process exits non-zero. Ctrl+C stops scheduling new files, preserves completed DB rows, and exits with status 130.

**Workers:** Default and recommended for first production runs: `--workers 1`.
Each worker runs the existing single-file pipeline with its own DB sessions (no
shared SQLAlchemy Session). When `workers > 1`, the indexer *attempts* to set
BLAS/OpenMP thread env vars to 1 via `setdefault`, but those caps are only
reliable if set **before** NumPy is imported — do not assume they always apply
after the CLI has already loaded scientific libraries.

**Discovery counters:** `discovered` = candidate files with a supported
extension after structural skips (hidden/temp, resolved output/DB paths,
symlink policy). `supported` = candidates accepted after include/exclude
(= files queued). Unsupported extensions, JSON/PNG reports, and paths under the
resolved output directory are not candidates.

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
back up the catalog. Before large index runs, copy the database directory.
Schema version is stored in `schema_meta`; incompatible versions refuse to open
rather than silently migrating or deleting data.

## Output locations

| Path | Purpose |
|------|---------|
| `data/input/` | Source audio (not tracked) |
| `data/output/` | Fingerprint JSON, plots, index reports |
| `data/db/` | SQLite database (not tracked) |
| `config/default.toml` | Default analysis configuration |

Artifact paths in the DB are stored as **absolute** resolved paths.

## Index report format

Reports (and `--json` stdout) are `IndexReport` JSON: root directory, database
path, timestamps, elapsed seconds, **`discovered`** (supported-extension
candidates after structural skips) vs **`supported`** (accepted after
include/exclude), analyzed/reused/forced/failed/skipped tallies, per-file
results (path, disposition, analysis id, duration, error), schema/analysis/vector
versions, and `config_hash`. Written atomically (temp file + replace). Stack
traces are not included in the public report. Disposition counters must equal
`len(files)` / `supported`.

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
- Batch workers share one process (threads); default concurrency is 1.
