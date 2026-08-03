# Architecture

## Overview

```
audio file  (or directory via `index`)
  -> validation and metadata probe
  -> decode / channel-aware load
  -> content SHA-256
  -> deterministic scientific config_hash
  -> SQLite lookup (optional): completed (hash, analysis_version, config_hash)
       | hit + not --force -> reuse fingerprint
       | miss / --force    -> full analysis
  -> spectral / energy / pitch / harmonics
  -> envelope / stereo / rhythm / file-tail decay
  -> Pydantic fingerprint + finite JSON check
  -> write artifacts (absolute paths)
  -> transactional SQLite upsert (optional)
```

## Module responsibilities

| Package | Role |
|---------|------|
| `neuroacoustic.config` | TOML + env/CLI overrides |
| `neuroacoustic.audio.*` | Probe, decode, preprocess, quality |
| `neuroacoustic.analysis.*` | Spectral, energy, pitch, harmonics, envelope, stereo, rhythm, reverb |
| `neuroacoustic.fingerprint.*` | Pydantic models, builder, preliminary vector |
| `neuroacoustic.persistence.*` | SQLite engine, ORM, config hash, repository |
| `neuroacoustic.pipeline` | Orchestrates analyze → JSON/plots → optional DB |
| `neuroacoustic.indexing.*` | Deterministic discovery + batch index over `run_pipeline` |
| `neuroacoustic.cli` | `doctor`, `probe`, `analyze`, `index`, `db init\|list\|show\|stats` |
| `neuroacoustic.visualization.*` | Non-interactive matplotlib PNG plots |

## SQLite schema (db_schema_version = 1)

Tables:

- **`schema_meta`** — key/value; includes `db_schema_version`
- **`tracks`** — one row per unique `content_hash` (path/filename informational)
- **`analyses`** — one row per `(content_hash, analysis_version, config_hash)`

Important indexes: `content_hash`; identity triple; `created_at`; `status`;
selected scalars (`centroid_mean_hz`, `flatness_mean`).

Pragmas: `foreign_keys=ON`, `busy_timeout=5000`, `journal_mode=WAL`.

Incompatible `db_schema_version` → hard error (no silent rebuild / destructive migration).

## Cache / config-hash semantics

Identity of a completed analysis:

```
content_hash + analysis_version + config_hash
```

`config_hash` = SHA-256 of canonical JSON from
`neuroacoustic.persistence.config_hash.scientific_config_payload`:

**Included:** analysis sample rate; silence/clipping thresholds that enter the
fingerprint; all `AnalysisConfig` algorithm fields (`n_fft`, hop, pitch,
harmonics, envelope, tempo gates, decay gates, `vector_version`, …).

**Excluded:** `paths.*`, `logging.*`, project name, `config_path`, probe
preference, `--plots` / `--output-dir` / display flags.

Failed and in-progress rows are never returned for reuse.

### `--force` semantics

1. Bypasses the max-duration guard.
2. When `--database` is set: skips cache reuse and re-runs analysis.
3. **Only after a successful re-analysis** replaces the existing completed row
   for the same identity in place (same primary key). Disposition: `forced`.
4. If a forced re-analysis **fails**, the previous completed fingerprint remains
   usable — `record_failed_analysis` refuses to overwrite a completed row.
   No partial overwrite of fingerprint JSON / vector columns occurs.

## Presentation artifacts vs scientific cache

`--plots`, `--output-dir`, and logging are **excluded** from `config_hash`.
On cache reuse:

- The scientific fingerprint is returned unchanged (same analysis id).
- Fingerprint JSON is re-materialized into the requested output directory when
  needed (or a copy is written with a warning if the original path differs).
- If plots are requested and missing on disk, they are regenerated from the
  loaded audio without creating a new analysis identity; regenerated paths are
  stored on the analysis row.
- If plots remain only under a previous directory, a warning lists the original
  location.
- The CLI never claims plots were created when files are absent.

## Track path semantics (MVP)

`tracks` stores **first-seen** `source_path` / `filename` for a content hash.
Re-analyzing identical bytes under another name updates `updated_at` (last-seen
activity) but does not overwrite the canonical path. Only one path is retained;
a multi-path history is a later enhancement.

## Paths and portability

- Artifact paths persisted as **absolute** resolved paths.
- Source audio is never copied into the database.
- Missing artifact files produce warnings on `db show`, not hard failures.
- FFmpeg always via argument arrays (`shell=False`).
- `db stats` size includes the main DB file plus `-wal` / `-shm` when present.

## Persistence flow (`analyze --database`)

1. Probe/hash input  
2. Compute `config_hash`  
3. Query completed analysis for the identity  
4. Reuse when eligible (plus presentation artifact refresh)  
5. Otherwise analyze  
6. Pydantic-validate + reject NaN/Inf  
7. Write artifacts  
8. Transactional persist (replace only on successful `--force`)  
9. Print disposition: `analyzed` | `reused` | `forced`  

Without `--database`, JSON/plots only (backward compatible).

## Batch indexing (`index DIRECTORY`)

```
directory
  -> pathlib discovery (sorted, extension filter, include/exclude)
  -> skip output dir, DB/WAL/SHM, hidden/temp, unsupported types
  -> for each file: run_pipeline(..., database=...)
  -> dispositions: analyzed | reused | forced | failed | skipped
  -> atomic IndexReport JSON (--report) and optional --json stdout
```

Discovery defaults: recursive on; **do not follow symlinks**; case-insensitive
extensions `{wav,wave,flac,aiff,aif,mp3}`; deterministic casefold sort;
realpath dedupe. Symlink directory loops are blocked via a visited-realpath set
when `--follow-symlinks` is enabled.

**Counters:** `discovered` = supported-extension candidates after structural
skips (not hidden/temp, not under resolved output/skip dirs, not skip files,
symlink policy, deduped). Include/exclude are applied next; `supported` =
accepted queue (`len(files)`). Unsupported types (`.txt`, `.json`, `.png`, …)
never enter either counter. Excluded-by-pattern files increment `discovered`
only.

**Artifacts:** `{output}/by_hash/{hh}/{content_hash}_{safe_stem}_*` so same
basenames with different bytes cannot collide. Writes use temp + `os.replace`
where practical (fingerprint JSON, plots, index reports).

**Source integrity:** after analysis (and before persist), size/mtime is checked
against the post-load signature; a re-hash runs only if metadata changed. A
content mismatch raises `file_changed_during_analysis` and refuses to persist a
completed fingerprint under the wrong identity.

**Resume:** there is no separate resume database — completed SQLite analyses are
the cache. Identical second runs should show `reused` for unchanged scientific
identities.

**Concurrency:** `--workers` default **1** (max 8). Prefer workers=1 for first
production runs. Threads only; each task owns its pipeline/DB session. Duplicate
content races recover via `IntegrityError` → reuse completed winner. When
workers > 1, BLAS/OpenMP env caps are `setdefault`'d to 1 but only help if set
**before** NumPy import/init — the CLI cannot guarantee that after load.

**Interruption:** SIGINT sets a stop flag, stops scheduling new files, lets
in-flight work finish or skip, writes a consistent partial report, preserves
completed rows, exits **130**.

**Exit codes:** `0` complete without failures; `2` complete with per-file
failures (continue-on-error); `1` fail-fast / fatal batch failure; `130`
interrupted; other non-zero = invalid invocation.

### IndexReport schema (summary)

Top-level fields include `root_directory`, `database_path`, `started_at`,
`finished_at`, `elapsed_seconds`, discovery/worker flags, counters
(`discovered`, `supported`, `analyzed`, `reused`, `forced`, `failed`,
`skipped`), version strings, `config_hash`, and `files[]` with per-file
`path`, `disposition`, `analysis_id`, `duration_seconds`, `error`, `warnings`.
Public reports omit stack traces.

## Future vector search

SQLite remains the local system of record for fingerprints and metadata.
A later vector database (e.g. Qdrant) will index embeddings without making
SQLite rows disposable. Batch indexing does **not** implement similarity search.

## Safety

- JSON written with `allow_nan=False`; pipeline rejects non-finite floats.
- Uniqueness enforced in SQLite (`uq_analysis_identity`); insert races catch
  `IntegrityError` and re-fetch only a **completed** winning row.
- Transactions roll back on error; sessions are closed in `session_scope`.
- Source audio is never overwritten.
- Indexing never uses `shell=True`; paths are never interpolated into shells.
- Output directory and database files are excluded from discovery self-scans.
