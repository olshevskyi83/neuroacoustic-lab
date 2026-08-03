# Architecture

## Overview

NeuroAcoustic Lab is a modular audio analysis pipeline:

```
audio file
  -> validation and ffprobe metadata
  -> decoding/loading
  -> channel-aware preprocessing
  -> frame-level acoustic analysis      (Milestone 2+)
  -> time aggregation                   (Milestone 2+)
  -> fingerprint construction           (Milestone 2+)
  -> JSON export
  -> SQLite persistence                 (Milestone 4)
  -> optional plot generation           (Milestone 4)
```

## Module responsibilities

| Package | Role |
|---------|------|
| `neuroacoustic.config` | Load TOML config; apply environment / CLI overrides |
| `neuroacoustic.logging` | Structured application logging |
| `neuroacoustic.exceptions` | Typed domain errors for CLI exit handling |
| `neuroacoustic.audio.probe` | File validation, content hash, container metadata |
| `neuroacoustic.audio.loader` | Decode via soundfile or FFmpeg fallback |
| `neuroacoustic.audio.preprocessing` | Channel handling, optional resample, float conversion |
| `neuroacoustic.analysis.*` | Feature extractors (Milestone 2+) |
| `neuroacoustic.fingerprint` | Pydantic models and fingerprint builder |
| `neuroacoustic.persistence` | SQLite storage (Milestone 4) |
| `neuroacoustic.visualization` | Non-interactive matplotlib plots (Milestone 4) |
| `neuroacoustic.pipeline` | End-to-end orchestration (Milestone 2+) |
| `neuroacoustic.cli` | Typer CLI (`doctor`, `probe`, …) |

## Milestone 1 processing flow

1. Resolve config from `config/default.toml` (+ env / CLI).
2. `probe`: validate path/extension → hash file → gather metadata via soundfile and/or ffprobe → return `SourceMetadata`.
3. `load` (used by tests / later analyze): decode samples → float32 in `[-1, 1]` (when integer source) → optional resample → basic quality checks (NaN/Inf, silence).

## Persistence flow (planned)

Milestone 4 will store track identity, content hash, schema/analysis versions, config hash, scalar columns, full fingerprint JSON, artifact paths, status, and timestamps. Duplicate analyses (same content + analysis version + config) will be detectable.

## Safety

- Never execute filenames or metadata as shell code.
- FFmpeg is invoked with argument arrays (`shell=False`).
- Source audio is never overwritten.
