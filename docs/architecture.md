# Architecture

## Overview

```
audio file
  -> validation and ffprobe metadata
  -> decoding/loading
  -> channel-aware preprocessing (mono mean for M2 features)
  -> FFT summary + STFT (in-memory)
  -> spectral + energy features
  -> fingerprint construction (summaries only)
  -> JSON export + optional PNG plots
  -> SQLite persistence                 (Milestone 4)
```

## Module responsibilities

| Package | Role |
|---------|------|
| `neuroacoustic.config` | TOML + env/CLI overrides |
| `neuroacoustic.audio.*` | Probe, decode, float convert, resample, quality |
| `neuroacoustic.analysis.spectrum` | FFT summary, STFT |
| `neuroacoustic.analysis.spectral` | Centroid, bandwidth, rolloff, contrast, flatness, entropy |
| `neuroacoustic.analysis.energy` | RMS, ZCR, peak, crest, estimated dynamic range |
| `neuroacoustic.fingerprint.*` | Pydantic models, builder, preliminary vector |
| `neuroacoustic.visualization.*` | Non-interactive matplotlib PNG plots |
| `neuroacoustic.pipeline` | Orchestrates analyze → JSON/plots |
| `neuroacoustic.cli` | `doctor`, `probe`, `analyze` |

## Milestone 2 persistence flow

1. `analyze` loads config and audio.
2. Pipeline writes `{stem}_{hash12}_fingerprint.json` under `--output-dir`.
3. Optional plots: waveform, FFT, log-frequency FFT, spectrogram PNGs.
4. SQLite is **not** written yet (Milestone 4); `--database` is accepted with a warning.

## Safety

- FFmpeg via argument arrays (`shell=False`).
- JSON written with `allow_nan=False`; pipeline rejects non-finite floats.
- Source audio is never overwritten.
- Long-file STFT is frame-aggregated; raw matrices are not serialized.
