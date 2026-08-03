# Architecture

## Overview

```
audio file
  -> validation and ffprobe metadata
  -> decoding/loading
  -> channel-aware preprocessing (mono mean for analysis)
  -> FFT summary + STFT (in-memory)
  -> spectral + energy features
  -> pitch (pYIN) + harmonic analysis
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
| `neuroacoustic.analysis.pitch` | pYIN F0 / voicing |
| `neuroacoustic.analysis.harmonics` | Harmonic peaks, density, HNR/inharmonicity estimates |
| `neuroacoustic.fingerprint.*` | Pydantic models, builder, preliminary vector |
| `neuroacoustic.visualization.*` | Non-interactive matplotlib PNG plots |
| `neuroacoustic.pipeline` | Orchestrates analyze → JSON/plots |
| `neuroacoustic.cli` | `doctor`, `probe`, `analyze` |

## Milestone 3 processing notes

1. Spectral/energy path unchanged from Milestone 2.
2. `analyze_pitch` runs pYIN on mono audio (same hop as STFT when configured).
3. `analyze_harmonics` maps each voiced F0 frame to the nearest STFT frame and
   aggregates peak lists — **no per-frame harmonic matrix in JSON**.
4. SQLite remains Milestone 4; `--database` is accepted with a warning.

## Safety

- FFmpeg via argument arrays (`shell=False`).
- JSON written with `allow_nan=False`; pipeline rejects non-finite floats.
- Source audio is never overwritten.
- Long-file STFT / pitch frame matrices are not serialized into JSON.
