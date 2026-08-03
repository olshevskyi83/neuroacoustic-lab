# Architecture

## Overview

```
audio file
  -> validation and ffprobe metadata
  -> decoding/loading
  -> channel-aware preprocessing (preserve channels; mono mean for most analyses)
  -> FFT summary + STFT (in-memory)
  -> spectral + energy features
  -> pitch (pYIN) + harmonic analysis
  -> envelope / stereo / rhythm / file-tail decay (Milestone 4A)
  -> fingerprint construction (summaries only)
  -> JSON export + optional PNG plots
  -> SQLite persistence                 (later; not 4A)
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
| `neuroacoustic.analysis.harmonics` | Harmonic peaks, density, energy-fraction / inharmonicity |
| `neuroacoustic.analysis.envelope` | Smoothed RMS envelope + ADSR-like estimates |
| `neuroacoustic.analysis.stereo` | L/R RMS, correlation, mid/side, width estimate |
| `neuroacoustic.analysis.rhythm` | Onset strength, optional tempo/beats |
| `neuroacoustic.analysis.reverb` | File-tail exponential decay heuristic |
| `neuroacoustic.fingerprint.*` | Pydantic models, builder, preliminary vector |
| `neuroacoustic.visualization.*` | Non-interactive matplotlib PNG plots |
| `neuroacoustic.pipeline` | Orchestrates analyze → JSON/plots |
| `neuroacoustic.cli` | `doctor`, `probe`, `analyze` |

## Milestone 4A processing notes

1. Spectral/energy/pitch/harmonics path unchanged from Milestone 3 (vector 0–21 stable).
2. Envelope, rhythm, and decay run on mono; stereo uses preserved multi-channel samples.
3. Tempo and T60-like fields are **null** when evidence is weak — never fabricated.
4. SQLite remains deferred; `--database` is accepted with a warning.

## Safety

- FFmpeg via argument arrays (`shell=False`).
- JSON written with `allow_nan=False`; pipeline rejects non-finite floats.
- Source audio is never overwritten.
- Long-file matrices are not serialized into JSON.
