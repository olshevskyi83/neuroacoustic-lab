# Metrics

Definitions, formulas, units, ranges, aggregation methods, and limitations.

## Source / probe metadata

| Field | Meaning | Unit | Notes |
|-------|---------|------|-------|
| `content_hash` | SHA-256 of file bytes | hex string | Exact content identity |
| `duration_seconds` | Media duration | s | From header or samples/rate |
| `native_sample_rate` | Source sampling rate | Hz | Preserved when practical |
| `analysis_sample_rate` | Rate used for analysis | Hz | Equals native unless overridden |
| `channels` | Channel count | count | Spectral/energy use mono mean |
| `bit_depth` / `bitrate` | Encoding metadata | bits / bit·s⁻¹ | May be null |

## Quality

| Field | Meaning | Range | Limitations |
|-------|---------|-------|-------------|
| `clipping_ratio` | Fraction \|x\| ≥ clipping threshold | 0–1 | Threshold heuristic |
| `silence_ratio` | Fraction \|x\| < silence sample threshold | 0–1 | Near-zero count |
| `peak_amplitude` | max\|x\| after float conversion | ≥0 linear | Not loudness |

## Spectral (Milestone 2)

Frame features use magnitude STFT `S[f,t]=|X[f,t]|` with Hann window, configurable `n_fft` / `hop_length` / `win_length`. Aggregation: mean, std, median, p05/p25/p75/p95, min, max over frames. **The full STFT matrix is never stored in JSON.**

| Metric | Formula / algorithm | Unit | Expected range | Limitations |
|--------|---------------------|------|----------------|-------------|
| FFT peak | argmax \|RFFT(x)\| | Hz | 0–Nyquist | Leakage, windowing; not pitch tracking |
| Spectral centroid | `Σ f S / Σ S` per frame | Hz | 0–Nyquist | Amplitude-weighted; not F0 |
| Spectral bandwidth | `√(Σ (f−C)² S / Σ S)` | Hz | ≥0 | Same weighting as centroid |
| Spectral rolloff | freq below which fraction `ρ` of `Σ S²` lies (default ρ=0.85) | Hz | 0–Nyquist | Sensitive to ρ |
| Spectral flatness | geometric mean / arithmetic mean of `S+ε` | 1 | ~0–1 | Floor `ε` affects silence |
| Spectral entropy | normalized Shannon entropy of power `S²` | 1 | ~0–1 | Normalized by `log(N_bins)` |
| Spectral contrast | librosa band peak−valley | dB | typically tens of dB | Band edges depend on SR |

## Energy / time domain (Milestone 2)

| Metric | Formula | Unit | Limitations |
|--------|---------|------|-------------|
| RMS | `√mean(x²)` | linear amplitude | Not loudness / not LUFS |
| `rms_db` | `20 log10(max(RMS, floor))` | dB re FS-ish | **Not LUFS** |
| Peak amplitude | `max\|x\|` | linear | Not True Peak |
| Crest factor | `peak / max(RMS, floor)` | 1 | Also as `crest_factor_db` |
| Zero-crossing rate | librosa frame ZCR, mean | crossings/sample | Sensitive to noise/DC |
| **Estimated** dynamic range | `20 log10(RMS_p95 / RMS_p05)` | dB | **Heuristic estimate only** — not LUFS, not AES/EBU loudness, not broadcast DR |

## Preliminary vector

Version `0.2.0-preliminary`: heuristically scaled scalars for experimentation. Dataset-level normalization is deferred until a real corpus exists. Do not treat as a calibrated similarity embedding.

## Reserved (later milestones)

Pitch, harmonics, envelope, stereo, rhythm, reverberation.
