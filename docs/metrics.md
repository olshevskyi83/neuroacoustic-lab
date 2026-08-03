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

Version `0.3.0-preliminary`:

- Indices **0–11**: Milestone 2 spectral/energy prefix (unchanged labels/scaling).
- Indices **12–21**: Milestone 3 pitch/harmonic appends (`voiced_ratio`, `f0_median_norm`, …).
See `neuroacoustic.fingerprint.normalization` for the exact ordering and scales.

Unavailable pitch/harmonics contribute **0.0** in the vector; F0 aggregates in the
fingerprint body use **null**, never a fabricated 0 Hz pitch.

## Pitch (Milestone 3)

Algorithm: librosa **pYIN** (`librosa.pyin`) with configurable `f0_min_hz` / `f0_max_hz`.

| Metric | Meaning | Unit | Aggregation | Limitations |
|--------|---------|------|-------------|-------------|
| `f0_median_hz` | Median voiced F0 | Hz | voiced frames only | **null** if none; never 0 Hz sentinel |
| `f0_voiced_hz.*` | Voiced F0 distribution | Hz | mean/std/percentiles | Single-F0 tracker |
| `voiced_ratio` | Usable voiced frames / frames | 1 | — | Thresholded by `voiced_prob_threshold` |
| `mean_voiced_probability` | Mean pYIN voiced prob on usable frames | 1 | — | Not calibrated confidence |
| `confidence` | Heuristic from voicing × coverage | 0–1 | — | Not a formal posterior |
| `f0_curve` | Downsampled F0 | Hz / null | — | Unvoiced → JSON `null` |

**Polyphonic / mixed audio:** pYIN estimates **one** F0 trajectory. It does **not**
represent all concurrent sources; octave errors and source switching can occur.

## Harmonics (Milestone 3)

Peaks near `n·F0` in the magnitude STFT (parabolic bin interpolation), relative
to the local fundamental peak.

| Metric | Definition | Unit | Notes |
|--------|------------|------|-------|
| `peak_frequencies_hz` | Median interpolated peak freq per harmonic index | Hz | null if undetected |
| `relative_amplitudes` | Median `mag_n / mag_1` | 1 | Fundamental → 1 |
| `harmonic_count` | Mean detected count over voiced frames | count | — |
| `harmonic_slots_available` | `min(max_harmonics, ⌊(Nyquist−ε)/F0⌋)` | count | Density denominator |
| `harmonic_density` | `mean(count / slots_available)` | 0–1 | Slot occupancy, **not** harmonics/Hz. Pure sine ≈ `1/12` when `max_harmonics=12` and F0≪Nyquist |
| `normalized_distribution` | Renormalized relative-amp vector | 1 | Length `max_harmonics` |
| `harmonic_energy_fraction_estimate` | `E_harm / E_band` | **linear ratio 0–1** | **Not HNR dB** |
| `inharmonicity_estimate` | Mean `\|f_n/(n·f₁)−1\|` for `n≥2` using measured fundamental peak `f₁` | 1 (relative) | **Estimate**; pYIN F0 only guides search windows |
| `frequency_resolution_hz` | `sample_rate / n_fft` | Hz | STFT bin width Δf |
| `inharmonicity_resolution_floor` | Mean `Δf/(n·F0)` for used `n≥2` | 1 | Differences below this are unresolved |

**Inharmonicity and FFT resolution:** at default `n_fft=2048`, `sr=44100`,
`Δf≈21.5 Hz`. Detuning much smaller than one bin is not scientifically
resolvable; the synthetic `detuned_partials` fixture uses offsets of 30–50 Hz
(>1 bin) so discrimination is above the resolution floor.

## Reserved (later milestones)

Envelope, stereo, rhythm, reverberation.
