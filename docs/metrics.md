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

Version `0.4.0-preliminary`:

- Indices **0–11**: Milestone 2 spectral/energy prefix (unchanged labels/scaling).
- Indices **12–21**: Milestone 3 pitch/harmonic appends (unchanged).
- Indices **22–35**: Milestone 4A envelope / stereo / rhythm / decay appends.

See `neuroacoustic.fingerprint.normalization` for the exact ordering and scales.

Unavailable pitch/harmonics/tempo/decay contribute **0.0** in the vector;
fingerprint body fields use **null** where evidence is insufficient.

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

## Envelope / ADSR estimates (Milestone 4A)

Derived from a **smoothed frame-RMS envelope**. These are shape descriptors,
**not** a recovered synthesizer ADSR program.

| Metric | Algorithm | Unit | Limitations |
|--------|-----------|------|-------------|
| `onset_time_s` | First time env ≥ `onset_ratio * peak` | s | Threshold heuristic |
| `attack_time_s` | Onset → `attack_high_ratio * peak` | s | Fallback to peak time |
| `decay_time_s` | Peak → sustain absolute level | s | 0 if never falls (sustained) |
| `sustain_level` | Median mid-file env / peak | 0–1 relative | Window `[sustain_start, sustain_end]` |
| `release_time_s` | Late fall mid-sustain → `release_ratio * peak` | s | Percussive uses 50%→10% path |
| `confidence` | Heuristic from shape/length | 0–1 | Always warns `not_synthesizer_adsr` |

Silent / near-silent / very short signals return null ADSR fields.

## Stereo (Milestone 4A)

Input layout `(n_samples, n_channels)`. Mono → `is_mono=true`, width/correlation **null**.

| Metric | Definition | Unit | Notes |
|--------|------------|------|-------|
| `left_rms` / `right_rms` | Channel RMS | linear | — |
| `correlation` | Pearson(L, R) | −1…1 | Null if a channel is silent |
| `mid_energy` | `mean(((L+R)/2)²)` | linear² | — |
| `side_energy` | `mean(((L-R)/2)²)` | linear² | — |
| `side_to_mid_ratio` | `side_energy / mid_energy` when mid usable | 1 | **null** if mid ≈ 0 (e.g. inverted identical) |
| `stereo_width_estimate` | `clip(0.5*(1−corr), 0, 1)` | 0–1 | **Correlation / phase-opposition heuristic only** — not a complete perceptual stereo-width metric |

Nearly inverted channels warn `phase_opposition` and `mono_compatibility_risk`.

## Rhythm (Milestone 4A)

| Metric | Meaning | Unit | Limitations |
|--------|---------|------|-------------|
| `onset_strength_*` | librosa onset-strength envelope stats | arb. | Not calibrated |
| `onset_event_count` | Detected onset frames | count | — |
| `tempo_bpm` | librosa `beat_track` tempo **if gate passes** | BPM | **null** if evidence weak |
| `beat_times_s` | Beat positions | s | Empty when tempo null |
| `beat_count` | Accepted beat count | count | 0 when tempo null |
| `beat_interval_cv` | `std(Δt)/mean(Δt)` of beat intervals | 1 | Needs ≥3 beats |
| `tempo_periodicity` | Onset-envelope autocorr `R(τ)` at beat period | −1…1 | Primary periodicity evidence |
| `confidence` | From periodicity × interval stability × beat density | 0–1 | Soft |

### Tempo reliability gate

A candidate BPM is **accepted only if all** hold (defaults in `config/default.toml`):

1. `duration ≥ min_duration_for_tempo_seconds` (0.5 s)
2. `onset_event_count ≥ min_onset_events_for_tempo` (4)
3. `beat_count ≥ min_beats_for_tempo` (3)
4. If ≥2 intervals: `beat_interval_cv ≤ max_beat_interval_cv` (0.35)
5. `R(τ) ≥ min_tempo_periodicity` (0.30), where `R(τ)` is the normalized
   autocorrelation of the zero-mean onset-strength envelope at lag
   `τ = 60/tempo` seconds

**Why periodicity, not BPM blacklists:** `beat_track` can invent a near-regular
grid on long broadband noise (low interval CV) while `R(τ)` stays low
(≲ 0.2). Periodic click tracks yield high `R(τ)` (typically ≳ 0.6). The 0.30
threshold is a conservative lower bound on that separation, not a fixture-tuned
BPM rule.

On failure: `tempo_bpm=null`, empty beats, low confidence, and an explicit
warning such as `tempo_suppressed_insufficient_periodicity`.

**Half-/double-time:** when tempo is reported, warning `half_double_time_ambiguity`
is always attached. Drones/silence yield **null** tempo.

## Reverberation / file-tail decay (Milestone 4A)

Broadband energy-envelope fit after the peak: `energy_db = a + b·t` with
`energy_db = 10 log10(mean(frame²))`.

| Metric | Meaning | Unit | Notes |
|--------|---------|------|-------|
| `decay_slope_db_per_s` | Fitted slope `b` | dB/s | Energy dB, not amplitude dB |
| `tail_decay_t60_estimate_seconds` | `60 / \|b\|` | s | Extrapolated; **file-tail T60 estimate**, not room RT60 |
| `fit_r_squared` | Linear-fit R² | 0–1 | Below `decay_min_r2` → null T60 |
| `fit_db_range` | Measured post-peak drop used | dB | — |
| `analyzed_frequency_range_hz` | Documented band | Hz | Fullband envelope in MVP |

**Not room RT60** for arbitrary mixed music. Sustained tones and poor fits
return **null**. Warnings always include `file_tail_decay_not_room_rt60` and
`not_exact_rt60_for_mixed_music`.
