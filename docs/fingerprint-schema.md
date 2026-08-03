# Fingerprint schema

## Current version

- `schema_version`: **0.4.0** (populated envelope / stereo / rhythm / reverberation)
- `analysis_version`: **0.4.0** (M4A envelope, stereo, rhythm, file-tail decay)
- `vector_version`: **0.4.0-preliminary** (M2+M3 prefix unchanged; M4A appends 22–35)

## Document shape (abridged)

```json
{
  "schema_version": "0.4.0",
  "analysis_version": "0.4.0",
  "source": {},
  "analysis_config": {},
  "quality": {},
  "spectral": {},
  "pitch": {},
  "harmonics": {},
  "energy": {},
  "envelope": {
    "onset_time_s": 0.0,
    "attack_time_s": 0.01,
    "decay_time_s": 0.08,
    "sustain_level": 0.12,
    "release_time_s": 0.05,
    "confidence": 0.7,
    "warnings": ["not_synthesizer_adsr"],
    "note": "ADSR fields describe envelope shape only..."
  },
  "stereo": {
    "is_mono": false,
    "left_rms": 0.2,
    "right_rms": 0.2,
    "correlation": 0.8,
    "mid_energy": 0.04,
    "side_energy": 0.01,
    "side_to_mid_ratio": 0.25,
    "stereo_width_estimate": 0.1
  },
  "rhythm": {
    "tempo_bpm": 120.0,
    "beat_times_s": [0.0, 0.5],
    "onset_event_count": 8,
    "confidence": 0.6,
    "warnings": ["half_double_time_ambiguity"]
  },
  "reverberation": {
    "tail_decay_t60_estimate_seconds": 0.85,
    "decay_slope_db_per_s": -70.0,
    "fit_r_squared": 0.95,
    "fit_db_range": 20.0,
    "warnings": ["file_tail_decay_not_room_rt60", "not_exact_rt60_for_mixed_music"]
  },
  "vector": [],
  "vector_meta": {
    "version": "0.4.0-preliminary",
    "labels": ["peak_freq_norm", "...", "attack_time_norm", "..."]
  },
  "artifacts": {},
  "created_at": "ISO-8601"
}
```

## What is intentionally omitted

- Full STFT / spectrogram matrices
- Full-resolution F0 / harmonic-per-frame / envelope matrices (curves are downsampled)
- SQLite persistence (deferred past 4A)

## Versioning / forward compatibility

- Bump `schema_version` on breaking JSON shape changes.
- Bump `analysis_version` when algorithms or defaults change numeric outputs.
- Preliminary vector: indices **0–21** are frozen in meaning/scaling from M2/M3;
  Milestone 4A **appends** only from index 22.
