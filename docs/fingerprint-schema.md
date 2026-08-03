# Fingerprint schema

## Current version

- `schema_version`: **0.3.0** (populated `pitch` / `harmonics` sections)
- `analysis_version`: **0.3.0** (pYIN pitch + harmonic peak analysis)
- `vector_version`: **0.3.0-preliminary** (M2 prefix + M3 append)

## Document shape (abridged)

```json
{
  "schema_version": "0.3.0",
  "analysis_version": "0.3.0",
  "source": {},
  "analysis_config": {},
  "quality": {},
  "spectral": {},
  "pitch": {
    "method": "librosa.pyin",
    "f0_median_hz": 440.0,
    "voiced_ratio": 0.9,
    "confidence": 0.8,
    "f0_voiced_hz": { "mean": 440.0, "median": 440.0, "count": 80 },
    "f0_curve": { "times_seconds": [], "values": [440.0, null], "unit": "Hz" },
    "note": "Single-F0 estimate..."
  },
  "harmonics": {
    "max_harmonics": 12,
    "harmonic_slots_available": 12.0,
    "frequency_resolution_hz": 21.53,
    "inharmonicity_resolution_floor": 0.05,
    "harmonic_count": 4.0,
    "harmonic_density": 0.33,
    "peak_frequencies_hz": [110.0, 220.0, 330.0, 440.0],
    "relative_amplitudes": [1.0, 0.5, 0.25, 0.12],
    "normalized_distribution": [],
    "harmonic_energy_fraction_estimate": 0.7,
    "inharmonicity_estimate": 0.001,
    "note": "energy fraction is linear [0,1], not HNR dB."
  },
  "energy": {},
  "envelope": {},
  "stereo": {},
  "rhythm": {},
  "reverberation": {},
  "vector": [],
  "vector_meta": {
    "version": "0.3.0-preliminary",
    "labels": ["peak_freq_norm", "...", "voiced_ratio", "..."]
  },
  "artifacts": {},
  "created_at": "ISO-8601"
}
```

## What is intentionally omitted

- Full STFT / spectrogram matrices
- Full-resolution F0 / harmonic-per-frame matrices (curves are downsampled; harmonic summary is aggregated)

## Versioning / forward compatibility

- Bump `schema_version` on breaking JSON shape changes.
- Bump `analysis_version` when algorithms or defaults change numeric outputs.
- Preliminary vector keeps indices 0–11 stable; M3 appends at 12+.
- Empty `envelope` / `stereo` / `rhythm` / `reverberation` reserve later milestones.
