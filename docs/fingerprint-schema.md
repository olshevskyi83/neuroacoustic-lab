# Fingerprint schema

## Current version

- `schema_version`: **0.2.0** (Milestone 2 structure)
- `analysis_version`: **0.2.0** (spectral + energy algorithms)

## Document shape

```json
{
  "schema_version": "0.2.0",
  "analysis_version": "0.2.0",
  "source": { "...": "SourceMetadata" },
  "analysis_config": { "...": "AnalysisConfigSnapshot" },
  "quality": {
    "clipping_ratio": 0.0,
    "silence_ratio": 0.0,
    "peak_amplitude": 0.0,
    "warnings": []
  },
  "spectral": {
    "fft": { "peak_frequency_hz": 440.0, "peaks": [], "summary_*": [] },
    "stft": {
      "n_fft": 2048,
      "hop_length": 512,
      "n_frames": 87,
      "mean_spectrum_frequencies_hz": [],
      "mean_spectrum_magnitudes": []
    },
    "centroid_hz": { "mean": 0, "std": 0, "median": 0, "p05": 0, "p25": 0, "p75": 0, "p95": 0, "minimum": 0, "maximum": 0, "count": 0 },
    "bandwidth_hz": {},
    "rolloff_hz": {},
    "flatness": {},
    "entropy": {},
    "contrast_db": {},
    "centroid_curve": { "times_seconds": [], "values": [], "unit": "Hz" },
    "flatness_curve": {}
  },
  "pitch": {},
  "harmonics": {},
  "energy": {
    "rms": 0.0,
    "rms_db": 0.0,
    "peak_amplitude": 0.0,
    "crest_factor": 0.0,
    "zero_crossing_rate": 0.0,
    "estimated_dynamic_range_db": 0.0,
    "rms_frame": {},
    "rms_curve": {}
  },
  "envelope": {},
  "stereo": {},
  "rhythm": {},
  "reverberation": {},
  "vector": [],
  "vector_meta": {
    "version": "0.2.0-preliminary",
    "values": [],
    "labels": [],
    "note": "Preliminary..."
  },
  "artifacts": {
    "fingerprint_json": "...",
    "waveform_png": "...",
    "fft_png": "...",
    "fft_log_png": "...",
    "spectrogram_png": "..."
  },
  "created_at": "ISO-8601"
}
```

## What is intentionally omitted

- The complete raw STFT / spectrogram matrix
- Full-resolution FFT bins (only peak list + downsampled summaries)
- Full-resolution time series (curves capped by `max_timeseries_points`)

## Versioning / forward compatibility

- Bump `schema_version` on breaking JSON shape changes.
- Bump `analysis_version` when algorithms or defaults change numeric outputs.
- Empty objects (`pitch`, `harmonics`, …) reserve slots for later milestones.
- Consumers should ignore unknown fields and tolerate missing optional sections.

## Source models (Milestone 1+)

`SourceMetadata`, `AnalysisConfigSnapshot`, `QualityMetrics`, `ProbeResult` remain the probe/identity layer.
