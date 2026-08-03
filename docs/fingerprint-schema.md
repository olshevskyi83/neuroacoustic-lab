# Fingerprint schema

## Status

**Milestone 1** defines source and analysis-config models only. The full fingerprint document lands in Milestone 2+.

## Target structure (Phase 1)

```json
{
  "schema_version": "0.1.0",
  "analysis_version": "0.1.0",
  "source": { "...": "SourceMetadata" },
  "analysis_config": { "...": "AnalysisConfigSnapshot" },
  "quality": {
    "clipping_ratio": 0.0,
    "silence_ratio": 0.0,
    "warnings": []
  },
  "spectral": {},
  "pitch": {},
  "harmonics": {},
  "energy": {},
  "envelope": {},
  "stereo": {},
  "rhythm": {},
  "reverberation": {},
  "vector": [],
  "created_at": "ISO-8601"
}
```

## Schema versioning

- `schema_version` — shape of the JSON document (fields, nesting). Bump on breaking structural changes.
- `analysis_version` — implementation of feature algorithms. Bump when formulas or defaults change in a way that alters numeric outputs.

Forward compatibility: unknown fields should be preserved by consumers; readers must tolerate missing optional sections introduced in later versions.

## Milestone 1 models

### `SourceMetadata`

Identity and technical metadata of an audio file after probing:

- `filename`, `path`, `content_hash` (SHA-256), `file_size_bytes`
- `container`, `codec`, `duration_seconds`
- `native_sample_rate`, `analysis_sample_rate`, `channels`, `bit_depth`, `bitrate`
- `probe_backend` (`soundfile` | `ffprobe` | `hybrid`)

### `LoadedAudio`

In-memory decode result (not serialized into the fingerprint as raw samples):

- samples as `float32` array shaped `(n_samples,)` or `(n_samples, channels)`
- sample rate actually used for analysis
- optional quality flags from load-time checks

### Preliminary vector (later)

A versioned preliminary comparison vector will be documented when introduced. Dataset-level normalization is deferred until a real corpus exists — do not mix differently scaled features into a “final” similarity embedding blindly.
