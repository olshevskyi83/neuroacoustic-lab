# Metrics

Definitions, formulas, units, ranges, aggregation methods, and limitations for every acoustic metric.

**Milestone 1:** source/probe metadata only. Spectral and higher-level metrics will be documented here as they are implemented (Milestones 2–4).

## Source / probe metadata

| Field | Meaning | Unit | Notes |
|-------|---------|------|-------|
| `content_hash` | SHA-256 of file bytes | hex string | Identity of file contents, not perceptual hash |
| `file_size_bytes` | On-disk size | bytes | Exact |
| `duration_seconds` | Media duration | seconds | From container/header or sample count / rate |
| `native_sample_rate` | Sampling rate of the source | Hz | Preserved when practical |
| `analysis_sample_rate` | Rate used for analysis | Hz | Equals native unless explicitly configured |
| `channels` | Channel count | count | Integer ≥ 1 |
| `bit_depth` | Bits per sample when known | bits | May be null (e.g. some lossy formats) |
| `bitrate` | Encoded bitrate when known | bit/s | Often available for MP3 via ffprobe; may be null |
| `container` / `codec` | Format labels | — | Best-effort from extension, soundfile, or ffprobe |

## Quality (load-time, Milestone 1)

| Field | Meaning | Unit / range | Limitations |
|-------|---------|--------------|-------------|
| `clipping_ratio` | Fraction of samples with \|x\| ≥ clipping threshold | 0–1 | Threshold-based; not a standards loudness measure |
| `silence_ratio` | Fraction of samples with \|x\| < silence sample threshold | 0–1 | Heuristic near-zero count |
| Near-silence file | Peak \|x\| below silence peak threshold | — | Raises or warns; does not invent features |

## Reserved for later milestones

Spectral, energy, pitch, harmonics, envelope, stereo, rhythm, and reverberation metrics will be added with full formula documentation. Any reverberation figure in the MVP will be labeled a **heuristic estimate** with confidence and limitations — not claimed RT60 from arbitrary mixed music.
