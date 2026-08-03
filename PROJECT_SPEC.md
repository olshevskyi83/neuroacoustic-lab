PROJECT: NeuroAcoustic Lab
LOCATION: /home/homelabuser/projects/neuroacoustic-lab

MISSION

Build a local-first research platform for analyzing sound by its physical
acoustic properties rather than genre, artist, album, or musical style.

The first milestone is a reliable Audio Analysis Engine that converts an
audio file into a reproducible acoustic fingerprint.

Do not implement AI/LLM features yet.
Do not implement synthesis yet.
Focus only on Phase 1 and the foundation for Phase 2.

============================================================
OPERATING RULES
============================================================

1. Work inside the current repository only.
2. First inspect the server environment:
   - OS and architecture
   - CPU and RAM
   - available disk space
   - Python version
   - FFmpeg/ffprobe availability
   - Docker and Docker Compose availability
   - Git status
3. Show the environment report before making major architectural changes.
4. Do not install packages globally.
5. Use a local Python virtual environment.
6. Do not use sudo without explicit permission.
7. Do not commit secrets, audio collections, databases, generated plots,
   model files, or virtual environments.
8. Make small, understandable commits.
9. Do not push to GitHub until the remote repository has been confirmed.
10. All analysis formulas and units must be documented.
11. Prefer deterministic and testable signal-processing algorithms.
12. A metric that is only an estimate must be named and documented as an
    estimate. Do not present heuristics as exact physical measurements.
13. Preserve the native sample rate for analysis when practical, but support
    an explicit configurable analysis sample rate.
14. Convert integer audio samples to floating point consistently.
15. Avoid destructive normalization that changes the properties being measured.

============================================================
TECHNOLOGY
============================================================

Language:
- Python 3.11 or newer

Core dependencies:
- numpy
- scipy
- librosa
- soundfile
- pydantic
- typer
- rich
- sqlalchemy
- matplotlib
- pytest

System dependency:
- ffmpeg
- ffprobe

Optional later:
- essentia
- FastAPI
- Qdrant
- SuperCollider

Do not add Essentia during the first implementation unless it is already
available and clearly necessary. The first MVP must work without Essentia.

Use:
- pyproject.toml
- src layout
- pytest
- type hints
- structured logging
- pathlib
- configuration through YAML or TOML plus environment overrides

============================================================
ARCHITECTURE
============================================================

Use a modular pipeline:

audio file
  -> validation and ffprobe metadata
  -> decoding/loading
  -> channel-aware preprocessing
  -> frame-level acoustic analysis
  -> time aggregation
  -> fingerprint construction
  -> JSON export
  -> SQLite persistence
  -> optional plot generation

Create approximately this repository structure:

neuroacoustic-lab/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── config/
│   └── default.toml
├── docs/
│   ├── architecture.md
│   ├── fingerprint-schema.md
│   └── metrics.md
├── src/
│   └── neuroacoustic/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── exceptions.py
│       ├── logging.py
│       ├── audio/
│       │   ├── probe.py
│       │   ├── loader.py
│       │   └── preprocessing.py
│       ├── analysis/
│       │   ├── spectrum.py
│       │   ├── pitch.py
│       │   ├── harmonics.py
│       │   ├── spectral.py
│       │   ├── energy.py
│       │   ├── envelope.py
│       │   ├── stereo.py
│       │   ├── rhythm.py
│       │   └── reverb.py
│       ├── fingerprint/
│       │   ├── models.py
│       │   ├── builder.py
│       │   └── normalization.py
│       ├── persistence/
│       │   ├── database.py
│       │   ├── models.py
│       │   └── repository.py
│       ├── visualization/
│       │   ├── spectrum.py
│       │   └── spectrogram.py
│       └── pipeline.py
├── tests/
│   ├── fixtures/
│   ├── test_loader.py
│   ├── test_spectral.py
│   ├── test_pitch.py
│   ├── test_harmonics.py
│   ├── test_fingerprint.py
│   └── test_pipeline.py
├── data/
│   ├── input/.gitkeep
│   ├── output/.gitkeep
│   └── db/.gitkeep
└── scripts/
    └── generate_test_signals.py

The exact structure may be adjusted if there is a strong technical reason,
but keep audio loading, analysis, fingerprints, persistence, and
visualization separated.

============================================================
SUPPORTED INPUT
============================================================

Support:
- WAV
- FLAC
- AIFF/AIF
- MP3

Use soundfile for directly supported formats.
Use FFmpeg as a fallback decoder, especially for MP3.

Validate:
- file existence
- extension/container
- duration
- sample rate
- channel count
- whether decoding succeeds
- NaN/Inf samples
- empty or nearly silent files

Do not overwrite the source audio.

============================================================
PHASE 1 FEATURES
============================================================

Implement these reliable features first:

Metadata:
- filename
- content hash, preferably SHA-256
- file size
- container/codec
- duration in seconds
- native sample rate
- analysis sample rate
- number of channels
- bit depth when available
- bitrate when available

Spectral:
- FFT summary
- STFT
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral contrast
- spectral flatness
- spectral entropy

Energy/time domain:
- RMS energy
- zero crossing rate
- peak amplitude
- crest factor
- estimated dynamic range
- loudness-related values must not be called LUFS unless a proper
  standards-compliant implementation is used

Pitch and harmonics:
- estimated fundamental frequency over time
- voiced/unvoiced decisions
- pitch confidence
- median and distribution of voiced F0
- harmonic peak frequencies
- harmonic amplitudes
- harmonic count
- harmonic-to-noise estimate
- inharmonicity estimate
- harmonic density
- normalized harmonic distribution

Envelope:
- onset/attack estimate
- decay estimate
- sustain-level estimate
- release estimate

Stereo:
- left/right RMS
- inter-channel correlation
- mid/side energy ratio
- stereo width estimate

Rhythm:
- estimated tempo
- beat positions
- onset strength summary

Reverberation:
- only a clearly named heuristic decay/reverberation estimate in the MVP
- include confidence and limitations
- do not claim exact RT60 from arbitrary mixed music without justification

For long files, do not store the complete raw STFT matrix inside the JSON.
Store summaries, distributions, selected time-series curves, and references
to separately generated artifacts.

============================================================
FINGERPRINT MODEL
============================================================

Create a versioned Pydantic schema.

The fingerprint should contain:

{
  "schema_version": "...",
  "analysis_version": "...",
  "source": {...},
  "analysis_config": {...},
  "quality": {
    "clipping_ratio": ...,
    "silence_ratio": ...,
    "warnings": [...]
  },
  "spectral": {...},
  "pitch": {...},
  "harmonics": {...},
  "energy": {...},
  "envelope": {...},
  "stereo": {...},
  "rhythm": {...},
  "reverberation": {...},
  "vector": [...],
  "created_at": "..."
}

Every metric must document:
- meaning
- formula or algorithm
- unit
- expected range where applicable
- aggregation method
- limitations

For time-varying features, provide useful statistics such as:
- mean
- standard deviation
- median
- 5th percentile
- 25th percentile
- 75th percentile
- 95th percentile
- minimum
- maximum

Do not create the final similarity vector by blindly mixing features with
different scales. Add an explicitly versioned preliminary vector and
document that dataset-level normalization will be introduced when a real
corpus is available.

============================================================
DATABASE
============================================================

Use SQLite initially.

Store:
- track identity and source metadata
- content hash
- schema version
- analysis configuration hash
- important scalar features in queryable columns
- full fingerprint JSON
- output artifact paths
- timestamps
- analysis status and errors

A repeated analysis of the same content using the same analysis version and
configuration should be detectable.

Do not commit the SQLite database.

============================================================
CLI
============================================================

Create these commands:

neuroacoustic doctor

Print environment and dependency status.

neuroacoustic probe AUDIO_FILE

Print source metadata without performing the full analysis.

neuroacoustic analyze AUDIO_FILE

Run analysis, write fingerprint JSON, and persist it in SQLite.

Options should include approximately:
- --output-dir
- --database
- --config
- --analysis-sample-rate
- --plots / --no-plots
- --force
- --json

neuroacoustic db list

List analyzed tracks.

neuroacoustic db show TRACK_ID

Show one stored fingerprint or summary.

Commands must return non-zero exit codes on failure and provide useful
messages.

============================================================
VISUAL OUTPUT FOR MVP
============================================================

Generate:
- waveform overview
- FFT magnitude plot
- log-frequency FFT plot
- spectrogram

Save plots as PNG files outside Git tracking.

Use labeled axes and physical units.
Do not display blocking GUI windows on the server.

============================================================
TESTING
============================================================

Create deterministic synthetic test signals:

1. pure sine at 110 Hz
2. pure sine at 440 Hz
3. additive signal:
   110 + 220 + 330 + 440 Hz with known amplitudes
4. white noise
5. silence
6. short impulse with exponential decay
7. stereo signal with controlled channel correlation

Test that:
- pure sine pitch is detected within a reasonable tolerance
- harmonic frequencies are detected near their expected values
- white noise has higher flatness/entropy than a pure sine
- silence is handled without crashes or invalid JSON
- stereo width reacts to channel relationships
- JSON validates against the Pydantic schema
- repeated analysis is detected
- MP3 decoding works when FFmpeg is available

Tests must use generated fixtures, not copyrighted recordings.

============================================================
PERFORMANCE AND SAFETY
============================================================

The application must not load unnecessarily large intermediate matrices
for an entire multi-hour file.

Use chunking or controlled downsampling for expensive summaries where
appropriate.

Preserve enough temporal resolution for meaningful analysis.
Record every important analysis parameter in the fingerprint.

Handle malformed audio cleanly.
Never execute metadata or filenames as shell code.
When calling FFmpeg, use argument arrays without shell=True.

============================================================
DOCUMENTATION
============================================================

README must explain:
- project philosophy
- installation
- FFmpeg requirement
- local venv setup
- example commands
- output locations
- JSON fingerprint purpose
- current limitations
- test commands

docs/architecture.md:
- module responsibilities
- processing pipeline
- persistence flow

docs/metrics.md:
- definitions and limitations of every metric

docs/fingerprint-schema.md:
- schema structure
- schema versioning
- forward compatibility

============================================================
GITIGNORE
============================================================

Ignore at least:
- .venv/
- __pycache__/
- .pytest_cache/
- .mypy_cache/
- .ruff_cache/
- .env
- *.db
- *.sqlite
- data/input/*
- data/output/*
- data/db/*
- generated audio
- generated plots
- temporary FFmpeg files

Keep .gitkeep where needed.

============================================================
DELIVERY STRATEGY
============================================================

Do not attempt every feature in one uncontrolled pass.

Work in milestones:

Milestone 1:
- environment inspection
- repository scaffold
- configuration
- audio probe/loader
- Pydantic source models
- doctor and probe CLI
- basic tests

Milestone 2:
- FFT/STFT
- spectral features
- RMS/ZCR
- fingerprint JSON
- tests using generated signals

Milestone 3:
- pitch and harmonic analysis
- harmonic distribution
- confidence and quality warnings
- tests

Milestone 4:
- envelope, stereo, rhythm, reverb heuristics
- SQLite persistence
- plots
- integration tests

After each milestone:
1. run tests
2. run the CLI on synthetic signals
3. summarize completed work
4. list known limitations
5. make a focused Git commit only after reviewing git diff

Start with Milestone 1 only.

Before editing files, inspect the server and repository. Then present:
- detected environment
- any missing prerequisites
- the exact Milestone 1 implementation plan

After presenting that plan, proceed with Milestone 1 unless an operation
requires sudo, credentials, destructive changes, or external publication.
