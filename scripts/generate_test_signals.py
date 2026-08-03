#!/usr/bin/env python3
"""Generate deterministic synthetic audio fixtures for tests.

Signals (default 1.0 s @ 44100 Hz unless noted):
  1. pure sine 110 Hz
  2. pure sine 440 Hz
  3. additive 110+220+330+440 Hz with known amplitudes
  4. white noise (seeded)
  5. silence
  6. short impulse with exponential decay
  7. stereo with controlled channel correlation
  8. mildly detuned partials (inharmonicity test)
  9. short 440 Hz sine (0.05 s)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SR = 44100
DEFAULT_DURATION = 1.0
RNG_SEED = 42


def _time_vector(duration: float, sr: int) -> np.ndarray:
    n = int(round(duration * sr))
    return np.arange(n, dtype=np.float64) / sr


def sine(freq: float, duration: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    t = _time_vector(duration, sr)
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def additive(duration: float, sr: int) -> np.ndarray:
    """110 + 220 + 330 + 440 Hz with amplitudes 0.5, 0.25, 0.125, 0.0625."""
    amps = {110.0: 0.5, 220.0: 0.25, 330.0: 0.125, 440.0: 0.0625}
    t = _time_vector(duration, sr)
    out = np.zeros_like(t, dtype=np.float64)
    for freq, amp in amps.items():
        out += amp * np.sin(2.0 * np.pi * freq * t)
    peak = np.max(np.abs(out)) or 1.0
    out = out * (0.9 / peak)
    return out.astype(np.float32)


def detuned_partials(duration: float, sr: int) -> np.ndarray:
    """Detuned partials for inharmonicity discrimination.

    Offsets are large enough to resolve above STFT bin width
    (Δf = 44100/2048 ≈ 21.5 Hz) after parabolic peak interpolation, while
    keeping a dominant 110 Hz partial so pYIN can still track F0:

      n=1: 110 Hz  (exact, amplitude 0.55)
      n=2: 235 Hz  (+15 Hz vs 220; ≈ 0.70 bins; +6.8%)
      n=3: 355 Hz  (+25 Hz vs 330; ≈ 1.16 bins; +7.6%)
      n=4: 470 Hz  (+30 Hz vs 440; ≈ 1.39 bins; +6.8%)

    True mean |f_n/(n·110)−1| for n=2..4 ≈ 0.071.
    With inharmonicity referenced to the measured f₁ peak (not drifted pYIN
    F0), this separates cleanly from an exact harmonic stack (~0.005).
    """
    amps = {110.0: 0.55, 235.0: 0.28, 355.0: 0.16, 470.0: 0.10}
    t = _time_vector(duration, sr)
    out = np.zeros_like(t, dtype=np.float64)
    for freq, amp in amps.items():
        out += amp * np.sin(2.0 * np.pi * freq * t)
    peak = np.max(np.abs(out)) or 1.0
    out = out * (0.9 / peak)
    return out.astype(np.float32)


def white_noise(duration: float, sr: int, amplitude: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    n = int(round(duration * sr))
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


def silence(duration: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(duration * sr)), dtype=np.float32)


def impulse_decay(sr: int, decay_tau: float = 0.05, amplitude: float = 0.9) -> np.ndarray:
    duration = 0.5
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    signal = amplitude * np.exp(-t / decay_tau)
    signal[0] = amplitude
    return signal.astype(np.float32)


def stereo_correlated(duration: float, sr: int, correlation: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + 1)
    n = int(round(duration * sr))
    left = rng.standard_normal(n)
    independent = rng.standard_normal(n)
    rho = float(np.clip(correlation, -1.0, 1.0))
    right = rho * left + np.sqrt(max(0.0, 1.0 - rho * rho)) * independent
    stereo = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo)) or 1.0
    stereo = (stereo * (0.5 / peak)).astype(np.float32)
    return stereo


def write_all(out_dir: Path, sr: int = DEFAULT_SR, duration: float = DEFAULT_DURATION) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, np.ndarray]] = [
        ("sine_110hz.wav", sine(110.0, duration, sr)),
        ("sine_440hz.wav", sine(440.0, duration, sr)),
        ("additive_harmonics.wav", additive(duration, sr)),
        ("detuned_partials.wav", detuned_partials(duration, sr)),
        ("white_noise.wav", white_noise(duration, sr)),
        ("silence.wav", silence(duration, sr)),
        ("impulse_decay.wav", impulse_decay(sr)),
        ("stereo_correlated.wav", stereo_correlated(duration, sr, correlation=0.8)),
        ("short_sine_440hz.wav", sine(440.0, 0.05, sr)),
    ]
    written: list[Path] = []
    for name, data in files:
        path = out_dir / name
        sf.write(str(path), data, sr, subtype="PCM_16")
        written.append(path)
        print(f"wrote {path} shape={data.shape} sr={sr}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures",
    )
    parser.add_argument("--sr", type=int, default=DEFAULT_SR)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    args = parser.parse_args()
    write_all(args.out_dir, sr=args.sr, duration=args.duration)


if __name__ == "__main__":
    main()
