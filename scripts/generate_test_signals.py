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
    # Scale to peak 0.9 without changing relative harmonic amplitudes.
    out = out * (0.9 / peak)
    return out.astype(np.float32)


def white_noise(duration: float, sr: int, amplitude: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    n = int(round(duration * sr))
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


def silence(duration: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(duration * sr)), dtype=np.float32)


def impulse_decay(sr: int, decay_tau: float = 0.05, amplitude: float = 0.9) -> np.ndarray:
    """Unit impulse at t=0 followed by exponential decay envelope on a short buffer."""
    duration = 0.5
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    # Impulse excitation with exponential decay (simple synthetic reverb-like tail).
    signal = np.zeros(n, dtype=np.float64)
    signal[0] = amplitude
    signal *= np.exp(-t / decay_tau)
    # Also apply decaying click as soft tone carrier for non-zero length content.
    signal = amplitude * np.exp(-t / decay_tau)
    signal[0] = amplitude
    return signal.astype(np.float32)


def stereo_correlated(duration: float, sr: int, correlation: float = 0.8) -> np.ndarray:
    """Two-channel signal with approximate inter-channel correlation ``correlation``."""
    rng = np.random.default_rng(RNG_SEED + 1)
    n = int(round(duration * sr))
    left = rng.standard_normal(n)
    independent = rng.standard_normal(n)
    # right = rho * left + sqrt(1-rho^2) * independent  (for unit-variance Gaussian)
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
        ("white_noise.wav", white_noise(duration, sr)),
        ("silence.wav", silence(duration, sr)),
        ("impulse_decay.wav", impulse_decay(sr)),
        ("stereo_correlated.wav", stereo_correlated(duration, sr, correlation=0.8)),
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
