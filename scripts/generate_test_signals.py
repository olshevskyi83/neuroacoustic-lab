#!/usr/bin/env python3
"""Generate deterministic synthetic audio fixtures for tests.

Signals (default 1.0 s @ 44100 Hz unless noted):
  1. pure sine 110 Hz
  2. pure sine 440 Hz
  3. additive 110+220+330+440 Hz with known amplitudes
  4. white noise (seeded)
  5. silence
  6. short impulse with exponential decay
  7. stereo with controlled channel correlation (ρ≈0.8)
  8. mildly detuned partials (inharmonicity test)
  9. short 440 Hz sine (0.05 s)
 Milestone 4A:
 10. slow-attack sine (linear fade-in)
 11. sustained tone (constant amplitude)
 12. mono (explicit single-channel sine)
 13. identical stereo channels
 14. inverted stereo channels
 15. independent/uncorrelated stereo channels
 16. click track at 120 BPM (4 s)
 17. drone without beats (long sine)
 18. exponentially decaying noise burst
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
    """Detuned partials for inharmonicity discrimination (see M3 docs)."""
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
    """Percussive attack + exponential amplitude decay (τ seconds)."""
    duration = 0.5
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    signal = amplitude * np.exp(-t / decay_tau)
    signal[0] = amplitude
    return signal.astype(np.float32)


def slow_attack_sine(
    freq: float = 220.0,
    duration: float = 1.0,
    sr: int = DEFAULT_SR,
    attack_s: float = 0.4,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Sine with linear fade-in over ``attack_s`` then sustained."""
    t = _time_vector(duration, sr)
    env = np.ones_like(t)
    n_att = int(round(attack_s * sr))
    if n_att > 0:
        env[:n_att] = np.linspace(0.0, 1.0, n_att, endpoint=False)
    return (amplitude * env * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def sustained_tone(
    freq: float = 330.0, duration: float = 1.0, sr: int = DEFAULT_SR, amplitude: float = 0.4
) -> np.ndarray:
    return sine(freq, duration, sr, amplitude=amplitude)


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


def stereo_identical(duration: float, sr: int) -> np.ndarray:
    mono = sine(440.0, duration, sr, amplitude=0.4)
    return np.stack([mono, mono], axis=1)


def stereo_inverted(duration: float, sr: int) -> np.ndarray:
    mono = sine(440.0, duration, sr, amplitude=0.4)
    return np.stack([mono, -mono], axis=1)


def stereo_independent(duration: float, sr: int) -> np.ndarray:
    """Near-zero correlation via independent seeded noise channels."""
    rng = np.random.default_rng(RNG_SEED + 7)
    n = int(round(duration * sr))
    left = rng.standard_normal(n)
    right = rng.standard_normal(n)
    stereo = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo)) or 1.0
    return (stereo * (0.4 / peak)).astype(np.float32)


def click_track(
    bpm: float = 120.0,
    duration: float = 4.0,
    sr: int = DEFAULT_SR,
    click_duration: float = 0.01,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Periodic clicks at ``bpm`` for tempo tests."""
    n = int(round(duration * sr))
    out = np.zeros(n, dtype=np.float64)
    period = 60.0 / bpm
    click_n = max(1, int(round(click_duration * sr)))
    t_click = np.arange(click_n, dtype=np.float64) / sr
    # Short decaying burst as click
    click = amplitude * np.exp(-t_click / 0.003) * np.sin(2.0 * np.pi * 1000.0 * t_click)
    t = 0.0
    while t < duration:
        start = int(round(t * sr))
        end = min(n, start + click_n)
        out[start:end] += click[: end - start]
        t += period
    peak = np.max(np.abs(out)) or 1.0
    return (out * (0.9 / peak)).astype(np.float32)


def drone(duration: float = 3.0, sr: int = DEFAULT_SR, freq: float = 110.0) -> np.ndarray:
    """Long steady tone with negligible onset contrast."""
    return sine(freq, duration, sr, amplitude=0.35)


def exp_decay_noise(
    sr: int = DEFAULT_SR,
    duration: float = 1.0,
    decay_tau: float = 0.12,
    amplitude: float = 0.7,
) -> np.ndarray:
    """Band-limited-ish noise with exponential amplitude envelope (known τ)."""
    rng = np.random.default_rng(RNG_SEED + 11)
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    noise = rng.standard_normal(n)
    env = amplitude * np.exp(-t / decay_tau)
    out = noise * env
    peak = np.max(np.abs(out)) or 1.0
    return (out * (0.9 / peak)).astype(np.float32)


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
        ("slow_attack_sine.wav", slow_attack_sine(duration=duration, sr=sr)),
        ("sustained_tone.wav", sustained_tone(duration=duration, sr=sr)),
        ("mono_sine.wav", sine(440.0, duration, sr)),
        ("stereo_identical.wav", stereo_identical(duration, sr)),
        ("stereo_inverted.wav", stereo_inverted(duration, sr)),
        ("stereo_independent.wav", stereo_independent(duration, sr)),
        ("click_track_120bpm.wav", click_track(bpm=120.0, duration=4.0, sr=sr)),
        ("drone.wav", drone(duration=3.0, sr=sr)),
        ("exp_decay_noise.wav", exp_decay_noise(sr=sr, duration=1.0, decay_tau=0.12)),
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
