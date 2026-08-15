import math
from pathlib import Path

import numpy as np
import soundfile as sf

from neuroacoustic.calibration import build_profile, explanation, transform
from neuroacoustic.pipeline import run_pipeline


def test_calibration_is_reproducible_and_zeroes_constant_dimensions() -> None:
    vectors = [[0.0] * 36, [0.0] * 36, [0.0] * 36]
    vectors[0][0], vectors[1][0], vectors[2][0] = 0.1, 0.2, 0.9
    vectors[0][5], vectors[1][5], vectors[2][5] = 0.1, 0.2, 0.8
    profile = build_profile(vectors, [1, 2, 3])
    assert 30 in profile.inactive_dimensions
    assert transform(vectors[0], profile) == transform(vectors[0], profile)
    assert transform(vectors[0], profile)[0] < 0


def test_same_source_variant_is_closer_than_contrasting_synthetic_signal() -> None:
    base = [0.0] * 36; variant = [0.0] * 36; different = [0.0] * 36; anchor = [0.0] * 36
    for i in (0, 1, 5, 12, 17, 22): base[i], variant[i], different[i], anchor[i] = .2, .1, .9, .3
    profile = build_profile([base, variant, different, anchor], [1, 2, 3, 4])
    a, b, c = (transform(x, profile) for x in (base, variant, different))
    dot = lambda x, y: sum(i*j for i, j in zip(x, y))
    assert dot(a, b) > dot(a, c)


def _write(path: Path, samples: np.ndarray, rate: int = 16_000) -> Path:
    sf.write(path, samples, rate, subtype="PCM_16")
    return path


def test_calibrated_physical_fixture_retrieval_and_explanations(app_config, tmp_path: Path) -> None:
    """Real pipeline vectors: controlled acoustic changes, no hand-made vectors."""
    rate = 16_000; t = np.arange(rate * 2) / rate
    sine = .35 * np.sin(2 * np.pi * 440 * t)
    sine_variant = .33 * np.sin(2 * np.pi * 440 * t) + .003 * np.sin(2 * np.pi * 31 * t)
    stack = sum(.18 / n * np.sin(2 * np.pi * 220 * n * t) for n in range(1, 5))
    stack_variant = sum(.17 / n * np.sin(2 * np.pi * 220 * n * t) for n in range(1, 5))
    rng = np.random.default_rng(7); noise = rng.normal(0, .2, len(t))
    decay_fast = .6 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 4)
    decay_slow = .6 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 1)
    stereo_same = np.column_stack((sine, sine))
    stereo_inverted = np.column_stack((sine, -sine))
    stereo_wide = np.column_stack((sine, .4 * sine))
    stereo_inverted_variant = np.column_stack((.9 * sine, -.9 * sine))
    signals = {"sine": sine, "sine_variant": sine_variant, "stack": stack,
               "stack_variant": stack_variant, "noise": noise, "decay_fast": decay_fast,
               "decay_slow": decay_slow, "stereo_same": stereo_same, "stereo_inverted": stereo_inverted,
               "stereo_wide": stereo_wide, "stereo_inverted_variant": stereo_inverted_variant}
    vectors = {}; raw = {}
    for name, signal in signals.items():
        result = run_pipeline(_write(tmp_path / f"{name}.wav", signal), app_config, output_dir=tmp_path / "out", plots=False)
        raw[name] = list(result.fingerprint.vector); vectors[name] = raw[name]
    profile = build_profile(list(vectors.values()), list(range(len(vectors))))
    calibrated = {name: transform(vector, profile) for name, vector in vectors.items()}
    cosine = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert cosine(calibrated["sine"], calibrated["sine_variant"]) > cosine(calibrated["sine"], calibrated["noise"])
    assert cosine(calibrated["stack"], calibrated["stack_variant"]) > cosine(calibrated["stack"], calibrated["noise"])
    stereo = explanation(raw["stereo_same"], raw["stereo_inverted"], profile)
    decay = explanation(raw["decay_fast"], raw["decay_slow"], profile)
    assert any(item["group"] == "stereo" and abs(item["standardized_difference"]) > .1 for item in stereo["features"])
    assert any(item["group"] == "envelope_decay" and abs(item["standardized_difference"]) > .1 for item in decay["features"])
    for detail in (stereo, decay):
        for item in detail["positive"] + detail["negative"]:
            assert item["index"] not in profile.inactive_dimensions
            assert all(math.isfinite(float(item[key])) for key in ("query", "neighbour", "standardized_difference", "weight"))
