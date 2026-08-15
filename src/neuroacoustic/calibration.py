"""Reproducible robust calibration for the preliminary acoustic vector.

This is a retrieval calibration over a small local corpus, not a scientific,
clinical, genre, or perceptual model.  Preliminary missing measurements are
already encoded as 0.0; calibration deliberately treats that value as the
recorded value because the source vector has no separate missingness mask.
"""
from __future__ import annotations

import json
import math
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from neuroacoustic.qdrant import VECTOR_DIMENSION, VECTOR_VERSION

CALIBRATION_VERSION = "calibrated-v1"
CALIBRATED_VECTOR_VERSION = f"{VECTOR_VERSION}+{CALIBRATION_VERSION}"
DEFAULT_CALIBRATED_COLLECTION = "neuroacoustic_fingerprints_calibrated_v1"
EPSILON = 1.0e-9

# Explicit retrieval priorities.  The raw vector's duplicated/common stereo and
# confidence values otherwise overwhelm cosine; each category's total weight is
# intentionally comparable rather than a claim of perceptual importance.
GROUPS = {
    "spectral": (tuple(range(0, 7)) + (9, 10, 11), 1.0),
    "energy": ((7, 8), 0.7),
    "harmonic": (tuple(range(12, 22)), 1.2),
    "envelope_decay": ((22, 23, 24, 25, 26, 34, 35), 1.0),
    "stereo": ((27, 28, 29, 30), 0.5),
    "rhythm": ((31, 32, 33), 0.7),
}
LABELS = ("peak frequency","spectral centroid","spectral bandwidth","spectral rolloff","spectral flatness","spectral entropy","spectral contrast","RMS","peak amplitude","crest factor","zero crossing rate","dynamic range","voiced ratio","median F0","F0 variation","voiced probability","pitch confidence","harmonic count","harmonic density","harmonic energy fraction","inharmonicity","harmonic confidence","attack time","decay time","sustain level","release time","envelope confidence","stereo correlation","stereo width","side/mid ratio","is stereo","tempo","onset strength","rhythm confidence","tail decay T60","reverb confidence")


@dataclass(frozen=True)
class CalibrationProfile:
    version: str
    source_vector_version: str
    sample_count: int
    analysis_ids: list[int]
    median: list[float]
    iqr: list[float]
    weights: list[float]
    inactive_dimensions: list[int]
    profile_sha256: str = ""

    def to_dict(self) -> dict:
        value = asdict(self)
        if not value["profile_sha256"]:
            value["profile_sha256"] = profile_hash(value)
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "CalibrationProfile":
        expected = profile_hash({**value, "profile_sha256": ""})
        if value.get("profile_sha256") != expected:
            raise ValueError("calibration profile SHA-256 mismatch")
        profile = cls(**value)
        if profile.version != CALIBRATION_VERSION or len(profile.median) != VECTOR_DIMENSION:
            raise ValueError("incompatible calibration profile")
        return profile


def profile_hash(value: dict) -> str:
    material = dict(value); material["profile_sha256"] = ""
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_profile(vectors: list[list[float]], analysis_ids: list[int]) -> CalibrationProfile:
    if not vectors or any(len(vector) != VECTOR_DIMENSION for vector in vectors):
        raise ValueError("calibration requires finite 36-dimensional vectors")
    columns = list(zip(*vectors))
    median = [sorted(column)[len(column) // 2] if len(column) % 2 else (sorted(column)[len(column)//2-1] + sorted(column)[len(column)//2]) / 2 for column in columns]
    def percentile(values: tuple[float, ...], p: float) -> float:
        ordered = sorted(values); pos = (len(ordered) - 1) * p; lo = int(pos); hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    iqr = [percentile(column, .75) - percentile(column, .25) for column in columns]
    weights = [0.0] * VECTOR_DIMENSION
    for indices, total in GROUPS.values():
        active = [i for i in indices if iqr[i] > EPSILON]
        for i in active:
            weights[i] = total / math.sqrt(len(active))
    profile = CalibrationProfile(CALIBRATION_VERSION, VECTOR_VERSION, len(vectors), analysis_ids, median, iqr, weights, [i for i, value in enumerate(iqr) if value <= EPSILON])
    value = asdict(profile)
    return CalibrationProfile(**{**value, "profile_sha256": profile_hash(value)})


def transform(vector: list[float], profile: CalibrationProfile) -> list[float]:
    if profile.source_vector_version != VECTOR_VERSION:
        raise ValueError("calibration profile source vector version mismatch")
    if len(vector) != VECTOR_DIMENSION or not all(math.isfinite(v) for v in vector):
        raise ValueError("invalid preliminary vector")
    values = [0.0 if profile.iqr[i] <= EPSILON else ((value - profile.median[i]) / profile.iqr[i]) * profile.weights[i] for i, value in enumerate(vector)]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= EPSILON:
        raise ValueError("calibrated vector has zero norm")
    return [value / norm for value in values]


def save_profile(profile: CalibrationProfile, path: Path) -> None:
    path.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_profile(path: Path) -> CalibrationProfile:
    return CalibrationProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def explanation(query: list[float], neighbour: list[float], profile: CalibrationProfile, limit: int = 5) -> dict:
    """Measured calibrated differences; inactive dimensions intentionally vanish."""
    rows = []
    for i, (left, right) in enumerate(zip(query, neighbour)):
        if i in profile.inactive_dimensions: continue
        delta = (right - left) / profile.iqr[i]
        contribution = -abs(delta) * profile.weights[i]
        group = next(name for name, (indices, _) in GROUPS.items() if i in indices)
        rows.append({"index":i,"name":LABELS[i],"group":group,"query":left,"neighbour":right,"standardized_difference":delta,"weight":profile.weights[i],"contribution":contribution})
    for row in rows:
        row["interpretation"] = "similar measured value" if abs(row["standardized_difference"]) < .25 else "measured value differs"
    groups = {name: sum(abs(r["standardized_difference"])*r["weight"] for r in rows if r["group"] == name) for name in GROUPS}
    ordered = sorted(rows, key=lambda r: r["contribution"], reverse=True)
    return {"groups":groups,"positive":ordered[:limit],"negative":list(reversed(ordered[-limit:])),"features":rows}
