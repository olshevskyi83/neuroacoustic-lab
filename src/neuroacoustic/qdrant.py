"""Rebuildable Qdrant projection of completed SQLite acoustic analyses.

This module deliberately has no SQLite write operations.  A Qdrant outage can
make projection commands fail, but cannot change or invalidate an analysis.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from neuroacoustic.exceptions import QdrantError
from neuroacoustic.persistence.models import Analysis

VECTOR_DIMENSION = 36
VECTOR_VERSION = "0.4.0-preliminary"
DEFAULT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "neuroacoustic_fingerprints"
_POINT_NAMESPACE = uuid.UUID("90744405-7b4b-4a83-9302-b17dd154db7e")
SCALAR_FIELDS = (
    "duration_seconds", "analysis_sample_rate", "rms", "peak_amplitude",
    "centroid_mean_hz", "flatness_mean", "entropy_mean", "f0_median_hz",
    "voiced_ratio", "tempo_bpm", "stereo_width_estimate", "attack_time_s",
    "tail_decay_t60_s",
)

# These groups refer to the documented ordering in fingerprint.normalization.
# They intentionally report the cosine calculation over the same normalized
# components Qdrant receives, rather than attempting to infer similarity from
# a few display scalars.
EXPLANATION_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("spectral", (0, 1, 2, 3, 4, 5, 6, 9, 10, 11)),
    ("harmonic", tuple(range(12, 22))),
    ("decay/envelope", (22, 23, 24, 25, 26, 34, 35)),
    ("stereo/energy", (7, 8, 27, 28, 29, 30)),
)


@dataclass(frozen=True, slots=True)
class QdrantSettings:
    url: str = DEFAULT_URL
    collection: str = DEFAULT_COLLECTION
    enabled: bool = False

    @classmethod
    def from_environment(cls) -> "QdrantSettings":
        enabled = os.environ.get("QDRANT_ENABLED", "false").strip().lower()
        if enabled not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise QdrantError("QDRANT_ENABLED must be true or false")
        return cls(
            url=os.environ.get("QDRANT_URL", DEFAULT_URL).rstrip("/"),
            collection=os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION),
            enabled=enabled in {"true", "1", "yes", "on"},
        )


@dataclass(frozen=True, slots=True)
class SyncReport:
    eligible: int
    synced: int
    skipped_invalid: int
    missing_before_sync: int
    stale_points: int


def deterministic_point_id(analysis_id: int) -> str:
    """Stable UUID accepted by Qdrant, derived only from SQLite analysis id."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"analysis:{analysis_id}"))


def _finite_vector(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        vector = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(vector, list) or len(vector) != VECTOR_DIMENSION:
        return None
    try:
        result = [float(item) for item in vector]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def validated_analysis_vector(row: Analysis) -> list[float]:
    """Return a Qdrant-safe vector or explain why this SQLite row is ineligible."""
    if row.vector_version != VECTOR_VERSION:
        raise QdrantError(
            f"Analysis {row.id} vector_version={row.vector_version!r}; "
            f"expected {VECTOR_VERSION!r}."
        )
    vector = _finite_vector(row.vector_json)
    if vector is None:
        raise QdrantError(
            f"Analysis {row.id} vector must contain exactly {VECTOR_DIMENSION} finite values."
        )
    return vector


def _scalar(value: Any) -> int | float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(value) if isinstance(value, int) else number


def point_payload(row: Analysis) -> dict[str, Any]:
    """Produce deliberately small, non-path payload used for filtering/explanation."""
    payload: dict[str, Any] = {
        "analysis_id": row.id,
        "track_id": row.track_id,
        "content_hash": row.content_hash,
        "filename": row.track.filename if row.track else None,
        "analysis_version": row.analysis_version,
        "vector_version": row.vector_version,
        "created_at": row.created_at.isoformat() if isinstance(row.created_at, datetime) else None,
    }
    payload["updated_at"] = row.updated_at.isoformat() if isinstance(row.updated_at, datetime) else None
    payload.update(
        {field: scalar for field in SCALAR_FIELDS if (scalar := _scalar(getattr(row, field))) is not None}
    )
    return payload


class QdrantClient:
    """Tiny REST client to keep Qdrant optional and easy to mock in tests."""

    def __init__(self, settings: QdrantSettings, *, timeout: float = 10.0) -> None:
        self.settings = settings
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        expect_json: bool = True,
    ) -> Any:
        data = json.dumps(body, allow_nan=False).encode() if body is not None else None
        request = Request(
            f"{self.settings.url}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - configured URL
                raw = response.read().decode()
        except HTTPError as exc:
            if exc.code == 404:
                raise QdrantError("not found") from exc
            raise QdrantError(f"Qdrant HTTP {exc.code}: {exc.reason}") from exc
        except (URLError, OSError) as exc:
            raise QdrantError(f"Qdrant unavailable at {self.settings.url}: {exc}") from exc
        if not expect_json:
            return raw
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise QdrantError("Qdrant returned invalid JSON") from exc

    @property
    def _collection_path(self) -> str:
        return f"/collections/{quote(self.settings.collection, safe='')}"

    def health(self) -> bool:
        # Qdrant health responses may be plain text rather than JSON.
        self._request("GET", "/healthz", expect_json=False)
        return True

    def collection_info(self) -> dict[str, Any] | None:
        try:
            response = self._request("GET", self._collection_path)
        except QdrantError as exc:
            if str(exc) == "not found":
                return None
            raise
        return response.get("result", response)

    def initialize_collection(self) -> bool:
        """Create our collection if absent; otherwise verify immutable vector settings."""
        info = self.collection_info()
        if info is None:
            self._request("PUT", self._collection_path, {
                "vectors": {"size": VECTOR_DIMENSION, "distance": "Cosine"},
            })
            return True
        vectors = info.get("config", {}).get("params", {}).get("vectors", {})
        size, distance = vectors.get("size"), str(vectors.get("distance", "")).lower()
        if size != VECTOR_DIMENSION or distance != "cosine":
            raise QdrantError(
                f"Collection {self.settings.collection!r} is incompatible: "
                f"expected size={VECTOR_DIMENSION}, distance=Cosine; got size={size}, "
                f"distance={vectors.get('distance')!r}. Refusing to write."
            )
        return False

    def upsert(self, points: list[dict[str, Any]]) -> None:
        if points:
            self._request("PUT", f"{self._collection_path}/points?wait=true", {"points": points})

    def delete_point_after_explicit_sqlite_delete(self, analysis_id: int) -> None:
        """Projection cleanup hook; call only after the SQLite deletion committed."""
        self._request("POST", f"{self._collection_path}/points/delete?wait=true", {
            "points": [deterministic_point_id(analysis_id)],
        })

    def point_ids(self, vector_version: str | None = None) -> set[str]:
        """Read IDs for reporting only; sync never deletes stale projection points."""
        ids: set[str] = set()
        offset: Any = None
        while True:
            body: dict[str, Any] = {
                "limit": 256, "with_payload": ["analysis_id"], "with_vector": False,
            }
            if vector_version is not None:
                body["filter"] = {
                    "must": [{"key": "vector_version", "match": {"value": vector_version}}]
                }
            if offset is not None:
                body["offset"] = offset
            result = self._request("POST", f"{self._collection_path}/points/scroll", body).get("result", {})
            ids.update(str(point["id"]) for point in result.get("points", []))
            offset = result.get("next_page_offset")
            if offset is None:
                return ids

    def search(self, vector: list[float], vector_version: str, analysis_id: int, limit: int) -> list[dict[str, Any]]:
        body = {
            "vector": vector, "limit": limit,
            # Candidate vectors are required only for the transparent grouped
            # cosine explanation shown by `similar`; they are never persisted
            # back to SQLite.
            "with_payload": True, "with_vector": True,
            "filter": {
                "must": [{"key": "vector_version", "match": {"value": vector_version}}],
                "must_not": [{"key": "analysis_id", "match": {"value": analysis_id}}],
            },
        }
        response = self._request("POST", f"{self._collection_path}/points/search", body)
        return list(response.get("result", response))


def completed_rows(session: Session) -> list[Analysis]:
    return list(session.scalars(
        select(Analysis).options(joinedload(Analysis.track)).where(
            Analysis.status == "completed", Analysis.vector_json.is_not(None),
        )
    ).unique())


def sync_completed(session: Session, client: QdrantClient, *, batch_size: int = 100) -> SyncReport:
    """Upsert valid completed rows. Caller should initialize first; SQLite is read-only."""
    client.initialize_collection()
    rows = completed_rows(session)
    current = client.point_ids()
    valid: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        try:
            vector = validated_analysis_vector(row)
        except QdrantError:
            skipped += 1
            continue
        valid.append({"id": deterministic_point_id(row.id), "vector": vector, "payload": point_payload(row)})
    expected = {point["id"] for point in valid}
    for start in range(0, len(valid), batch_size):
        client.upsert(valid[start : start + batch_size])
    return SyncReport(len(rows), len(valid), skipped, len(expected - current), len(current - expected))


def sync_calibrated(session: Session, client: QdrantClient, profile: Any, *, batch_size: int = 100) -> SyncReport:
    """Project a separately calibrated space; raw SQLite vectors are never changed."""
    from neuroacoustic.calibration import CALIBRATED_VECTOR_VERSION, transform
    client.initialize_collection()
    rows = completed_rows(session); current = client.point_ids(); valid = []; skipped = 0
    for row in rows:
        try:
            vector = transform(validated_analysis_vector(row), profile)
        except (QdrantError, ValueError):
            skipped += 1; continue
        payload = point_payload(row)
        payload.update({"vector_version": CALIBRATED_VECTOR_VERSION, "calibration_version": profile.version, "calibration_profile_sha256": profile.profile_sha256, "preliminary_vector": validated_analysis_vector(row)})
        valid.append({"id": deterministic_point_id(row.id), "vector": vector, "payload": payload})
    for start in range(0, len(valid), batch_size): client.upsert(valid[start:start + batch_size])
    expected = {point["id"] for point in valid}
    return SyncReport(len(rows), len(valid), skipped, len(expected-current), len(current-expected))


def scalar_comparisons(query: Analysis, payload: dict[str, Any]) -> str:
    """Concise numeric deltas, never a semantic or LLM-generated similarity claim."""
    parts: list[str] = []
    for field, label, unit in (("centroid_mean_hz", "centroid", "Hz"), ("f0_median_hz", "F0", "Hz"), ("tempo_bpm", "tempo", "bpm"), ("rms", "RMS", "")):
        left, right = getattr(query, field), payload.get(field)
        if left is not None and right is not None:
            try:
                delta = abs(float(right) - float(left))
                if math.isfinite(delta):
                    parts.append(f"{label} Δ {delta:.3g}{unit}")
            except (TypeError, ValueError):
                pass
    return "; ".join(parts) or "no comparable scalar fields"


def _cosine_for_indices(left: list[float], right: list[float], indices: tuple[int, ...]) -> float | None:
    """Cosine similarity over one named subset of the indexed vector."""
    dot = sum(left[index] * right[index] for index in indices)
    left_norm = math.sqrt(sum(left[index] ** 2 for index in indices))
    right_norm = math.sqrt(sum(right[index] ** 2 for index in indices))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return dot / (left_norm * right_norm)


def metric_explanation(query_vector: list[float], candidate_vector: list[float]) -> str:
    """Explain a neighbour using grouped cosine scores from the actual vectors."""
    if len(query_vector) != VECTOR_DIMENSION or len(candidate_vector) != VECTOR_DIMENSION:
        return "group metrics unavailable (invalid vector dimension)"
    scores = []
    for name, indices in EXPLANATION_GROUPS:
        score = _cosine_for_indices(query_vector, candidate_vector, indices)
        scores.append(f"{name} {'n/a' if score is None else f'{score:.3f}'}")
    return "; ".join(scores)
