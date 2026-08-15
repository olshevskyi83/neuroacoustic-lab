"""Qdrant projection tests using a fake REST client; no service is required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from neuroacoustic.cli import app
from neuroacoustic.config import AppConfig
from neuroacoustic.exceptions import QdrantError
from neuroacoustic.persistence.database import assert_db_usable
from neuroacoustic.persistence.models import Analysis
from neuroacoustic.pipeline import run_pipeline
from neuroacoustic.qdrant import (
    QdrantClient,
    QdrantSettings,
    metric_explanation,
    sync_completed,
    validated_analysis_vector,
)


class FakeQdrant(QdrantClient):
    def __init__(self) -> None:
        super().__init__(QdrantSettings(enabled=True))
        self.info = None
        self.points: dict[str, dict] = {}
        self.upsert_calls = 0

    def collection_info(self):  # type: ignore[no-untyped-def]
        return self.info

    def _request(self, method, path, body=None):  # type: ignore[no-untyped-def]
        if method == "PUT" and path == self._collection_path:
            self.info = {"config": {"params": {"vectors": body["vectors"]}}, "points_count": 0}
            return {"result": True}
        if path.endswith("/points/scroll"):
            return {"result": {"points": [{"id": key} for key in self.points], "next_page_offset": None}}
        if "/points?wait=true" in path:
            self.upsert_calls += 1
            for point in body["points"]:
                self.points[point["id"]] = point
            self.info["points_count"] = len(self.points)
            return {"result": {"status": "completed"}}
        raise AssertionError((method, path, body))


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.sqlite"


def _populate(db_path: Path, fixtures_dir: Path, app_config: AppConfig, tmp_path: Path) -> list[int]:
    ids = []
    for name in ("sine_110hz.wav", "sine_440hz.wav"):
        result = run_pipeline(fixtures_dir / name, app_config, output_dir=tmp_path / name, plots=False, database=db_path)
        assert result.analysis_id is not None
        ids.append(result.analysis_id)
    return ids


def test_initialize_is_idempotent() -> None:
    client = FakeQdrant()
    assert client.initialize_collection() is True
    assert client.initialize_collection() is False


def test_initialize_refuses_wrong_dimension() -> None:
    client = FakeQdrant()
    client.info = {"config": {"params": {"vectors": {"size": 35, "distance": "Cosine"}}}}
    with pytest.raises(QdrantError, match="incompatible"):
        client.initialize_collection()


def test_sync_is_idempotent_and_skips_nonfinite(
    db_path: Path, fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    _populate(db_path, fixtures_dir, app_config, tmp_path)
    engine = assert_db_usable(db_path)
    client = FakeQdrant()
    with Session(engine) as session:
        first = sync_completed(session, client)
        second = sync_completed(session, client)
    assert first.eligible == first.synced == 2
    assert first.missing_before_sync == 2
    assert second.missing_before_sync == 0
    assert len(client.points) == 2
    # Invalid JSON/vector is never sent to Qdrant.
    with Session(engine) as session:
        row = session.get(Analysis, 1)
        assert row is not None
        row.vector_json = json.dumps([float("nan")] * 36)
        session.commit()
    with Session(engine) as session:
        report = sync_completed(session, client)
    assert report.skipped_invalid == 1


def test_qdrant_disabled_does_not_change_sqlite(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from neuroacoustic.persistence.database import init_db

    init_db(db_path)
    monkeypatch.setenv("QDRANT_ENABLED", "false")
    result = CliRunner().invoke(app, ["qdrant", "sync", "--database", str(db_path)])
    assert result.exit_code != 0
    assert assert_db_usable(db_path)


def test_unavailable_qdrant_does_not_invalidate_completed_sqlite_analysis(
    db_path: Path, fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    ids = _populate(db_path, fixtures_dir, app_config, tmp_path)

    class Unavailable(FakeQdrant):
        def initialize_collection(self):  # type: ignore[no-untyped-def]
            raise QdrantError("Qdrant unavailable")

    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        with pytest.raises(QdrantError, match="unavailable"):
            sync_completed(session, Unavailable())
        row = session.get(Analysis, ids[0])
        assert row is not None and row.status == "completed" and row.vector_json


def test_search_filter_excludes_self_and_honors_version() -> None:
    client = FakeQdrant()
    captured = {}

    def request(method, path, body=None):  # type: ignore[no-untyped-def]
        captured.update(body)
        return {"result": []}

    client._request = request  # type: ignore[method-assign]
    assert client.search([0.0] * 36, "0.4.0-preliminary", 7, 3) == []
    assert captured["filter"]["must"][0]["match"]["value"] == "0.4.0-preliminary"
    assert captured["filter"]["must_not"][0]["match"]["value"] == 7
    assert captured["with_vector"] is True


def test_sync_rejects_wrong_vector_version(
    db_path: Path, fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    _populate(db_path, fixtures_dir, app_config, tmp_path)
    engine = assert_db_usable(db_path)
    with Session(engine) as session:
        row = session.get(Analysis, 1)
        assert row is not None
        row.vector_version = "not-compatible"
        session.commit()
    with Session(engine) as session:
        report = sync_completed(session, FakeQdrant())
        row = session.get(Analysis, 1)
        assert row is not None
        with pytest.raises(QdrantError, match="vector_version"):
            validated_analysis_vector(row)
    assert report.eligible == 2
    assert report.synced == 1
    assert report.skipped_invalid == 1


def test_metric_explanation_uses_grouped_cosines() -> None:
    query = [0.0] * 36
    candidate = [0.0] * 36
    query[0] = candidate[0] = 1.0
    query[17] = candidate[17] = 1.0
    query[22] = candidate[22] = 1.0
    query[27] = candidate[27] = 1.0
    explanation = metric_explanation(query, candidate)
    assert "spectral 1.000" in explanation
    assert "harmonic 1.000" in explanation
    assert "decay/envelope 1.000" in explanation
    assert "stereo/energy 1.000" in explanation
