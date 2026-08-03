"""Index batch report models and atomic writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from neuroacoustic.io_atomic import atomic_write_text

FileDisposition = Literal["analyzed", "reused", "forced", "failed", "skipped"]


class FileIndexResult(BaseModel):
    """Per-file outcome in an index run."""

    path: str
    disposition: FileDisposition
    analysis_id: int | None = None
    content_hash: str | None = None
    duration_seconds: float | None = None
    elapsed_seconds: float | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class IndexReport(BaseModel):
    """Deterministic summary of a batch index run.

    Counter semantics
    -----------------
    - ``discovered``: candidate audio files (supported extension) after
      structural skips; include/exclude not applied.
    - ``supported``: candidates accepted after include/exclude (= ``len(files)``).
    - ``analyzed`` / ``reused`` / ``forced`` / ``failed`` / ``skipped``:
      dispositions for each supported file (must sum to ``supported``).
    """

    root_directory: str
    database_path: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    recursive: bool
    follow_symlinks: bool
    workers: int
    force: bool
    plots: bool
    continue_on_error: bool
    interrupted: bool = False
    discovered: int
    supported: int
    analyzed: int
    reused: int
    forced: int
    failed: int
    skipped: int
    schema_version: str
    analysis_version: str
    vector_version: str
    config_hash: str
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    files: list[FileIndexResult] = Field(default_factory=list)

    def counts_consistent(self) -> bool:
        """Return True when per-file dispositions match summary counters."""
        tallies = {"analyzed": 0, "reused": 0, "forced": 0, "failed": 0, "skipped": 0}
        for item in self.files:
            tallies[item.disposition] += 1
        return (
            tallies["analyzed"] == self.analyzed
            and tallies["reused"] == self.reused
            and tallies["forced"] == self.forced
            and tallies["failed"] == self.failed
            and tallies["skipped"] == self.skipped
            and len(self.files) == self.supported
            and self.discovered >= self.supported
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_report_atomic(report: IndexReport, path: Path | str) -> Path:
    """Write report JSON atomically (temp file + ``os.replace``)."""
    target = Path(path).expanduser()
    payload = report.model_dump(mode="json")
    data = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    return atomic_write_text(target, data)


def truncate_error(message: str, *, limit: int = 500) -> str:
    text = message.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def assert_json_finite(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert_json_finite(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            assert_json_finite(value, f"{path}[{i}]")
    elif isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError(f"Non-finite float at {path}: {obj!r}")


__all__ = [
    "FileDisposition",
    "FileIndexResult",
    "IndexReport",
    "assert_json_finite",
    "truncate_error",
    "write_report_atomic",
    "_utc_now_iso",
]
