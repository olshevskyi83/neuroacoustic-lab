"""Atomic filesystem helpers (temp file + ``os.replace``)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` via a same-directory temp file and replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target.resolve()


def atomic_replace_file(tmp_path: Path | str, final_path: Path | str) -> Path:
    """Replace ``final_path`` with an already-written ``tmp_path`` (same filesystem)."""
    target = Path(final_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(tmp_path), target)
    return target.resolve()


__all__ = ["atomic_replace_file", "atomic_write_text"]
