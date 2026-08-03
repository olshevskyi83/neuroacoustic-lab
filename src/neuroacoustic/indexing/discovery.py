"""Deterministic audio-file discovery for batch indexing."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from neuroacoustic.logging import get_logger

logger = get_logger(__name__)

# Default extensions (case-insensitive). ``wave`` is an alias for WAV.
DEFAULT_INDEX_EXTENSIONS: frozenset[str] = frozenset(
    {"wav", "wave", "flac", "aiff", "aif", "mp3"}
)

_SKIP_NAME_SUFFIXES = (
    ".tmp",
    ".temp",
    ".part",
    ".crdownload",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".sqlite-wal",
    ".sqlite-shm",
    ".db-wal",
    ".db-shm",
)


@dataclass(frozen=True)
class DiscoveryResult:
    """Discovery counters and accepted paths.

    **discovered** — candidate audio files with a supported extension after
    structural skips (hidden/temp names, resolved skip dirs/files, symlink
    policy, realpath dedupe). Include/exclude patterns are **not** applied.

    **supported** / ``files`` — candidates accepted after include/exclude
    (queued for analysis).

    Not counted in either:
    - unsupported extensions (e.g. ``.txt``, ``.json``, ``.png``);
    - hidden / temporary / database-sidecar names;
    - paths under resolved output/skip directories;
    - resolved skip files (database, ``-wal``, ``-shm``, …);
    - symlinks when ``follow_symlinks`` is False.
    """

    discovered: int
    files: list[Path] = field(default_factory=list)

    @property
    def supported(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.files)

    def __len__(self) -> int:
        return len(self.files)


def normalize_extensions(extensions: Iterable[str] | None) -> frozenset[str]:
    if extensions is None:
        return DEFAULT_INDEX_EXTENSIONS
    return frozenset(ext.lower().lstrip(".") for ext in extensions)


def _is_hidden_or_temp(name: str) -> bool:
    lower = name.lower()
    if name.startswith("."):
        return True
    return any(lower.endswith(suf) for suf in _SKIP_NAME_SUFFIXES)


def _matches_any(rel: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    name = Path(rel).name
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
            return True
        if fnmatch.fnmatch(rel, pattern.lstrip("./")):
            return True
    return False


def _is_under(path: Path, ancestor: Path | None) -> bool:
    if ancestor is None:
        return False
    try:
        path.resolve().relative_to(ancestor.resolve())
        return True
    except (ValueError, OSError):
        return False


def discover_audio_files(
    root: Path | str,
    *,
    recursive: bool = True,
    follow_symlinks: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    extensions: Iterable[str] | None = None,
    skip_dirs: Sequence[Path | str] | None = None,
    skip_files: Sequence[Path | str] | None = None,
) -> DiscoveryResult:
    """Discover audio under ``root`` in deterministic order.

    Symlink directories/files are skipped unless ``follow_symlinks`` is True.
    When following, a realpath visitation set prevents directory loops.
    Paths are deduplicated by ``os.path.realpath``.

    Returns a :class:`DiscoveryResult` with distinct ``discovered`` (candidates)
    and ``supported`` (accepted after include/exclude) counts.
    """
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Index root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Index root is not a directory: {root_path}")

    try:
        root_resolved = root_path.resolve()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot resolve index root: {root_path}: {exc}") from exc

    exts = normalize_extensions(extensions)
    include_pats = list(include or [])
    exclude_pats = list(exclude or [])

    skip_dir_resolved: list[Path] = []
    for d in skip_dirs or []:
        try:
            skip_dir_resolved.append(Path(d).expanduser().resolve())
        except OSError:
            continue

    skip_file_resolved: set[Path] = set()
    for f in skip_files or []:
        try:
            skip_file_resolved.add(Path(f).expanduser().resolve())
        except OSError:
            continue

    found: list[Path] = []
    discovered = 0
    seen_real: set[str] = set()
    visited_dirs: set[str] = set()

    def _consider_file(path: Path) -> None:
        nonlocal discovered
        try:
            if path.is_symlink() and not follow_symlinks:
                return
            if not path.is_file():
                return
            name = path.name
            if _is_hidden_or_temp(name):
                return
            ext = path.suffix.lower().lstrip(".")
            if ext not in exts:
                return
            try:
                resolved = path.resolve()
            except OSError:
                return
            if resolved in skip_file_resolved:
                return
            if any(_is_under(resolved, d) for d in skip_dir_resolved):
                return
            key = os.path.realpath(str(resolved))
            if key in seen_real:
                return
            try:
                rel = resolved.relative_to(root_resolved).as_posix()
            except ValueError:
                rel = resolved.as_posix()

            # Candidate with supported extension after structural skips.
            seen_real.add(key)
            discovered += 1

            if exclude_pats and _matches_any(rel, exclude_pats):
                return
            if include_pats and not _matches_any(rel, include_pats):
                return
            found.append(resolved)
        except OSError as exc:
            logger.warning("Skipping inaccessible path %s: %s", path, exc)

    def _walk(current: Path) -> None:
        try:
            real = os.path.realpath(str(current))
        except OSError:
            return
        if real in visited_dirs:
            return
        visited_dirs.add(real)

        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.casefold())
        except PermissionError as exc:
            logger.warning("Permission denied listing %s: %s", current, exc)
            return
        except OSError as exc:
            logger.warning("Cannot list %s: %s", current, exc)
            return

        for entry in entries:
            try:
                if entry.is_symlink() and not follow_symlinks:
                    continue
                if entry.is_dir():
                    if _is_hidden_or_temp(entry.name):
                        continue
                    try:
                        resolved_dir = entry.resolve()
                    except OSError:
                        continue
                    if any(
                        _is_under(resolved_dir, d) or resolved_dir == d
                        for d in skip_dir_resolved
                    ):
                        continue
                    if recursive:
                        _walk(entry)
                    continue
                _consider_file(entry)
            except OSError as exc:
                logger.warning("Skipping entry %s: %s", entry, exc)

    if recursive:
        _walk(root_path)
    else:
        try:
            entries = sorted(root_path.iterdir(), key=lambda p: p.name.casefold())
        except PermissionError as exc:
            raise PermissionError(f"Permission denied: {root_path}: {exc}") from exc
        for entry in entries:
            try:
                if entry.is_symlink() and not follow_symlinks:
                    continue
                if entry.is_dir():
                    continue
                _consider_file(entry)
            except OSError as exc:
                logger.warning("Skipping entry %s: %s", entry, exc)

    found.sort(key=lambda p: str(p).casefold())
    return DiscoveryResult(discovered=discovered, files=found)


__all__ = [
    "DEFAULT_INDEX_EXTENSIONS",
    "DiscoveryResult",
    "discover_audio_files",
    "normalize_extensions",
]
