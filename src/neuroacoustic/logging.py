"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False


def setup_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure root logger for the CLI.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``INFO``).
    json_logs:
        When True, emit single-line ``key=value`` style messages suitable for
        later JSON wrapping; Milestone 1 keeps a simple text formatter either way.
    """
    global _CONFIGURED
    root = logging.getLogger()
    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric)

    handler = logging.StreamHandler(sys.stderr)
    if json_logs:
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    handler.setFormatter(formatter)

    # Replace existing handlers so repeated CLI calls stay consistent.
    root.handlers.clear()
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (call :func:`setup_logging` from the CLI first)."""
    return logging.getLogger(name)


def log_extra(**fields: Any) -> str:
    """Format structured fields for inclusion in a log message."""
    return " ".join(f"{key}={value!r}" for key, value in fields.items())
