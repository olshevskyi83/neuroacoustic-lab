"""Numeric helpers: finite sanitization and distribution statistics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


def to_mono(samples: NDArray[np.floating]) -> NDArray[np.float64]:
    """Average channels to mono for spectral/energy analysis."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return np.mean(arr, axis=1)
    raise ValueError(f"Expected 1-D or 2-D samples, got shape {arr.shape}")


def finite_or(
    value: float | np.floating | None,
    default: float | None = None,
) -> float | None:
    """Return ``value`` if finite, otherwise ``default``."""
    if value is None:
        return default
    v = float(value)
    return v if np.isfinite(v) else default


def sanitize_array(values: NDArray[np.floating], fill: float = 0.0) -> NDArray[np.float64]:
    """Replace non-finite values with ``fill``."""
    out = np.asarray(values, dtype=np.float64).copy()
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = fill
    return out


def downsample_series(
    times: NDArray[np.floating],
    values: NDArray[np.floating],
    max_points: int,
) -> tuple[list[float], list[float]]:
    """Uniformly downsample a time series for JSON export."""
    t = np.asarray(times, dtype=np.float64)
    v = sanitize_array(np.asarray(values, dtype=np.float64))
    n = t.size
    if n == 0:
        return [], []
    if n <= max_points:
        return [float(x) for x in t], [float(x) for x in v]
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return [float(x) for x in t[idx]], [float(x) for x in v[idx]]


class DistributionStats(BaseModel):
    """Summary statistics for a scalar time series.

    Aggregation: computed over valid (finite) frame-level observations.
    Missing / empty series yield null fields rather than NaN.
    """

    mean: float | None = None
    std: float | None = None
    median: float | None = None
    p05: float | None = Field(default=None, description="5th percentile")
    p25: float | None = Field(default=None, description="25th percentile")
    p75: float | None = Field(default=None, description="75th percentile")
    p95: float | None = Field(default=None, description="95th percentile")
    minimum: float | None = None
    maximum: float | None = None
    count: int = 0

    @classmethod
    def from_values(cls, values: Sequence[float] | NDArray[np.floating]) -> DistributionStats:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return cls(count=0)
        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            median=float(np.median(arr)),
            p05=float(np.percentile(arr, 5)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            minimum=float(np.min(arr)),
            maximum=float(np.max(arr)),
            count=int(arr.size),
        )


class TimeSeriesSummary(BaseModel):
    """Downsampled curve retained in the fingerprint (not full-resolution)."""

    times_seconds: list[float] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    unit: str
    description: str = ""
