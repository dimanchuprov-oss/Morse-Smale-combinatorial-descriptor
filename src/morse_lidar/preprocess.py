"""Depth-map validation and optional sensor-noise preprocessing."""

from __future__ import annotations

import numpy as np


def load_depth(path: str) -> np.ndarray:
    field = np.load(path)
    field = np.asarray(field, dtype=float)
    _validate_depth(field)
    return field


def denoise_depth(depth: np.ndarray, median_size: int = 3, sigma: float = 0.0, *, periodic: bool = False) -> np.ndarray:
    """Apply optional sensor smoothing; this is not persistence cancellation."""
    result = np.asarray(depth, dtype=float)
    _validate_depth(result)
    if median_size < 1 or median_size % 2 == 0:
        raise ValueError("median_size must be a positive odd integer")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be a finite non-negative number")
    if median_size == 1 and sigma == 0:
        return result.copy()
    try:
        from scipy.ndimage import gaussian_filter, median_filter
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional signal extra for depth smoothing") from error
    if median_size > 1:
        result = median_filter(result, size=median_size, mode="wrap" if periodic else "nearest")
    if sigma > 0:
        result = gaussian_filter(result, sigma=sigma, mode="wrap" if periodic else "nearest")
    return result


def _validate_depth(field: np.ndarray) -> None:
    if field.ndim != 2:
        raise ValueError("depth input must be a 2D NumPy array")
    if not np.isfinite(field).all():
        raise ValueError("depth input contains NaN or infinity")
