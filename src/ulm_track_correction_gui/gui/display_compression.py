"""Intensity-compression math for the bubble-movie display."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ulm_track_correction_gui.gui.layer_styles import DEFAULT_DB_MAX, DEFAULT_DB_MIN

DISPLAY_MODE_DB = "db"
DISPLAY_MODE_POWER = "power"
DEFAULT_DISPLAY_MODE = DISPLAY_MODE_POWER
DEFAULT_POWER_GAMMA = 1.0
DEFAULT_POWER_CLIM_MIN = 0.1
DEFAULT_POWER_CLIM_MAX = 0.7


@dataclass(frozen=True)
class DisplaySettings:
    """Display-only compression state shared by the panel and viewer."""

    mode: str = DEFAULT_DISPLAY_MODE
    db_min: float = DEFAULT_DB_MIN
    db_max: float = DEFAULT_DB_MAX
    gamma: float = DEFAULT_POWER_GAMMA
    clim_min: float = DEFAULT_POWER_CLIM_MIN
    clim_max: float = DEFAULT_POWER_CLIM_MAX


def _finite_magnitude(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values).astype(np.float32, copy=False)
    if not np.isfinite(magnitude).all():
        magnitude = np.where(np.isfinite(magnitude), magnitude, 0.0)
    return magnitude


def frame_maximum_magnitudes(stack: np.ndarray) -> np.ndarray:
    """Return sanitized ``max(abs(frame))`` values without copying the stack."""

    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("Display references require a 3-D (z, x, frames) stack.")
    if any(size == 0 for size in stack.shape):
        raise ValueError("Display references require a non-empty stack.")

    maxima = np.empty(stack.shape[2], dtype=np.float64)
    for frame_idx in range(stack.shape[2]):
        magnitude = np.abs(stack[:, :, frame_idx])
        if not np.isfinite(magnitude).all():
            magnitude = np.where(np.isfinite(magnitude), magnitude, 0.0)
        maxima[frame_idx] = float(np.max(magnitude))
    return maxima


def display_references(stack: np.ndarray) -> tuple[float, float]:
    """Return the dB maximum and Power Law ensemble reference for one stack."""

    frame_maxima = frame_maximum_magnitudes(stack)
    return float(np.max(frame_maxima)), float(np.median(frame_maxima))


def normalize_db_frame(
    frame: np.ndarray,
    reference: float,
    db_min: float = DEFAULT_DB_MIN,
    db_max: float = DEFAULT_DB_MAX,
) -> np.ndarray:
    """Map one amplitude frame to uint8 through logarithmic compression."""

    magnitude = _finite_magnitude(frame)
    reference = float(reference)
    if not np.isfinite(reference) or reference <= 0.0:
        reference = 1.0

    normalized = magnitude / reference
    compressed = 20.0 * np.log10(normalized + np.finfo(np.float32).eps)
    compressed = np.clip(compressed, db_min, db_max)

    span = float(db_max) - float(db_min)
    if span <= 0.0:
        span = 1.0
    display = (compressed - float(db_min)) / span
    return np.clip(display * 255.0, 0.0, 255.0).astype(np.uint8)


def normalize_power_frame(
    frame: np.ndarray,
    reference: float,
    gamma: float,
    clim_min: float,
    clim_max: float,
) -> np.ndarray:
    """Map one amplitude frame to uint8 through Power Law compression.

    ``reference`` is the median of the stack's per-frame maximum magnitudes.
    Values above it saturate before the selected ``[0, 1]`` Clim is applied.
    """

    reference = float(reference)
    if not np.isfinite(reference) or reference <= 0.0:
        reference = 1.0

    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Power Law gamma must be in [0, 1].")

    clim_min = float(clim_min)
    clim_max = float(clim_max)
    if not 0.0 <= clim_min < clim_max <= 1.0:
        raise ValueError("Power Law Clim must satisfy 0 <= min < max <= 1.")

    normalized = np.clip(_finite_magnitude(frame) / reference, 0.0, 1.0)
    if gamma == 0.0:
        compressed = np.zeros_like(normalized)
        compressed[normalized > 0.0] = 1.0
    else:
        compressed = np.power(normalized, gamma)

    display = (compressed - clim_min) / (clim_max - clim_min)
    return np.clip(display * 255.0, 0.0, 255.0).astype(np.uint8)


def normalize_display_frame(
    frame: np.ndarray,
    *,
    settings: DisplaySettings,
    db_reference: float,
    power_reference: float,
) -> np.ndarray:
    """Normalize one frame with the compression selected in ``settings``."""

    if settings.mode == DISPLAY_MODE_DB:
        return normalize_db_frame(
            frame,
            db_reference,
            settings.db_min,
            settings.db_max,
        )
    if settings.mode == DISPLAY_MODE_POWER:
        return normalize_power_frame(
            frame,
            power_reference,
            settings.gamma,
            settings.clim_min,
            settings.clim_max,
        )
    raise ValueError(f"Unknown display compression mode: {settings.mode}")
