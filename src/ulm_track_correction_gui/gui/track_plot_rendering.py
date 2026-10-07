"""Pure NumPy rendering helpers for selected-track and dataset ULM overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ulm_track_correction_gui.core.accumulation_checkpoint import (
    AA_DENSITY,
    HARD_DENSITY,
    VELOCITY_TRACK_MAPS,
    RasterCropGeometry,
    raster_crop_geometry,
)

DENSITY_CYAN_RGB = (0, 216, 255)
DEFAULT_ACCUMULATION_GAMMA = 1.0
DEFAULT_ACCUMULATION_CLIM = (0.0, 1.0)


@dataclass(frozen=True)
class RenderedTrackMap:
    rgba: np.ndarray
    geometry: RasterCropGeometry


def matlab_jet(values: np.ndarray) -> np.ndarray:
    """Return the compact MATLAB-like jet mapping used by Batch plotting."""

    values = np.asarray(values, dtype=float)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def render_sparse_track_map(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    map_name: str,
    params: dict,
    ulm_scale_z: float,
    ulm_scale_x: float,
    density_rgb: tuple[int, int, int] = DENSITY_CYAN_RGB,
    velocity_clim: tuple[float, float] | None = None,
) -> RenderedTrackMap | None:
    """Render only a track's occupied bounding box with a transparent background."""

    return _render_sparse_map(
        rows,
        cols,
        values,
        map_name,
        params,
        ulm_scale_z,
        ulm_scale_x,
        density_rgb,
        None,
        velocity_clim,
        dataset_density=False,
    )


def render_sparse_dataset_map(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    map_name: str,
    params: dict,
    ulm_scale_z: float,
    ulm_scale_x: float,
    density_rgb: tuple[int, int, int] = DENSITY_CYAN_RGB,
    velocity_clim: tuple[float, float] | None = None,
    density_clim: tuple[float, float] | None = None,
    gamma: float = DEFAULT_ACCUMULATION_GAMMA,
) -> RenderedTrackMap | None:
    """Render an all-track map after normalization and display compression."""

    return _render_sparse_map(
        rows,
        cols,
        values,
        map_name,
        params,
        ulm_scale_z,
        ulm_scale_x,
        density_rgb,
        density_clim,
        velocity_clim,
        dataset_density=True,
        dataset_gamma=gamma,
    )


def _render_sparse_map(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    map_name: str,
    params: dict,
    ulm_scale_z: float,
    ulm_scale_x: float,
    density_rgb: tuple[int, int, int],
    density_clim: tuple[float, float] | None,
    velocity_clim: tuple[float, float] | None,
    *,
    dataset_density: bool,
    dataset_gamma: float = DEFAULT_ACCUMULATION_GAMMA,
) -> RenderedTrackMap | None:
    """Render occupied pixels into one cropped transparent RGBA raster."""

    rows = np.asarray(rows).reshape(-1)
    cols = np.asarray(cols).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if not (len(rows) == len(cols) == len(values)):
        raise ValueError("track-map rows, columns, and values must align")
    if len(rows) == 0:
        return None
    if np.any(~np.isfinite(values)):
        raise ValueError("track-map values must be finite")

    geometry = raster_crop_geometry(rows, cols, ulm_scale_z, ulm_scale_x)
    rgba = np.zeros((geometry.height, geometry.width, 4), dtype=np.uint8)
    local_rows = rows.astype(np.int64) - geometry.row0
    local_cols = cols.astype(np.int64) - geometry.col0

    if map_name in {HARD_DENSITY, AA_DENSITY}:
        density_color = np.asarray(density_rgb, dtype=np.int64)
        if density_color.shape != (3,) or np.any((density_color < 0) | (density_color > 255)):
            raise ValueError("density RGB must contain three values from 0 to 255")
        colors = np.tile(
            density_color.astype(np.uint8),
            (len(values), 1),
        )

    if dataset_density:
        clim = (
            density_clim
            if map_name in {HARD_DENSITY, AA_DENSITY}
            else velocity_clim
        )
        displayed = normalize_accumulation_values(values, dataset_gamma, clim)
        if map_name in {HARD_DENSITY, AA_DENSITY}:
            alpha = np.round(displayed * 255.0).astype(np.uint8)
        elif map_name in VELOCITY_TRACK_MAPS:
            colors = np.round(matlab_jet(displayed) * 255.0).astype(np.uint8)
            alpha = np.full(len(values), 255, dtype=np.uint8)
        else:
            raise ValueError(f"unknown dataset accumulation map: {map_name}")
    elif map_name == HARD_DENSITY:
        alpha = np.where(values > 0, 255, 0).astype(np.uint8)
    elif map_name == AA_DENSITY:
        power = float(params["counterPower"])
        powered = np.maximum(values, 0.0) ** power
        maximum = float(powered.max())
        normalized = powered / maximum if maximum > 0 else np.zeros_like(powered)
        alpha = np.round(normalized * 255.0).astype(np.uint8)
    elif map_name in VELOCITY_TRACK_MAPS:
        minimum, maximum = (
            (0.0, float(params["velocityDisplayMax"]))
            if velocity_clim is None
            else (float(velocity_clim[0]), float(velocity_clim[1]))
        )
        if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
            raise ValueError("velocity color limits must be finite with minimum < maximum")
        normalized = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
        colors = np.round(matlab_jet(normalized) * 255.0).astype(np.uint8)
        alpha = np.full(len(values), 255, dtype=np.uint8)
    else:
        raise ValueError(f"unknown selected-track map: {map_name}")

    rgba[local_rows, local_cols, :3] = colors
    rgba[local_rows, local_cols, 3] = alpha
    rgba.setflags(write=False)
    return RenderedTrackMap(rgba=rgba, geometry=geometry)


def normalize_accumulation_values(
    values: np.ndarray,
    gamma: float = DEFAULT_ACCUMULATION_GAMMA,
    clim: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize one accumulation map to [0, 1], then apply Gamma and Clim.

    The raw map remains untouched. Its largest non-negative occupied value is
    the normalization reference, so multiplying every raw value by a constant
    does not change the rendered result. Clim is intentionally allowed above
    one even though the normalized data itself never exceeds one.
    """

    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("accumulation values must be finite")

    gamma = float(gamma)
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("accumulation Gamma must be finite and in [0, 1]")

    minimum, maximum = (
        DEFAULT_ACCUMULATION_CLIM
        if clim is None
        else (float(clim[0]), float(clim[1]))
    )
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
        raise ValueError("accumulation Clim must be finite with minimum < maximum")
    if minimum < 0.0:
        raise ValueError("accumulation Clim minimum cannot be negative")

    non_negative = np.maximum(values, 0.0)
    reference = float(non_negative.max()) if non_negative.size else 0.0
    normalized = (
        non_negative / reference if reference > 0.0 else np.zeros_like(non_negative)
    )
    if gamma == 0.0:
        powered = np.zeros_like(normalized)
        powered[normalized > 0.0] = 1.0
    else:
        powered = np.power(normalized, gamma)
    return np.clip((powered - minimum) / (maximum - minimum), 0.0, 1.0)


__all__ = [
    "DEFAULT_ACCUMULATION_CLIM",
    "DEFAULT_ACCUMULATION_GAMMA",
    "DENSITY_CYAN_RGB",
    "RenderedTrackMap",
    "matlab_jet",
    "normalize_accumulation_values",
    "render_sparse_dataset_map",
    "render_sparse_track_map",
]
