"""Dynamic PALA Fixed accumulation of the current correction-session tracks.

The Batch checkpoint overlay is intentionally immutable.  This module provides
the complementary view requested by correction work: rebuild the ULM maps from
the *current* ``tracks`` and update only tracks touched by later edits.

The numerical pipeline mirrors ``ULM_Python_GUI``'s PALA Fixed backend:
endpoint-preserving moving-average smoothing, dense segment subdivision in the
anisotropic ULM grid, per-track Hard occupancy, and a sampled 3x3 Gaussian AA
density splat.  It is Qt-free and never mutates the correction session.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import chain
from math import sqrt
from typing import TYPE_CHECKING, Iterable

import numpy as np

from ulm_track_correction_gui.core.accumulation_checkpoint import (
    AA_DENSITY,
    HARD_DENSITY,
    HARD_LOCAL_VELOCITY,
    HARD_TRACK_MEAN_VELOCITY,
    AccumulationCheckpoint,
)
from ulm_track_correction_gui.core.pala_contracts import ensure_mat_tracking

if TYPE_CHECKING:
    from ulm_track_correction_gui.core.correction_session import CorrectionSession


NEIGHBOR_OFFSETS = np.asarray(
    [
        (z_offset, x_offset)
        for z_offset in (-1, 0, 1)
        for x_offset in (-1, 0, 1)
    ],
    dtype=np.int64,
)


def _positive_float(value: object, name: str) -> float:
    output = float(value)
    if not np.isfinite(output) or output <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return output


def _positive_int(value: object, name: str) -> int:
    numeric = float(value)
    output = int(numeric)
    if not np.isfinite(numeric) or output < 1 or output != numeric:
        raise ValueError(f"{name} must be a positive integer")
    return output


@dataclass(frozen=True)
class DatasetAccumulationSettings:
    """Validated PALA Fixed grid, numerical, and display parameters."""

    input_shape: tuple[int, int]
    map_shape: tuple[int, int]
    input_pixel_size_z_um: float
    input_pixel_size_x_um: float
    ulm_pixel_size_z_um: float
    ulm_pixel_size_x_um: float
    ulm_scale_z: float
    ulm_scale_x: float
    moving_average_span: int
    max_sampling_step_ulm_px: float
    prf_hz: float
    anti_alias_size: float
    counter_power: float
    counter_display_max: float
    counter_aa_display_max: float
    velocity_display_max: float

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: AccumulationCheckpoint,
    ) -> DatasetAccumulationSettings:
        """Read the exact Batch grid and PALA parameters used for accumulation."""

        params = checkpoint.params
        metadata = checkpoint.metadata

        def required(source: dict, name: str) -> object:
            if name not in source or source[name] is None:
                raise ValueError(
                    f"Batch checkpoint does not contain required accumulation "
                    f"parameter {name}."
                )
            return source[name]

        input_z = _positive_float(
            required(metadata, "inputPixelSizeZUm"), "inputPixelSizeZUm"
        )
        input_x = _positive_float(
            required(metadata, "inputPixelSizeXUm"), "inputPixelSizeXUm"
        )
        ulm_z = _positive_float(
            required(metadata, "ulmPixelSizeZUm"), "ulmPixelSizeZUm"
        )
        ulm_x = _positive_float(
            required(metadata, "ulmPixelSizeXUm"), "ulmPixelSizeXUm"
        )
        scale_z = input_z / ulm_z
        scale_x = input_x / ulm_x
        if not np.isclose(scale_z, checkpoint.ulm_scale_z, rtol=1e-8, atol=1e-12):
            raise ValueError("checkpoint axial pixel sizes do not match ulmScaleZ")
        if not np.isclose(scale_x, checkpoint.ulm_scale_x, rtol=1e-8, atol=1e-12):
            raise ValueError("checkpoint lateral pixel sizes do not match ulmScaleX")

        return cls(
            input_shape=checkpoint.input_shape,
            map_shape=checkpoint.map_shape,
            input_pixel_size_z_um=input_z,
            input_pixel_size_x_um=input_x,
            ulm_pixel_size_z_um=ulm_z,
            ulm_pixel_size_x_um=ulm_x,
            ulm_scale_z=scale_z,
            ulm_scale_x=scale_x,
            moving_average_span=_positive_int(
                required(params, "movingAverageSpan"), "movingAverageSpan"
            ),
            max_sampling_step_ulm_px=_positive_float(
                required(params, "maxSamplingStepUlmpx"), "maxSamplingStepUlmpx"
            ),
            prf_hz=_positive_float(required(params, "prfHz"), "prfHz"),
            anti_alias_size=_positive_float(
                required(params, "antiAliasSize"), "antiAliasSize"
            ),
            counter_power=_positive_float(
                required(params, "counterPower"), "counterPower"
            ),
            counter_display_max=_positive_float(
                required(params, "counterDisplayMax"), "counterDisplayMax"
            ),
            counter_aa_display_max=_positive_float(
                required(params, "counterAADisplayMax"), "counterAADisplayMax"
            ),
            velocity_display_max=_positive_float(
                required(params, "velocityDisplayMax"), "velocityDisplayMax"
            ),
        )

    @property
    def display_params(self) -> dict[str, float]:
        return {
            "counterPower": self.counter_power,
            "counterDisplayMax": self.counter_display_max,
            "counterAADisplayMax": self.counter_aa_display_max,
            "velocityDisplayMax": self.velocity_display_max,
        }

    def incompatible_fields(
        self,
        other: DatasetAccumulationSettings,
    ) -> tuple[str, ...]:
        """Return parameters that prevent two datasets sharing one ULM map."""

        mismatches: list[str] = []
        for field in fields(self):
            left = getattr(self, field.name)
            right = getattr(other, field.name)
            if isinstance(left, float):
                matches = bool(np.isclose(left, right, rtol=1e-8, atol=1e-12))
            else:
                matches = left == right
            if not matches:
                mismatches.append(field.name)
        return tuple(mismatches)


@dataclass(frozen=True)
class TrackRasterContribution:
    """Sparse sufficient statistics contributed by one current track."""

    hard_linear: np.ndarray
    hard_local_velocity: np.ndarray
    hard_track_mean_velocity: np.ndarray
    aa_linear: np.ndarray
    aa_density_weight: np.ndarray


def pala_moving_average(coords: np.ndarray, span: int) -> np.ndarray:
    """Match MATLAB ``smooth(..., span)`` endpoint behavior used by PALA."""

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_points, 2)")
    if len(coords) < 2:
        return coords.copy()

    span = min(_positive_int(span, "movingAverageSpan"), len(coords))
    if span % 2 == 0:
        span -= 1
    if span <= 1:
        return coords.copy()

    half_window = span // 2
    smoothed = np.empty_like(coords)
    for index in range(len(coords)):
        endpoint_half = min(index, len(coords) - 1 - index)
        half = min(half_window, endpoint_half)
        smoothed[index] = coords[index - half : index + half + 1].mean(axis=0)
    return smoothed


def subdivide_track_segments(
    coords_ulm: np.ndarray,
    frames: np.ndarray,
    segment_speeds: np.ndarray,
    max_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide each straight segment and retain the final endpoint."""

    coords_ulm = np.asarray(coords_ulm, dtype=float)
    frames = np.asarray(frames, dtype=float)
    segment_speeds = np.asarray(segment_speeds, dtype=float)
    max_step = _positive_float(max_step, "maxSamplingStepUlmpx")
    if len(coords_ulm) == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    if len(coords_ulm) == 1:
        return coords_ulm.copy(), np.zeros(1, dtype=float)
    if len(segment_speeds) != len(coords_ulm) - 1:
        raise ValueError("segment_speeds must contain one value per segment")

    coord_parts: list[np.ndarray] = []
    speed_parts: list[np.ndarray] = []
    for index in range(len(coords_ulm) - 1):
        start = coords_ulm[index]
        stop = coords_ulm[index + 1]
        subdivisions = max(
            1,
            int(np.ceil(float(np.linalg.norm(stop - start)) / max_step)),
        )
        fractions = np.linspace(0.0, 1.0, subdivisions + 1)
        if index > 0:
            fractions = fractions[1:]
        coord_parts.append(start + fractions[:, None] * (stop - start))
        speed_parts.append(np.full(len(fractions), segment_speeds[index]))

    return np.concatenate(coord_parts), np.concatenate(speed_parts)


def gaussian_anti_alias_weights(
    interp_coords: np.ndarray,
    rounded_coords: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    """Return normalized sampled-Gaussian weights over a 3x3 neighborhood."""

    interp_coords = np.asarray(interp_coords, dtype=float)
    rounded_coords = np.asarray(rounded_coords, dtype=np.int64)
    if interp_coords.shape != rounded_coords.shape or interp_coords.shape[1:] != (2,):
        raise ValueError("coordinate arrays must both have shape (n_points, 2)")
    sigma = _positive_float(fwhm, "antiAliasSize") / (
        2.0 * sqrt(2.0 * np.log(2.0))
    )
    delta = (
        rounded_coords[:, None, :]
        + NEIGHBOR_OFFSETS[None, :, :]
        - interp_coords[:, None, :]
    )
    weights = np.exp(-0.5 * np.sum((delta / sigma) ** 2, axis=2))
    totals = weights.sum(axis=1, keepdims=True)
    np.divide(weights, totals, out=weights, where=totals > 0.0)
    return weights


def _empty_contribution() -> TrackRasterContribution:
    return TrackRasterContribution(
        hard_linear=np.empty(0, dtype=np.int64),
        hard_local_velocity=np.empty(0, dtype=np.float32),
        hard_track_mean_velocity=np.empty(0, dtype=np.float32),
        aa_linear=np.empty(0, dtype=np.int64),
        aa_density_weight=np.empty(0, dtype=np.float32),
    )


class DatasetAccumulator:
    """Cached all-track maps with per-track incremental replacement."""

    def __init__(self, settings: DatasetAccumulationSettings) -> None:
        self.settings = settings
        if np.prod(settings.map_shape, dtype=np.int64) > 100_000_000:
            raise ValueError(
                f"requested ULM map {settings.map_shape} is too large; "
                "increase the ULM pixel size in Batch Processing"
            )
        self.density = np.zeros(settings.map_shape, dtype=np.float32)
        self.density_aa = np.zeros(settings.map_shape, dtype=np.float32)
        self.local_velocity_sum = np.zeros(settings.map_shape, dtype=np.float32)
        self.track_velocity_sum = np.zeros(settings.map_shape, dtype=np.float32)
        self._contributions: dict[int, TrackRasterContribution] = {}
        self._track_count = 0
        self._dataset_count = 0
        self.revision = 0

    @classmethod
    def build(
        cls,
        session: CorrectionSession,
        checkpoint: AccumulationCheckpoint,
    ) -> DatasetAccumulator:
        settings = DatasetAccumulationSettings.from_checkpoint(checkpoint)
        if session.image_stack is not None:
            actual_shape = tuple(int(value) for value in session.image_stack.shape[:2])
            if actual_shape != settings.input_shape:
                raise ValueError(
                    f"current dataset shape {actual_shape} does not match the "
                    f"accumulation grid {settings.input_shape}"
                )
        accumulator = cls(settings)
        accumulator.rebuild(session)
        return accumulator

    @classmethod
    def build_multiple(
        cls,
        datasets: Iterable[tuple[CorrectionSession, AccumulationCheckpoint]],
    ) -> DatasetAccumulator:
        """Accumulate independent sessions on one compatible PALA ULM grid.

        The iterable is consumed once so callers can load large MAT files
        lazily.  No per-track replacement cache is retained because this
        snapshot changes only when the user explicitly imports it again.
        """

        iterator = iter(datasets)
        try:
            first_session, first_checkpoint = next(iterator)
        except StopIteration as exc:
            raise ValueError("Select at least one dataset to accumulate.") from exc

        settings = DatasetAccumulationSettings.from_checkpoint(first_checkpoint)
        accumulator = cls(settings)
        for dataset_index, (session, checkpoint) in enumerate(
            chain(((first_session, first_checkpoint),), iterator),
            start=1,
        ):
            candidate_settings = DatasetAccumulationSettings.from_checkpoint(checkpoint)
            mismatches = settings.incompatible_fields(candidate_settings)
            if mismatches:
                raise ValueError(
                    f"dataset {dataset_index} has incompatible accumulation "
                    f"settings: {', '.join(mismatches)}"
                )
            cls._validate_session_shape(session, settings, dataset_index=dataset_index)
            for track_id in session.active_track_ids:
                accumulator._apply(
                    accumulator._rasterize_track(session, track_id),
                    1.0,
                )
            accumulator._track_count += len(session.active_nonempty_track_ids)
            accumulator._dataset_count += 1

        accumulator.revision += 1
        return accumulator

    @staticmethod
    def _validate_session_shape(
        session: CorrectionSession,
        settings: DatasetAccumulationSettings,
        *,
        dataset_index: int | None = None,
    ) -> None:
        if session.image_stack is None:
            return
        actual_shape = tuple(int(value) for value in session.image_stack.shape[:2])
        if actual_shape != settings.input_shape:
            subject = (
                f"dataset {dataset_index}" if dataset_index is not None else "current dataset"
            )
            raise ValueError(
                f"{subject} shape {actual_shape} does not match the "
                f"accumulation grid {settings.input_shape}"
            )

    @property
    def track_count(self) -> int:
        return self._track_count

    @property
    def dataset_count(self) -> int:
        return self._dataset_count

    def rebuild(self, session: CorrectionSession) -> None:
        """Build every current track once; later edits use ``update_tracks``."""

        contributions = {
            track_id: self._rasterize_track(session, track_id)
            for track_id in session.active_track_ids
        }
        self.density.fill(0.0)
        self.density_aa.fill(0.0)
        self.local_velocity_sum.fill(0.0)
        self.track_velocity_sum.fill(0.0)
        self._contributions.clear()
        for track_id, contribution in contributions.items():
            self._apply(contribution, 1.0)
            self._contributions[track_id] = contribution
        self._track_count = len(session.active_nonempty_track_ids)
        self._dataset_count = 1
        self.revision += 1

    def update_tracks(
        self,
        session: CorrectionSession,
        track_ids: Iterable[int],
    ) -> None:
        """Atomically replace sparse contributions for only the edited tracks."""

        ids = sorted({int(track_id) for track_id in track_ids})
        if not ids:
            return
        for track_id in ids:
            if track_id < 0:
                raise IndexError(f"track_id out of range: {track_id}")
        replacements = {
            track_id: (
                self._rasterize_track(session, track_id)
                if track_id < session.n_tracks and session.is_track_active(track_id)
                else _empty_contribution()
            )
            for track_id in ids
        }
        removed = [
            self._contributions.get(track_id, _empty_contribution()) for track_id in ids
        ]
        for contribution in removed:
            self._apply(contribution, -1.0)
        for track_id, contribution in replacements.items():
            self._apply(contribution, 1.0)
            if track_id < session.n_tracks and session.is_track_active(track_id):
                self._contributions[track_id] = contribution
            else:
                self._contributions.pop(track_id, None)
        self._clean_touched(removed + list(replacements.values()))
        self._track_count = len(session.active_nonempty_track_ids)
        self.revision += 1

    def map_values(self, map_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sparse rows, columns, and current global-map values."""

        map_name = str(map_name)
        if map_name == HARD_DENSITY:
            support = self.density > 0.0
            values = self.density[support]
        elif map_name == AA_DENSITY:
            support = self.density_aa > 0.0
            values = self.density_aa[support]
        elif map_name == HARD_LOCAL_VELOCITY:
            support = self.density > 0.0
            values = np.divide(
                self.local_velocity_sum[support],
                self.density[support],
            )
        elif map_name == HARD_TRACK_MEAN_VELOCITY:
            support = self.density > 0.0
            values = np.divide(
                self.track_velocity_sum[support],
                self.density[support],
            )
        else:
            raise ValueError(f"unknown dataset accumulation map: {map_name}")
        rows, cols = np.nonzero(support)
        return (
            rows.astype(np.int64, copy=False),
            cols.astype(np.int64, copy=False),
            np.asarray(values, dtype=np.float32),
        )

    def _track_points(
        self,
        session: CorrectionSession,
        track_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        track = np.asarray(session.tracks[track_id], dtype=float).reshape(-1)
        frames = np.flatnonzero(~np.isnan(track))
        if len(frames) == 0:
            return frames.astype(float), np.empty((0, 2), dtype=float)

        coords = np.empty((len(frames), 2), dtype=float)
        for row_index, frame_idx_raw in enumerate(frames):
            frame_idx = int(frame_idx_raw)
            raw_local_idx = float(track[frame_idx])
            if not np.isfinite(raw_local_idx):
                raise ValueError(
                    f"track {track_id} frame {frame_idx + 1} has non-finite "
                    f"local_idx {raw_local_idx}"
                )
            local_idx = int(raw_local_idx)
            if raw_local_idx < 0 or local_idx != raw_local_idx:
                raise ValueError(
                    f"track {track_id} frame {frame_idx + 1} has invalid local_idx "
                    f"{raw_local_idx}"
                )
            localizations = ensure_mat_tracking(
                session.localized_by_frame.get(frame_idx, [])
            )
            if local_idx >= len(localizations):
                raise ValueError(
                    f"track {track_id} frame {frame_idx + 1} references local_idx "
                    f"{local_idx}, but the frame contains {len(localizations)} rows"
                )
            z, x = localizations[local_idx, 1:3]
            if not np.isfinite(z) or not np.isfinite(x):
                raise ValueError(
                    f"track {track_id} frame {frame_idx + 1} has non-finite coordinates"
                )
            coords[row_index] = (z, x)
        return frames.astype(float), coords

    def _rasterize_track(
        self,
        session: CorrectionSession,
        track_id: int,
    ) -> TrackRasterContribution:
        frames, coords_pixels = self._track_points(session, track_id)
        if len(frames) == 0:
            return _empty_contribution()

        settings = self.settings
        input_pixel_mm = np.asarray(
            [settings.input_pixel_size_z_um, settings.input_pixel_size_x_um],
            dtype=float,
        ) / 1000.0
        ulm_pixel_mm = np.asarray(
            [settings.ulm_pixel_size_z_um, settings.ulm_pixel_size_x_um],
            dtype=float,
        ) / 1000.0
        smoothed_mm = pala_moving_average(
            (coords_pixels - 1.0) * input_pixel_mm,
            settings.moving_average_span,
        )
        if len(smoothed_mm) < 2:
            segment_speeds = np.empty(0, dtype=float)
            track_mean_speed = 0.0
        else:
            frame_steps = np.diff(frames)
            if np.any(frame_steps <= 0.0):
                raise ValueError(f"track {track_id} frames must be strictly increasing")
            distances = np.linalg.norm(np.diff(smoothed_mm, axis=0), axis=1)
            segment_speeds = distances * settings.prf_hz / frame_steps
            elapsed_frames = frames[-1] - frames[0]
            track_mean_speed = (
                float(distances.sum() * settings.prf_hz / elapsed_frames)
                if elapsed_frames > 0.0
                else 0.0
            )

        interp_coords, interp_speeds = subdivide_track_segments(
            smoothed_mm / ulm_pixel_mm,
            frames,
            segment_speeds,
            settings.max_sampling_step_ulm_px,
        )
        rounded = np.floor(interp_coords + 0.5).astype(np.int64)
        in_map = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < settings.map_shape[0])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < settings.map_shape[1])
        )
        interp_coords = interp_coords[in_map]
        interp_speeds = interp_speeds[in_map]
        rounded = rounded[in_map]
        if len(rounded) == 0:
            return _empty_contribution()

        hard_linear = np.ravel_multi_index(rounded.T, settings.map_shape)
        hard_linear, inverse = np.unique(hard_linear, return_inverse=True)
        speed_sum = np.bincount(inverse, weights=interp_speeds)
        speed_count = np.bincount(inverse)
        hard_local = np.asarray(speed_sum / speed_count, dtype=np.float32)
        hard_track = np.full(len(hard_linear), track_mean_speed, dtype=np.float32)

        aa_weights = gaussian_anti_alias_weights(
            interp_coords,
            rounded,
            settings.anti_alias_size,
        ).reshape((-1, 9))
        neighbor_coords = (
            rounded[:, None, :] + NEIGHBOR_OFFSETS[None, :, :]
        ).reshape((-1, 2))
        flat_weights = aa_weights.reshape(-1)
        valid = (
            (neighbor_coords[:, 0] >= 0)
            & (neighbor_coords[:, 0] < settings.map_shape[0])
            & (neighbor_coords[:, 1] >= 0)
            & (neighbor_coords[:, 1] < settings.map_shape[1])
            & (flat_weights > 0.0)
        )
        aa_linear = np.ravel_multi_index(
            neighbor_coords[valid].T,
            settings.map_shape,
        )
        aa_linear, aa_inverse = np.unique(aa_linear, return_inverse=True)
        aa_density = np.asarray(
            np.bincount(aa_inverse, weights=flat_weights[valid]),
            dtype=np.float32,
        )
        stored = aa_density > 0.0
        return TrackRasterContribution(
            hard_linear=np.asarray(hard_linear, dtype=np.int64),
            hard_local_velocity=hard_local,
            hard_track_mean_velocity=hard_track,
            aa_linear=np.asarray(aa_linear[stored], dtype=np.int64),
            aa_density_weight=aa_density[stored],
        )

    def _apply(self, contribution: TrackRasterContribution, sign: float) -> None:
        if len(contribution.hard_linear):
            np.add.at(self.density.ravel(), contribution.hard_linear, sign)
            np.add.at(
                self.local_velocity_sum.ravel(),
                contribution.hard_linear,
                sign * contribution.hard_local_velocity,
            )
            np.add.at(
                self.track_velocity_sum.ravel(),
                contribution.hard_linear,
                sign * contribution.hard_track_mean_velocity,
            )
        if len(contribution.aa_linear):
            np.add.at(
                self.density_aa.ravel(),
                contribution.aa_linear,
                sign * contribution.aa_density_weight,
            )

    def _clean_touched(self, contributions: Iterable[TrackRasterContribution]) -> None:
        hard_parts = [item.hard_linear for item in contributions if len(item.hard_linear)]
        aa_parts = [item.aa_linear for item in contributions if len(item.aa_linear)]
        hard = (
            np.unique(np.concatenate(hard_parts))
            if hard_parts
            else np.empty(0, dtype=int)
        )
        aa = np.unique(np.concatenate(aa_parts)) if aa_parts else np.empty(0, dtype=int)
        for array in (self.density, self.local_velocity_sum, self.track_velocity_sum):
            values = array.ravel()[hard]
            values[np.abs(values) < 1e-5] = 0.0
            array.ravel()[hard] = values
        values = self.density_aa.ravel()[aa]
        values[np.abs(values) < 1e-5] = 0.0
        self.density_aa.ravel()[aa] = values


__all__ = [
    "DatasetAccumulationSettings",
    "DatasetAccumulator",
    "TrackRasterContribution",
    "gaussian_anti_alias_weights",
    "pala_moving_average",
    "subdivide_track_segments",
]
