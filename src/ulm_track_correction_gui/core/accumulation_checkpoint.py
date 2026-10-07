"""Read-only data model for Batch PALA accumulation checkpoints.

The checkpoint is visualization-only auxiliary state.  It never replaces the
PALA localization tables or track vectors owned by :class:`CorrectionSession`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CURRENT_CHECKPOINT_SCHEMA_VERSION = 2
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = frozenset({2})
# Backwards-compatible alias for callers that used the old singular name.
CHECKPOINT_SCHEMA_VERSION = CURRENT_CHECKPOINT_SCHEMA_VERSION
CONTRIBUTION_MODES = ("none", "density_only", "full")

HARD_DENSITY = "hard_density"
AA_DENSITY = "aa_density"
HARD_LOCAL_VELOCITY = "hard_local_velocity"
AA_LOCAL_VELOCITY = "aa_local_velocity"
HARD_TRACK_MEAN_VELOCITY = "hard_track_mean_velocity"
AA_TRACK_MEAN_VELOCITY = "aa_track_mean_velocity"

TRACK_MAP_LABELS = {
    HARD_DENSITY: "Hard Density",
    AA_DENSITY: "AA Density",
    HARD_LOCAL_VELOCITY: "Hard Local Velocity",
    AA_LOCAL_VELOCITY: "AA Local Velocity",
    HARD_TRACK_MEAN_VELOCITY: "Hard Track-Mean Velocity",
    AA_TRACK_MEAN_VELOCITY: "AA Track-Mean Velocity",
}

DENSITY_TRACK_MAPS = (HARD_DENSITY, AA_DENSITY)
VELOCITY_TRACK_MAPS = (
    HARD_LOCAL_VELOCITY,
    AA_LOCAL_VELOCITY,
    HARD_TRACK_MEAN_VELOCITY,
    AA_TRACK_MEAN_VELOCITY,
)
ALL_TRACK_MAPS = (
    HARD_DENSITY,
    AA_DENSITY,
    HARD_LOCAL_VELOCITY,
    AA_LOCAL_VELOCITY,
    HARD_TRACK_MEAN_VELOCITY,
    AA_TRACK_MEAN_VELOCITY,
)

FULL_TRACK_MAPS = (
    HARD_DENSITY,
    AA_DENSITY,
    HARD_LOCAL_VELOCITY,
    HARD_TRACK_MEAN_VELOCITY,
)

_TRACK_MAP_NUMERATOR_FIELDS = {
    HARD_LOCAL_VELOCITY: "hardLocalVelocityNumerator",
    HARD_TRACK_MEAN_VELOCITY: "hardTrackMeanVelocityNumerator",
}


class TrackMapUnavailableError(ValueError):
    """Raised when a checkpoint cannot provide a requested per-track map."""


def _readonly_vector(values: object, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).reshape(-1)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TrackContribution:
    """Sparse support and sufficient-statistic slices for one source track."""

    track_id: int
    hard_rows: np.ndarray
    hard_cols: np.ndarray
    aa_rows: np.ndarray
    aa_cols: np.ndarray
    aa_density_weight: np.ndarray
    hard_local_velocity_numerator: np.ndarray | None
    hard_track_mean_velocity_numerator: np.ndarray | None


@dataclass
class SparseTrackContributions:
    """Ragged COO arrays saved by the Batch track-plotting checkpoint."""

    mode: str
    track_ids: np.ndarray
    hard_offsets: np.ndarray
    hard_rows: np.ndarray
    hard_cols: np.ndarray
    aa_offsets: np.ndarray
    aa_rows: np.ndarray
    aa_cols: np.ndarray
    aa_density_weight: np.ndarray
    hard_local_velocity_numerator: np.ndarray | None = None
    hard_track_mean_velocity_numerator: np.ndarray | None = None
    _row_by_track_id: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        self.track_ids = _readonly_vector(self.track_ids, np.int64)
        self.hard_offsets = _readonly_vector(self.hard_offsets, np.uint64)
        self.hard_rows = _readonly_vector(self.hard_rows, np.uint32)
        self.hard_cols = _readonly_vector(self.hard_cols, np.uint32)
        self.aa_offsets = _readonly_vector(self.aa_offsets, np.uint64)
        self.aa_rows = _readonly_vector(self.aa_rows, np.uint32)
        self.aa_cols = _readonly_vector(self.aa_cols, np.uint32)
        self.aa_density_weight = _readonly_vector(self.aa_density_weight, np.float32)
        for name in (
            "hard_local_velocity_numerator",
            "hard_track_mean_velocity_numerator",
        ):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, _readonly_vector(values, np.float32))
        self._row_by_track_id = {
            int(track_id): row for row, track_id in enumerate(self.track_ids)
        }

    @property
    def track_count(self) -> int:
        return int(self.track_ids.size)

    @property
    def supported_track_maps(self) -> tuple[str, ...]:
        """Return map types backed by the contribution arrays that are present."""

        if self.mode not in {"density_only", "full"}:
            return ()
        available = set(DENSITY_TRACK_MAPS)
        if self.mode == "full":
            for map_name, attribute in (
                (HARD_LOCAL_VELOCITY, "hard_local_velocity_numerator"),
                (HARD_TRACK_MEAN_VELOCITY, "hard_track_mean_velocity_numerator"),
            ):
                if getattr(self, attribute) is not None:
                    available.add(map_name)
        return tuple(map_name for map_name in ALL_TRACK_MAPS if map_name in available)

    def validate(self, map_shape: tuple[int, int]) -> SparseTrackContributions:
        if self.mode not in {"density_only", "full"}:
            raise ValueError("stored contributions must use density_only or full mode")
        if len(self._row_by_track_id) != self.track_count:
            raise ValueError("trackIds must contain unique track IDs")

        nz, nx = (int(value) for value in map_shape)
        if nz < 1 or nx < 1:
            raise ValueError("mapShape must contain two positive dimensions")

        for label, offsets, count in (
            ("hardOffsets", self.hard_offsets, len(self.hard_rows)),
            ("aaOffsets", self.aa_offsets, len(self.aa_rows)),
        ):
            if len(offsets) != self.track_count + 1:
                raise ValueError(f"{label} must contain T + 1 entries")
            if int(offsets[0]) != 0:
                raise ValueError(f"{label} must start at zero")
            if np.any(offsets[1:] < offsets[:-1]):
                raise ValueError(f"{label} must be monotonic")
            if int(offsets[-1]) != count:
                raise ValueError(f"{label} final value must equal its entry count")

        if len(self.hard_rows) != len(self.hard_cols):
            raise ValueError("hardRows and hardCols must have equal length")
        if not (len(self.aa_rows) == len(self.aa_cols) == len(self.aa_density_weight)):
            raise ValueError("aaRows, aaCols, and aaDensityWeight must align")
        if np.any(self.hard_rows >= nz) or np.any(self.hard_cols >= nx):
            raise ValueError("Hard contribution coordinates are outside mapShape")
        if np.any(self.aa_rows >= nz) or np.any(self.aa_cols >= nx):
            raise ValueError("AA contribution coordinates are outside mapShape")
        if np.any(~np.isfinite(self.aa_density_weight)) or np.any(self.aa_density_weight <= 0):
            raise ValueError("aaDensityWeight must contain positive finite values")

        for label, offsets, rows, cols in (
            ("Hard", self.hard_offsets, self.hard_rows, self.hard_cols),
            ("AA", self.aa_offsets, self.aa_rows, self.aa_cols),
        ):
            for row in range(self.track_count):
                start, stop = int(offsets[row]), int(offsets[row + 1])
                linear = rows[start:stop].astype(np.uint64) * nx + cols[start:stop]
                if len(np.unique(linear)) != len(linear):
                    raise ValueError(f"{label} pixels must be unique within each track")

        expected_lengths = {
            "hardLocalVelocityNumerator": (
                self.hard_local_velocity_numerator,
                len(self.hard_rows),
            ),
            "hardTrackMeanVelocityNumerator": (
                self.hard_track_mean_velocity_numerator,
                len(self.hard_rows),
            ),
        }
        for label, (values, expected) in expected_lengths.items():
            if values is None:
                continue
            if len(values) != expected:
                raise ValueError(f"{label} must align with its sparse support")
            if np.any(~np.isfinite(values)):
                raise ValueError(f"{label} must contain finite values")
        return self

    def has_track(self, track_id: int) -> bool:
        return int(track_id) in self._row_by_track_id

    def track_contribution(self, track_id: int) -> TrackContribution:
        track_id = int(track_id)
        try:
            row = self._row_by_track_id[track_id]
        except KeyError as exc:
            raise KeyError(f"checkpoint does not contain source track {track_id}") from exc
        h0, h1 = int(self.hard_offsets[row]), int(self.hard_offsets[row + 1])
        a0, a1 = int(self.aa_offsets[row]), int(self.aa_offsets[row + 1])

        def hard_slice(values: np.ndarray | None) -> np.ndarray | None:
            return None if values is None else values[h0:h1]

        return TrackContribution(
            track_id=track_id,
            hard_rows=self.hard_rows[h0:h1],
            hard_cols=self.hard_cols[h0:h1],
            aa_rows=self.aa_rows[a0:a1],
            aa_cols=self.aa_cols[a0:a1],
            aa_density_weight=self.aa_density_weight[a0:a1],
            hard_local_velocity_numerator=hard_slice(self.hard_local_velocity_numerator),
            hard_track_mean_velocity_numerator=hard_slice(
                self.hard_track_mean_velocity_numerator
            ),
        )


@dataclass
class AccumulationCheckpoint:
    """Validated schema-v2 accumulation checkpoint used only for display."""

    schema_version: int
    mode: str
    map_shape: tuple[int, int]
    input_shape: tuple[int, int]
    ulm_scale_z: float
    ulm_scale_x: float
    params: dict
    metadata: dict
    contributions: SparseTrackContributions | None
    source_path: Path

    def __post_init__(self) -> None:
        self.schema_version = int(self.schema_version)
        self.mode = str(self.mode).strip().lower()
        self.map_shape = tuple(int(value) for value in self.map_shape)
        self.input_shape = tuple(int(value) for value in self.input_shape)
        self.ulm_scale_z = float(self.ulm_scale_z)
        self.ulm_scale_x = float(self.ulm_scale_x)
        self.source_path = Path(self.source_path)
        if self.schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported checkpoint schema version {self.schema_version}; "
                f"supported version is {CURRENT_CHECKPOINT_SCHEMA_VERSION}"
            )
        if self.mode not in CONTRIBUTION_MODES:
            raise ValueError(f"unsupported perTrackContributionMode: {self.mode}")
        if len(self.map_shape) != 2 or min(self.map_shape) < 1:
            raise ValueError("mapShape must contain two positive dimensions")
        if len(self.input_shape) != 2 or min(self.input_shape) < 1:
            raise ValueError("inputShape must contain two positive dimensions")
        if not np.isfinite(self.ulm_scale_z) or self.ulm_scale_z <= 0:
            raise ValueError("ulmScaleZ must be a positive finite value")
        if not np.isfinite(self.ulm_scale_x) or self.ulm_scale_x <= 0:
            raise ValueError("ulmScaleX must be a positive finite value")
        if self.mode == "none":
            if self.contributions is not None:
                raise ValueError("none mode must not contain per-track contributions")
        else:
            if self.contributions is None:
                raise ValueError(f"{self.mode} mode requires per-track contributions")
            if self.contributions.mode != self.mode:
                raise ValueError("checkpoint mode does not match its contributions")
            self.contributions.validate(self.map_shape)
            for map_name in self._schema_track_maps():
                if map_name not in self.supported_track_maps:
                    field = _TRACK_MAP_NUMERATOR_FIELDS.get(map_name, map_name)
                    raise ValueError(
                        f"schema v{self.schema_version} {self.mode} checkpoint "
                        f"requires aligned {field}"
                    )

    def _schema_track_maps(self) -> tuple[str, ...]:
        if self.mode == "density_only":
            return DENSITY_TRACK_MAPS
        if self.mode == "full":
            return FULL_TRACK_MAPS
        return ()

    @property
    def supported_track_maps(self) -> tuple[str, ...]:
        """Map types this checkpoint can provide for any stored source track."""

        if self.contributions is None:
            return ()
        present = set(self.contributions.supported_track_maps)
        return tuple(map_name for map_name in self._schema_track_maps() if map_name in present)

    def validate_input_shape(self, input_shape: tuple[int, int]) -> None:
        actual = tuple(int(value) for value in input_shape)
        if actual != self.input_shape:
            raise ValueError(
                "checkpoint inputShape "
                f"{self.input_shape} does not match bubble movie shape {actual}"
            )

    def has_track(self, track_id: int) -> bool:
        return self.contributions is not None and self.contributions.has_track(track_id)

    def available_track_maps(self, track_id: int) -> tuple[str, ...]:
        if not self.has_track(track_id):
            return ()
        return self.supported_track_maps

    def track_contribution(self, track_id: int) -> TrackContribution:
        if self.contributions is None:
            raise TrackMapUnavailableError(
                "checkpoint does not contain per-track contributions"
            )
        return self.contributions.track_contribution(track_id)

    def track_map_values(
        self,
        track_id: int,
        map_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sparse ``rows, cols, values`` without allocating a full map."""

        map_name = str(map_name)
        if map_name not in TRACK_MAP_LABELS:
            raise ValueError(f"unknown selected-track map: {map_name}")
        available = self.available_track_maps(track_id)
        if map_name not in available:
            label = TRACK_MAP_LABELS[map_name]
            raise TrackMapUnavailableError(
                f"{label} is unavailable in schema v{self.schema_version} "
                f"{self.mode} checkpoint"
            )

        contribution = self.track_contribution(track_id)
        if map_name == HARD_DENSITY:
            values = np.ones(len(contribution.hard_rows), dtype=np.float32)
            return contribution.hard_rows, contribution.hard_cols, values
        if map_name == AA_DENSITY:
            return (
                contribution.aa_rows,
                contribution.aa_cols,
                contribution.aa_density_weight,
            )

        if map_name in {HARD_LOCAL_VELOCITY, HARD_TRACK_MEAN_VELOCITY}:
            numerator = (
                contribution.hard_local_velocity_numerator
                if map_name == HARD_LOCAL_VELOCITY
                else contribution.hard_track_mean_velocity_numerator
            )
            if numerator is None:  # Defensive guard for hand-constructed invalid models.
                raise TrackMapUnavailableError(
                    f"{TRACK_MAP_LABELS[map_name]} has no saved numerator"
                )
            return contribution.hard_rows, contribution.hard_cols, numerator

        raise TrackMapUnavailableError(
            f"{TRACK_MAP_LABELS[map_name]} has no saved schema-v2 numerator"
        )


@dataclass(frozen=True)
class RasterCropGeometry:
    row0: int
    col0: int
    height: int
    width: int
    pos_x: float
    pos_y: float
    scale_x: float
    scale_y: float


def raster_pixel_scene_center(
    row: int,
    col: int,
    ulm_scale_z: float,
    ulm_scale_x: float,
) -> tuple[float, float]:
    """Map one zero-based ULM raster pixel center to the raw-image scene."""

    scale_z, scale_x = float(ulm_scale_z), float(ulm_scale_x)
    if scale_z <= 0 or scale_x <= 0:
        raise ValueError("ULM scales must be positive")
    return 0.5 + int(col) / scale_x, 0.5 + int(row) / scale_z


def raster_crop_geometry(
    rows: np.ndarray,
    cols: np.ndarray,
    ulm_scale_z: float,
    ulm_scale_x: float,
) -> RasterCropGeometry:
    """Return the cropped pixmap geometry preserving ULM pixel-center alignment."""

    rows = np.asarray(rows).reshape(-1)
    cols = np.asarray(cols).reshape(-1)
    if len(rows) == 0 or len(rows) != len(cols):
        raise ValueError("rows and cols must be non-empty aligned vectors")
    scale_z, scale_x = float(ulm_scale_z), float(ulm_scale_x)
    if scale_z <= 0 or scale_x <= 0:
        raise ValueError("ULM scales must be positive")
    row0, row1 = int(rows.min()), int(rows.max())
    col0, col1 = int(cols.min()), int(cols.max())
    return RasterCropGeometry(
        row0=row0,
        col0=col0,
        height=row1 - row0 + 1,
        width=col1 - col0 + 1,
        pos_x=0.5 + (col0 - 0.5) / scale_x,
        pos_y=0.5 + (row0 - 0.5) / scale_z,
        scale_x=1.0 / scale_x,
        scale_y=1.0 / scale_z,
    )


def current_to_checkpoint_track_ids(
    metadata: dict,
    n_tracks: int,
) -> dict[int, int] | None:
    """Reverse saved ``source old ID -> corrected ID`` metadata when present.

    ``None`` means identity mapping.  A dictionary is deliberately strict:
    missing current IDs must not silently fall back to an unrelated source ID.
    """

    raw = dict(metadata or {}).get("track_id_remap")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("metadata track_id_remap must be an object")
    reverse: dict[int, int] = {}
    for source_id, current_id in raw.items():
        try:
            source_int = int(source_id)
            current_int = int(current_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata track_id_remap must contain integer IDs") from exc
        if not (0 <= current_int < int(n_tracks)):
            raise ValueError("metadata track_id_remap contains an invalid corrected ID")
        if current_int in reverse:
            raise ValueError("metadata track_id_remap is not one-to-one")
        reverse[current_int] = source_int
    return reverse


__all__ = [
    "AA_DENSITY",
    "AA_LOCAL_VELOCITY",
    "AA_TRACK_MEAN_VELOCITY",
    "ALL_TRACK_MAPS",
    "AccumulationCheckpoint",
    "CHECKPOINT_SCHEMA_VERSION",
    "CURRENT_CHECKPOINT_SCHEMA_VERSION",
    "CONTRIBUTION_MODES",
    "DENSITY_TRACK_MAPS",
    "FULL_TRACK_MAPS",
    "HARD_DENSITY",
    "HARD_LOCAL_VELOCITY",
    "HARD_TRACK_MEAN_VELOCITY",
    "RasterCropGeometry",
    "SparseTrackContributions",
    "SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS",
    "TRACK_MAP_LABELS",
    "TrackContribution",
    "TrackMapUnavailableError",
    "VELOCITY_TRACK_MAPS",
    "current_to_checkpoint_track_ids",
    "raster_crop_geometry",
    "raster_pixel_scene_center",
]
