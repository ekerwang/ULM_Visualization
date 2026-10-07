"""PALA-compatible localization and tracking data contracts.

This project intentionally keeps the existing PALA-style result structures as
the source of truth:

* localization rows are ``[intensity, z, x, frame]``.
* ``z`` and ``x`` are 1-based pixel coordinates.
* the row ``frame`` column is 1-based.
* ``localized_by_frame`` uses 0-based Python dict keys.
* each track is a length-``n_frames`` vector of frame-local localization
  indices, with ``NaN`` where the track has no detection.

Correction code may wrap these arrays with session metadata, status flags, or
an edit log, but it should not replace the underlying representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping, MutableMapping, Sequence

import numpy as np


LOCALIZATION_COLUMNS = ("intensity", "z", "x", "frame")
TRACK_LINE_COLUMNS = ("track_id", "frame", "z", "x", "local_idx")
TRACK_POINT_COLUMNS = ("track_id", "frame", "z", "x", "local_idx")
TRACK_PATH_COLUMNS = ("track_id", "frame", "z", "x")

MatTracking = np.ndarray
LocalizedByFrame = dict[int, MatTracking]
TrackVector = np.ndarray
TrackList = list[TrackVector]


@dataclass(frozen=True)
class PalaIndexing:
    """Indexing and coordinate conventions inherited from PALA outputs."""

    frame_key_base: int = 0
    row_frame_base: int = 1
    coordinate_base: int = 1
    units: str = "pixel"
    axis_order: tuple[str, str] = ("z", "x")


@dataclass(frozen=True)
class PalaPointRef:
    """Stable reference to one localization row inside PALA-style data."""

    frame_idx: int
    local_idx: int


def pala_to_scene_xy(z: float, x: float) -> tuple[float, float]:
    """Map 1-based pixel-center PALA ``(z, x)`` to Qt scene ``(x, y)``.

    PALA ``z = x = 1.0`` is the center of the first pixel, which sits at
    scene ``(0.5, 0.5)`` when the pixmap is drawn at the origin.
    """

    return float(x) - 0.5, float(z) - 0.5


def scene_to_pala_zx(scene_x: float, scene_y: float) -> tuple[float, float]:
    """Inverse of :func:`pala_to_scene_xy` for click-to-coordinate mapping."""

    return float(scene_y) + 0.5, float(scene_x) + 0.5


def ensure_mat_tracking(rows: np.ndarray | Sequence[Sequence[float]]) -> MatTracking:
    """Return a float ``(N, 4)`` localization table or raise ``ValueError``."""

    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        return np.empty((0, 4), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 4:
            raise ValueError("A single localization row must have 4 columns.")
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            "PALA localization data must be an array shaped (N, 4): "
            "[intensity, z, x, frame]."
        )
    if not np.all(np.isfinite(arr)):
        row_idx, column_idx = np.argwhere(~np.isfinite(arr))[0]
        raise ValueError(
            "PALA localization data must contain only finite values; "
            f"row {int(row_idx)}, column {int(column_idx)} is {arr[row_idx, column_idx]!r}."
        )
    return arr


def normalize_localized_by_frame(
    localized_by_frame: Mapping[int, np.ndarray],
) -> LocalizedByFrame:
    """Validate and copy a ``frame_idx -> MatTracking`` mapping."""

    normalized: LocalizedByFrame = {}
    for key, rows in localized_by_frame.items():
        frame_idx = int(key)
        if frame_idx < 0 or key != frame_idx:
            raise ValueError(
                f"Localization frame keys must be non-negative integers; got {key!r}."
            )
        arr = ensure_mat_tracking(rows)
        if arr.size:
            expected_frame = frame_idx + 1
            row_frames = arr[:, 3]
            if not np.all(row_frames == expected_frame):
                raise ValueError(
                    f"Frame key {frame_idx} contains row frame values "
                    f"{sorted(set(row_frames.tolist()))}; expected "
                    f"{expected_frame}."
                )
        normalized[frame_idx] = arr.copy()
    return normalized


def split_mat_tracking_by_frame(mat_tracking: np.ndarray) -> LocalizedByFrame:
    """Split a stack-level PALA ``MatTracking`` table into 0-based frame keys."""

    arr = ensure_mat_tracking(mat_tracking)
    localized: LocalizedByFrame = {}
    if arr.size == 0:
        return localized

    for frame_1based in np.unique(arr[:, 3]).astype(int):
        frame_idx = int(frame_1based) - 1
        localized[frame_idx] = arr[arr[:, 3].astype(int) == frame_1based].copy()
    return localized


def localized_by_frame_to_mat_tracking(
    localized_by_frame: Mapping[int, np.ndarray],
) -> MatTracking:
    """Flatten ``localized_by_frame`` back to a stack-level PALA table."""

    rows: list[np.ndarray] = []
    for frame_idx in sorted(localized_by_frame):
        arr = ensure_mat_tracking(localized_by_frame[frame_idx])
        if arr.size:
            rows.append(arr)
    if not rows:
        return np.empty((0, 4), dtype=float)
    return np.vstack(rows)


def localized_by_frame_to_export_array(
    localized_by_frame: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Current-GUI export shape: ``[frame_idx, intensity, z, x, frame]``."""

    rows: list[np.ndarray] = []
    for frame_idx in sorted(localized_by_frame):
        arr = ensure_mat_tracking(localized_by_frame[frame_idx])
        if arr.size == 0:
            continue
        frame_col = np.full((arr.shape[0], 1), frame_idx, dtype=float)
        rows.append(np.hstack([frame_col, arr]))
    if not rows:
        return np.empty((0, 5), dtype=float)
    return np.vstack(rows)


def export_array_to_localized_by_frame(export_rows: np.ndarray) -> LocalizedByFrame:
    """Parse current-GUI ``localized_points`` export arrays.

    The current GUI saves localization as ``[frame_idx, intensity, z, x, frame]``.
    PALA itself uses only the last four columns.
    """

    arr = np.asarray(export_rows, dtype=float)
    if arr.size == 0:
        return {}
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 4:
        return split_mat_tracking_by_frame(arr)
    if arr.shape[1] != 5:
        raise ValueError(
            "Localization export arrays must have 4 PALA columns or 5 current "
            "GUI columns."
        )

    localized: LocalizedByFrame = {}
    for frame_idx in np.unique(arr[:, 0]).astype(int):
        frame_rows = arr[arr[:, 0].astype(int) == frame_idx, 1:5]
        localized[int(frame_idx)] = ensure_mat_tracking(frame_rows)
    return localized


def ensure_track_vector(track: np.ndarray | Sequence[float], n_frames: int) -> TrackVector:
    """Return one PALA/simpletracker track vector."""

    arr = np.asarray(track, dtype=float).reshape(-1)
    if arr.shape[0] != int(n_frames):
        raise ValueError(
            f"Track vector length {arr.shape[0]} does not match n_frames "
            f"{n_frames}."
        )
    assigned = arr[~np.isnan(arr)]
    valid = (
        np.isfinite(assigned)
        & (assigned >= 0)
        & (assigned == np.floor(assigned))
    )
    if not np.all(valid):
        bad = assigned[np.flatnonzero(~valid)[0]]
        raise ValueError(
            "Track assignments must be NaN gaps or finite non-negative integer "
            f"frame-local indices; got {bad!r}."
        )
    return arr


def tracks_to_matrix(tracks: Sequence[np.ndarray], n_frames: int | None = None) -> np.ndarray:
    """Convert a list of track vectors to a ``(n_tracks, n_frames)`` matrix."""

    if not tracks:
        cols = 0 if n_frames is None else int(n_frames)
        return np.empty((0, cols), dtype=float)
    if n_frames is None:
        n_frames = len(np.asarray(tracks[0]).reshape(-1))
    return np.vstack([ensure_track_vector(track, n_frames) for track in tracks])


def tracks_to_track_lines_array(
    tracks: Sequence[np.ndarray],
    localized_by_frame: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Flatten tracks to canonical ``[track_id, frame, z, x, local_idx]`` rows."""

    if not tracks:
        return np.empty((0, len(TRACK_LINE_COLUMNS)), dtype=float)

    n_frames = len(np.asarray(tracks[0]).reshape(-1))
    localization_cache: dict[int, np.ndarray] = {}
    rows: list[list[float]] = []
    for track_id, raw_track in enumerate(tracks):
        track = ensure_track_vector(raw_track, n_frames)
        for frame_idx in np.flatnonzero(~np.isnan(track)):
            local_idx = int(track[frame_idx])
            locs = localization_cache.get(int(frame_idx))
            if locs is None:
                locs = ensure_mat_tracking(localized_by_frame.get(int(frame_idx), []))
                localization_cache[int(frame_idx)] = locs
            if local_idx >= locs.shape[0]:
                raise ValueError(
                    f"Track {track_id} references invalid local_idx={local_idx} "
                    f"at frame {int(frame_idx)}."
                )
            _, z, x, _ = locs[local_idx]
            rows.append(
                [
                    float(track_id),
                    float(frame_idx),
                    float(z),
                    float(x),
                    float(local_idx),
                ]
            )

    if not rows:
        return np.empty((0, len(TRACK_LINE_COLUMNS)), dtype=float)
    return np.asarray(rows, dtype=float)


def matrix_to_tracks(
    track_matrix: np.ndarray,
    n_frames: int | None = None,
) -> TrackList:
    """Convert a ``(n_tracks, n_frames)`` matrix to a list of vectors.

    ``scipy.io.loadmat(squeeze_me=True)`` collapses 1×1 matrices to 0-d and
    single-row/column matrices to 1-d; ``n_frames`` disambiguates whether a
    1-d array is one track (length == n_frames) or a column of single-frame
    tracks.
    """

    arr = np.asarray(track_matrix, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        if n_frames == 1 and arr.shape[0] != 1:
            arr = arr.reshape(-1, 1)
        else:
            arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Track matrix must be 2D.")
    return [arr[i, :].copy() for i in range(arr.shape[0])]


def tracks_from_track_lines_array(
    track_lines: np.ndarray,
    localized_by_frame: Mapping[int, np.ndarray],
    n_frames: int,
    n_track_slots: int | None = None,
) -> TrackList:
    """Read current Batch ``[track_id, frame, z, x, local_idx]`` rows.

    ``local_idx`` is the authoritative detection identity. The z/x columns
    are checked against the referenced localization row for data-integrity
    diagnostics; they are never used to find or substitute a detection. Sparse
    track IDs are retained, with missing IDs represented by all-NaN slots.
    ``n_track_slots`` preserves trailing empty IDs in Correction GUI exports;
    current Batch files omit it and infer the count from the largest ID.
    """

    arr = np.asarray(track_lines, dtype=float)
    if n_track_slots is not None:
        raw_slot_count = n_track_slots
        n_track_slots = int(raw_slot_count)
        if (
            isinstance(raw_slot_count, bool)
            or n_track_slots < 0
            or n_track_slots != raw_slot_count
        ):
            raise ValueError(
                "n_track_slots must be a non-negative integer, "
                f"got {raw_slot_count!r}."
            )
    if arr.size == 0:
        return [
            np.full(int(n_frames), np.nan, dtype=float)
            for _ in range(n_track_slots or 0)
        ]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != len(TRACK_LINE_COLUMNS):
        raise ValueError(
            "Batch track_lines must be shaped (N, 5) with columns "
            f"{TRACK_LINE_COLUMNS}. Legacy 4-column track_lines are unsupported; "
            "rerun the data with the current Batch Processing."
        )

    n_frames_int = int(n_frames)
    if n_frames_int < 0 or n_frames_int != n_frames:
        raise ValueError(f"n_frames must be a non-negative integer, got {n_frames!r}.")

    index_columns = arr[:, [0, 1, 4]]
    valid_indices = (
        np.isfinite(index_columns)
        & (index_columns >= 0)
        & (index_columns < 2**63)
        & (index_columns == np.floor(index_columns))
    )
    if not np.all(valid_indices):
        row_idx, column_idx = np.argwhere(~valid_indices)[0]
        column_name = ("track_id", "frame_idx", "local_idx")[int(column_idx)]
        value = index_columns[row_idx, column_idx]
        raise ValueError(
            f"track_lines row {int(row_idx)} has invalid {column_name}={value!r}; "
            "track_id, frame_idx, and local_idx must be finite non-negative integers."
        )
    indices = index_columns.astype(np.int64)

    track_ids = np.unique(indices[:, 0])
    inferred_track_slots = int(track_ids[-1]) + 1
    if n_track_slots is not None and inferred_track_slots > n_track_slots:
        raise ValueError(
            "track_lines references track_id="
            f"{inferred_track_slots - 1}, outside n_track_slots={n_track_slots}."
        )
    track_slot_count = (
        inferred_track_slots if n_track_slots is None else n_track_slots
    )
    tracks = [
        np.full(n_frames_int, np.nan, dtype=float)
        for _ in range(track_slot_count)
    ]
    assigned_track_frames: set[tuple[int, int]] = set()
    localization_cache: dict[int, np.ndarray] = {}

    for row_idx, row in enumerate(arr):
        track_id, frame_idx, local_idx = (int(value) for value in indices[row_idx])
        if frame_idx >= n_frames_int:
            raise ValueError(
                f"track_lines row {row_idx} has frame_idx={frame_idx} outside "
                f"0 <= frame_idx < n_frames ({n_frames_int}) for track_id={track_id}."
            )
        if frame_idx not in localized_by_frame:
            raise ValueError(
                f"track_lines row {row_idx} references missing localization frame "
                f"frame_idx={frame_idx} for track_id={track_id}, local_idx={local_idx}."
            )

        locs = localization_cache.get(frame_idx)
        if locs is None:
            locs = ensure_mat_tracking(localized_by_frame[frame_idx])
            localization_cache[frame_idx] = locs
        if local_idx >= locs.shape[0]:
            raise ValueError(
                f"track_lines row {row_idx} has local_idx={local_idx} outside frame "
                f"{frame_idx}'s {locs.shape[0]} localization row(s) for track_id={track_id}."
            )

        assignment_key = (track_id, frame_idx)
        if assignment_key in assigned_track_frames:
            raise ValueError(
                f"track_lines contains more than one assignment for track_id={track_id}, "
                f"frame_idx={frame_idx}."
            )
        assigned_track_frames.add(assignment_key)

        z, x = float(row[2]), float(row[3])
        localization_z, localization_x = locs[local_idx, 1:3]
        if not (
            np.isclose(z, localization_z, atol=1e-9, rtol=0.0)
            and np.isclose(x, localization_x, atol=1e-9, rtol=0.0)
        ):
            raise ValueError(
                "track_lines coordinate mismatch for "
                f"track_id={track_id}, frame_idx={frame_idx}, local_idx={local_idx}: "
                f"track_lines z/x=({z}, {x}), localization z/x="
                f"({localization_z}, {localization_x}). local_idx is authoritative; "
                "the loader will not substitute a coordinate-matched detection."
            )

        tracks[track_id][frame_idx] = local_idx

    return tracks


def iter_track_point_refs(track: np.ndarray) -> list[PalaPointRef]:
    """Return all non-gap point references in a track vector."""

    refs: list[PalaPointRef] = []
    for frame_idx, local_idx in enumerate(np.asarray(track, dtype=float)):
        if np.isnan(local_idx):
            continue
        refs.append(PalaPointRef(frame_idx=frame_idx, local_idx=int(local_idx)))
    return refs


def find_matching_detection(
    localized_by_frame: Mapping[int, np.ndarray],
    frame_idx: int,
    z: float,
    x: float,
    tolerance: float = 0.25,
    *,
    excluded_local_indices: Collection[int] = (),
) -> int | None:
    """Return the ``local_idx`` of an existing detection at (z, x), if any.

    Used to avoid appending duplicate localization rows when a re-localized
    candidate coincides with a detection that already exists in this frame.
    """

    rows = ensure_mat_tracking(localized_by_frame.get(frame_idx, []))
    if rows.size == 0:
        return None
    distances = np.hypot(rows[:, 1] - float(z), rows[:, 2] - float(x))
    excluded = [
        int(local_idx)
        for local_idx in excluded_local_indices
        if 0 <= int(local_idx) < rows.shape[0]
    ]
    if excluded:
        distances[excluded] = np.inf
    if not np.any(np.isfinite(distances)):
        return None
    local_idx = int(np.argmin(distances))
    return local_idx if distances[local_idx] <= tolerance else None


def get_localization_row(
    localized_by_frame: Mapping[int, np.ndarray],
    point_ref: PalaPointRef,
) -> np.ndarray:
    """Fetch one localization row by frame-local index."""

    rows = ensure_mat_tracking(localized_by_frame.get(point_ref.frame_idx, []))
    if point_ref.local_idx < 0 or point_ref.local_idx >= rows.shape[0]:
        raise IndexError(
            f"No localization local_idx={point_ref.local_idx} in frame "
            f"{point_ref.frame_idx}."
        )
    return rows[point_ref.local_idx].copy()


def set_track_assignment(
    track: np.ndarray,
    frame_idx: int,
    local_idx: int | float | None,
) -> TrackVector:
    """Return a copy of ``track`` with one frame assignment changed."""

    updated = np.asarray(track, dtype=float).copy()
    if frame_idx < 0 or frame_idx >= updated.shape[0]:
        raise IndexError(f"frame_idx {frame_idx} is outside this track.")
    updated[frame_idx] = np.nan if local_idx is None else int(local_idx)
    return updated


def append_localization_row(
    localized_by_frame: MutableMapping[int, np.ndarray],
    frame_idx: int,
    intensity: float,
    z: float,
    x: float,
) -> int:
    """Append a PALA row and return its new frame-local index."""

    row = np.array([[float(intensity), float(z), float(x), frame_idx + 1]], dtype=float)
    existing = ensure_mat_tracking(localized_by_frame.get(frame_idx, []))
    localized_by_frame[frame_idx] = np.vstack([existing, row])
    return int(existing.shape[0])
