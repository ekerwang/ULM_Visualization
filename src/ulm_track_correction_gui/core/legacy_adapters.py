"""Derived adapters for current-GUI compatibility views."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from .pala_contracts import (
    LocalizedByFrame,
    ensure_mat_tracking,
    pala_to_scene_xy,
)


def infer_n_frames(
    localized_by_frame: Mapping[int, np.ndarray] | None = None,
    tracks: Sequence[np.ndarray] | None = None,
    stack: np.ndarray | None = None,
) -> int:
    """Infer frame count from stack, tracks, or localization keys."""

    if stack is not None and getattr(stack, "ndim", 0) == 3:
        return int(stack.shape[2])
    if tracks:
        return int(np.asarray(tracks[0]).reshape(-1).shape[0])
    if localized_by_frame:
        return max(int(k) for k in localized_by_frame.keys()) + 1
    return 0


def track_points_from_tracks(
    tracks: Sequence[np.ndarray],
    localized_by_frame: LocalizedByFrame,
) -> list[dict]:
    """Build current-GUI ``track_points`` dictionaries from track vectors."""

    points: list[dict] = []
    for track_id, track in enumerate(tracks):
        for frame_idx, local_idx in enumerate(np.asarray(track, dtype=float)):
            if np.isnan(local_idx):
                continue
            local_idx_int = int(local_idx)
            locs = ensure_mat_tracking(localized_by_frame.get(frame_idx, []))
            if local_idx_int < 0 or local_idx_int >= locs.shape[0]:
                raise ValueError(
                    f"Track {track_id} references invalid local_idx={local_idx_int} "
                    f"at frame {frame_idx}."
                )
            _, z, x, _ = locs[local_idx_int]
            points.append(
                {
                    "track_id": int(track_id),
                    "frame": int(frame_idx),
                    "z": float(z),
                    "x": float(x),
                    "local_idx": int(local_idx_int),
                }
            )
    return points


def track_paths_from_tracks(
    tracks: Sequence[np.ndarray],
    localized_by_frame: LocalizedByFrame,
) -> list[dict]:
    """Build current-GUI ``track_paths`` dictionaries from track vectors."""

    paths: list[dict] = []
    for track_id, track in enumerate(tracks):
        coords: list[tuple[int, float, float]] = []
        for frame_idx, local_idx in enumerate(np.asarray(track, dtype=float)):
            if np.isnan(local_idx):
                continue
            local_idx_int = int(local_idx)
            locs = ensure_mat_tracking(localized_by_frame.get(frame_idx, []))
            if local_idx_int < 0 or local_idx_int >= locs.shape[0]:
                raise ValueError(
                    f"Track {track_id} references invalid local_idx={local_idx_int} "
                    f"at frame {frame_idx}."
                )
            _, z, x, _ = locs[local_idx_int]
            coords.append((int(frame_idx), float(z), float(x)))
        if coords:
            paths.append({"track_id": int(track_id), "coords": coords})
    return paths


def track_points_to_array(track_points: Sequence[dict]) -> np.ndarray:
    """Flatten current-GUI track point dicts to an export array."""

    rows = [
        [
            p["track_id"],
            p["frame"],
            p["z"],
            p["x"],
            p["local_idx"],
        ]
        for p in track_points
    ]
    if not rows:
        return np.empty((0, 5), dtype=float)
    return np.asarray(rows, dtype=float)


def track_paths_to_array(track_paths: Sequence[dict]) -> np.ndarray:
    """Flatten current-GUI track path dicts to an export array."""

    rows: list[list[float]] = []
    for track in track_paths:
        track_id = int(track["track_id"])
        for frame_idx, z, x in track["coords"]:
            rows.append([track_id, frame_idx, z, x])
    if not rows:
        return np.empty((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def track_lines_by_frame_from_tracks(
    tracks: Sequence[np.ndarray],
    localized_by_frame: LocalizedByFrame,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Build Qt-scene line segments from track vectors.

    Coordinates are converted from PALA's 1-based pixel-center ``z/x`` to Qt
    scene coordinates via ``pala_to_scene_xy``.
    """

    lines_by_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for path in track_paths_from_tracks(tracks, localized_by_frame):
        coords = sorted(path["coords"], key=lambda p: p[0])
        if len(coords) < 2:
            continue
        segments: list[tuple[float, float, float, float]] = []
        for idx in range(len(coords) - 1):
            _, z1, x1 = coords[idx]
            _, z2, x2 = coords[idx + 1]
            sx1, sy1 = pala_to_scene_xy(z1, x1)
            sx2, sy2 = pala_to_scene_xy(z2, x2)
            segments.append((sx1, sy1, sx2, sy2))

        frames = [frame for frame, _, _ in coords]
        for frame_idx in range(min(frames), max(frames) + 1):
            lines_by_frame[frame_idx].extend(segments)
    return dict(lines_by_frame)
