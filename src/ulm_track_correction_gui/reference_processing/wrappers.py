"""Reference localization/tracking wrappers migrated from the current GUI."""

from __future__ import annotations

import numpy as np

from ulm_track_correction_gui.core.legacy_adapters import (
    infer_n_frames,
    track_paths_from_tracks,
    track_points_from_tracks,
)
from ulm_track_correction_gui.core.pala_contracts import LocalizedByFrame

from .localization_funcs.ULM_localization2D_NCCbased import ULM_localization2D_NCCbased
from .localization_funcs.ULM_localization2D_NonNCC import ULM_localization2D_NonNCC
from .tracking_funcs.simpletracker import simpletracker


RADIAL_NON_NCC_METHOD = "radial"
FS_GRADIENT_NCC_METHOD = "FS_Gradient"
RADIAL_NCC_METHOD = "radial_ncc"

FS_GRADIENT_NCC_METHODS = {
    "fs_gradient",
    "fsgradient",
    "fs gradient",
    "fs_gradient_ncc",
}
RADIAL_NCC_METHODS = {
    "radial_ncc",
    "radial symmetry (ncc)",
}
NCC_BASED_METHODS = FS_GRADIENT_NCC_METHODS | RADIAL_NCC_METHODS


def localization_wrapper(dataset: np.ndarray, local_params: dict) -> LocalizedByFrame | np.ndarray:
    """Run the migrated localization implementation.

    Stack input returns ``dict[frame_idx_0based] -> MatTracking``. Single-frame
    input returns one PALA ``MatTracking`` table, matching the old GUI behavior.
    """

    method = str(local_params.get("method", "")).strip().lower()
    fs_filter_size = local_params.get("fsFilterSize", local_params.get("filterSize"))

    if method in NCC_BASED_METHODS:
        result = ULM_localization2D_NCCbased(
            dataset,
            PSFSize=[
                local_params.get("psfSizeZAxis", 4.0),
                local_params.get("psfSizeXAxis", 3.5),
            ],
            PSFWindowSize=local_params.get("psfWindowSize"),
            AmpThreshold=local_params.get("ampThreshold", 0.05),
            CorrThreshold=local_params.get("corrThreshold", 0.54),
            FSFilterSize=fs_filter_size if fs_filter_size is not None else 7,
            LocMethod="radial" if method in RADIAL_NCC_METHODS else "FS_Gradient",
        )
    else:
        localization_kwargs = {
            "MatIn": dataset,
            "fwhm": [
                local_params.get("fwhmXAxis"),
                local_params.get("fwhmZAxis"),
            ],
            "numberOfParticles": local_params.get("numMB"),
            "LocMethod": local_params.get("method"),
            "NLocalMax": local_params.get("numLocalMax"),
        }
        if fs_filter_size is not None:
            localization_kwargs["FSFilterSize"] = fs_filter_size
        result = ULM_localization2D_NonNCC(**localization_kwargs)

    if len(dataset.shape) == 3:
        localized_by_frame: LocalizedByFrame = {}
        for frame_num in np.unique(result[:, 3]).astype(int):
            localized_by_frame[frame_num - 1] = result[result[:, 3] == frame_num]
        return localized_by_frame

    return result


def tracking_wrapper(localized_points: LocalizedByFrame, track_params: dict) -> dict:
    """Run migrated simpletracker and return legacy-compatible structures."""

    n_frames = infer_n_frames(localized_by_frame=localized_points)
    points: list[np.ndarray] = []
    for frame_idx in range(n_frames):
        locs = localized_points.get(frame_idx, np.empty((0, 4)))
        if locs is None or len(locs) == 0:
            points.append(np.empty((0, 2)))
        else:
            points.append(np.asarray(locs)[:, [1, 2]])

    track_method = str(track_params.get("method", "Hungarian")).strip()
    if track_method.lower() in {"simple tracker", "simpletracker", ""}:
        track_method = "Hungarian"

    tracks, adjacency_tracks, adjacency_matrix, coordinate_tracks = simpletracker(
        points,
        method=track_method,
        max_linking_distance=track_params.get("maxLink", np.inf),
        max_gap_closing=track_params.get("maxGap", 0),
        debug=bool(track_params.get("debug", False)),
    )

    return {
        "track_paths": track_paths_from_tracks(tracks, localized_points),
        "track_points": track_points_from_tracks(tracks, localized_points),
        "tracks": tracks,
        "adjacency_tracks": adjacency_tracks,
        "adjacency_matrix": adjacency_matrix,
        "coordinate_tracks": coordinate_tracks,
    }
