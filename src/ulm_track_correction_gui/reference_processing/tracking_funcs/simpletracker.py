import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from .nearestneighborlinker import nearestneighborlinker
from .hungarianlinker import hungarianlinker


def simpletracker(
    points,
    method="Hungarian",
    max_linking_distance=np.inf,
    max_gap_closing=3,
    debug=False,
):
    """
    Python version of simpletracker.m by Jean-Yves Tinevez 
            < jeanyves.tinevez@gmail.com> November 2011 - 2012

    Parameters
    ----------
    points : list of ndarray
        One array per frame. Each array is shape (n_points, n_dimensions).

    Returns
    -------
    tracks : list of ndarray
        Per-track frame indices. Uses 0-based indexing.
        NaN means no detection in that frame.

    adjacency_tracks : list of ndarray
        Tracks using global concatenated point indices, 0-based.

    A : scipy.sparse.csr_matrix
        Sparse adjacency matrix.


        will need to just restructure my array that's (z, x, t) into (t, (z, x))
    """

    n_slices = len(points)
    n_cells = np.array([len(p) for p in points], dtype=int)
    n_total_cells = int(np.sum(n_cells))

    row_indices = []
    col_indices = []

    unmatched_targets = [np.array([], dtype=int) for _ in range(n_slices)]
    unmatched_sources = [np.array([], dtype=int) for _ in range(n_slices)]

    current_slice_index = 0

    # ------------------------------------------------------------
    # Frame-to-frame linking
    # ------------------------------------------------------------
    for i in range(n_slices - 1):

        source = np.asarray(points[i], dtype=float)
        target = np.asarray(points[i + 1], dtype=float)

        if method.lower() == "hungarian":
            target_indices, _, unmatched_targets[i + 1], _ = hungarianlinker(
                source,
                target,
                max_linking_distance,
            )
        elif method.lower() in ["nearestneighbor", "nearest_neighbor"]:
            target_indices, _, unmatched_targets[i + 1] = nearestneighborlinker(
                source,
                target,
                max_linking_distance,
            )
        else:
            raise ValueError("method must be 'Hungarian' or 'NearestNeighbor'")

        unmatched_sources[i] = np.where(target_indices == -1)[0]

        for source_idx, target_idx in enumerate(target_indices):
            if target_idx == -1:
                continue

            row_indices.append(current_slice_index + source_idx)
            col_indices.append(current_slice_index + n_cells[i] + target_idx)

        current_slice_index += n_cells[i]

    A = lil_matrix((n_total_cells, n_total_cells), dtype=np.uint8)

    for r, c in zip(row_indices, col_indices):
        A[r, c] = 1

    # ------------------------------------------------------------
    # Gap closing
    # ------------------------------------------------------------
    current_slice_index = 0

    for i in range(n_slices - 2):

        current_target_slice_index = (
            current_slice_index + n_cells[i] + n_cells[i + 1]
        )

        final_frame = min(i + max_gap_closing, n_slices - 1)

        for j in range(i + 2, final_frame + 1):

            source_indices = unmatched_sources[i]
            target_indices_available = unmatched_targets[j]

            source = np.asarray(points[i], dtype=float)[source_indices]
            target = np.asarray(points[j], dtype=float)[target_indices_available]

            if len(source) == 0 or len(target) == 0:
                current_target_slice_index += n_cells[j]
                continue

            linked_targets, _, _ = nearestneighborlinker(
                source,
                target,
                max_linking_distance,
            )

            for k, local_target_idx in enumerate(linked_targets):

                if local_target_idx == -1:
                    continue

                row = current_slice_index + source_indices[k]
                col = (
                    current_target_slice_index
                    + target_indices_available[local_target_idx]
                )

                A[row, col] = 1

                if debug:
                    print(
                        f"Creating link: frame {i}, point {source_indices[k]} "
                        f"-> frame {j}, point "
                        f"{target_indices_available[local_target_idx]}"
                    )

            new_links = linked_targets != -1

            unmatched_sources[i] = source_indices[~new_links]

            used_target_local_indices = linked_targets[new_links]
            unmatched_targets[j] = np.delete(
                target_indices_available,
                used_target_local_indices,
            )

            current_target_slice_index += n_cells[j]

        current_slice_index += n_cells[i]

    A = A.tocsr()

    # ------------------------------------------------------------
    # Build adjacency tracks
    # ------------------------------------------------------------
    incoming_counts = np.asarray(A.sum(axis=0)).ravel()
    cells_without_source = np.where(incoming_counts == 0)[0]

    adjacency_tracks = []

    for start_cell in cells_without_source:

        track = []
        current = start_cell

        while True:
            track.append(current)

            next_cells = A[current, :].nonzero()[1]

            if len(next_cells) == 0:
                break

            current = next_cells[0]

        adjacency_tracks.append(np.array(track, dtype=int))

    # ------------------------------------------------------------
    # Convert global adjacency indices back to frame-local indices
    # ------------------------------------------------------------
    cumulative_cells = np.concatenate([[0], np.cumsum(n_cells)])

    tracks = []

    for adjacency_track in adjacency_tracks:

        track = np.full(n_slices, np.nan)

        for global_idx in adjacency_track:

            frame_idx = np.searchsorted(
                cumulative_cells,
                global_idx,
                side="right",
            ) - 1

            local_idx = global_idx - cumulative_cells[frame_idx]

            track[frame_idx] = local_idx

        tracks.append(track)

    # ------------------------------------------------------------
    # Collect coordinates for each tracked microbubble
    # ------------------------------------------------------------

    all_points = np.vstack(points)

    track_points = []

    for adjacency_track in adjacency_tracks:
        coords = all_points[adjacency_track, :]
        track_points.append(coords)

    return tracks, adjacency_tracks, A, track_points
