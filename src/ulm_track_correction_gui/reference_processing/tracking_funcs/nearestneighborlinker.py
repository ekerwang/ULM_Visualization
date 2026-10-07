import numpy as np
from scipy.spatial.distance import cdist

def nearestneighborlinker(source, target, max_distance=np.inf):
    """
    Link source points to target points using greedy nearest-neighbor linking.
    Python version of nearestneighborlinker.m by Jean-Yves Tinevez 
            < jeanyves.tinevez@gmail.com> November 2011 - 2012


    Returns
    -------
    target_indices : ndarray
        0-based target index for each source point.
        -1 means unmatched.

    target_distances : ndarray
        Euclidean distance for each match, NaN if unmatched.

    unassigned_targets : ndarray
        0-based indices of unused target points.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    n_source = source.shape[0]
    n_target = target.shape[0]

    target_indices = np.full(n_source, -1, dtype=int)
    target_distances = np.full(n_source, np.nan)

    if n_source == 0:
        return target_indices, target_distances, np.arange(n_target)

    if n_target == 0:
        return target_indices, target_distances, np.array([], dtype=int)

    D = cdist(source, target, metric="sqeuclidean")
    D[D > max_distance ** 2] = np.inf

    while not np.all(np.isinf(D)):

        closest_targets = np.argmin(D, axis=1)
        min_D = D[np.arange(n_source), closest_targets]

        sorted_sources = np.argsort(min_D)

        for source_index in sorted_sources:

            if np.isinf(min_D[source_index]):
                continue

            target_index = closest_targets[source_index]

            if target_index in target_indices:
                break

            target_indices[source_index] = target_index
            target_distances[source_index] = np.sqrt(min_D[source_index])

            D[:, target_index] = np.inf
            D[source_index, :] = np.inf

            if np.all(np.isinf(D)):
                break

    unassigned_targets = np.setdiff1d(
        np.arange(n_target),
        target_indices[target_indices >= 0],
    )

    return target_indices, target_distances, unassigned_targets