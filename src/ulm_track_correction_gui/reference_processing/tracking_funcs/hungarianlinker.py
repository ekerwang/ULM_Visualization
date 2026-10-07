import numpy as np
from scipy.spatial.distance import cdist
from .munkres import munkres

def hungarianlinker(source, target, max_distance=np.inf):
    """
    Link source points to target points using Hungarian assignment.
    Python version of hungarianlinker.m by Jean-Yves Tinevez <jeanyves.tinevez@gmail.com>.
        "However all credits should go to Yi Cao, which did the hard job of
        implementing the Munkres algorithm; this file is merely a wrapper for it."

    Returns
    -------
    target_indices : ndarray
        0-based target index for each source point.
        -1 means unmatched.

    target_distances : ndarray
        Euclidean distance for each match, NaN if unmatched.

    unassigned_targets : ndarray
        0-based indices of unused target points.

    total_cost : float
        Sum of squared distances for accepted matches.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    n_source = source.shape[0]
    n_target = target.shape[0]

    target_indices = np.full(n_source, -1, dtype=int)
    target_distances = np.full(n_source, np.nan)

    if n_source == 0:
        return target_indices, target_distances, np.arange(n_target), 0.0

    if n_target == 0:
        return target_indices, target_distances, np.array([], dtype=int), 0.0

    D = cdist(source, target, metric="sqeuclidean")
    D[D > max_distance ** 2] = np.inf

    target_indices, total_cost = munkres(D)

    for i, j in enumerate(target_indices):
        if j >= 0:
            target_distances[i] = np.sqrt(D[i, j])

    unassigned_targets = np.setdiff1d(
        np.arange(n_target),
        target_indices[target_indices >= 0],
    )

    return target_indices, target_distances, unassigned_targets, total_cost

