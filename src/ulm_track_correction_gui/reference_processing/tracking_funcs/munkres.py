import numpy as np
from scipy.optimize import linear_sum_assignment


def munkres(cost_mat):
    """
    Python replacement for MATLAB munkres.m
    "Reference:
        "Munkres' Assignment Algorithm, Modified for Rectangular Matrices", 
        http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html

        version 2.3 by Yi Cao at Cranfield University on 11th September 2011"

    Returns
    -------
    assignment : ndarray
        assignment[i] is the assigned column for row i.
        Uses 0-based Python indexing.
        Unassigned rows are -1.

    cost : float
        Total assignment cost.
    """
    cost_mat = np.asarray(cost_mat, dtype=float)

    n_rows, n_cols = cost_mat.shape
    assignment = np.full(n_rows, -1, dtype=int)

    valid = np.isfinite(cost_mat)

    if not np.any(valid):
        return assignment, 0.0

    big_m = np.nanmax(cost_mat[valid]) * 1e6 + 1
    safe_cost = cost_mat.copy()
    safe_cost[~valid] = big_m

    row_ind, col_ind = linear_sum_assignment(safe_cost)

    total_cost = 0.0

    for r, c in zip(row_ind, col_ind):
        if valid[r, c]:
            assignment[r] = c
            total_cost += cost_mat[r, c]

    return assignment, total_cost


