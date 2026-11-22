"""Numba kernel keeping each candidate's strongest contributions, sorted."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def fill_top_contributions(
    candidates: np.ndarray,
    values: np.ndarray,
    table: np.ndarray,
    filled: np.ndarray,
) -> None:
    """Insert positive contributions into per-candidate descending rows.

    Parameters
    ----------
    candidates : np.ndarray
        Candidate id of each contribution.
    values : np.ndarray
        Each contribution; values that are not positive are skipped.
    table : np.ndarray
        ``(num_candidates, depth)`` zeros, filled in place with each
        candidate's ``depth`` strongest values, descending.
    filled : np.ndarray
        ``(num_candidates,)`` zeros, set in place to each row's used length.
    """
    depth = table.shape[1]
    for index in range(candidates.size):
        value = values[index]
        if value <= 0:
            continue
        candidate = candidates[index]
        count = filled[candidate]
        if count < depth:
            slot = count
            filled[candidate] = count + 1
        elif value > table[candidate, depth - 1]:
            slot = depth - 1
        else:
            continue
        while slot > 0 and table[candidate, slot - 1] < value:
            table[candidate, slot] = table[candidate, slot - 1]
            slot -= 1
        table[candidate, slot] = value
