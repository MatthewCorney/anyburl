"""Rank candidates by their sorted rule contributions, compared position by position."""

import numpy as np
import torch
from torch import Tensor

from ._lexicographic_kernel import fill_top_contributions

__all__ = ["LEXICOGRAPHIC_DEPTH", "lexicographic_scores"]

LEXICOGRAPHIC_DEPTH: int = 10
"""Number of strongest contributions compared per candidate."""

OFFSET_SHARE: float = 0.5
"""Share of the gap to the next stronger value a tie-break offset may use."""

TOP_GAP: float = float(torch.finfo(torch.float32).eps)
"""Gap assumed above the strongest value, which has no next value."""


def lexicographic_scores(
    candidates: Tensor,
    values: Tensor,
    num_candidates: int,
    *,
    depth: int = LEXICOGRAPHIC_DEPTH,
) -> Tensor:
    """Score candidates so they rank by their sorted contributions.

    Each candidate's ``depth`` strongest contributions, sorted descending,
    are compared lexicographically: the strongest decides, the next breaks
    its ties, and so on. A score equals the strongest contribution plus an
    offset smaller than the gap to the next distinct strongest value.

    Parameters
    ----------
    candidates : Tensor
        Candidate id of each contribution; a candidate may repeat.
    values : Tensor
        Each contribution, in ``[0, 1]``.
    num_candidates : int
        Number of candidates scored.
    depth : int
        Number of strongest contributions compared.

    Returns
    -------
    Tensor
        ``(num_candidates,)`` float64 scores; zero where nothing contributes.
    """
    scores = np.zeros(num_candidates, dtype=np.float64)
    table = np.zeros((num_candidates, depth), dtype=values.numpy().dtype)
    filled = np.zeros(num_candidates, dtype=np.int64)
    fill_top_contributions(candidates.numpy(), values.numpy(), table, filled)
    reached = np.flatnonzero(filled)
    if reached.size == 0:
        return torch.from_numpy(scores)
    order, dense = _lexicographic_order(table[reached])
    best = table[reached[order], 0].astype(np.float64)
    scores[reached[order]] = best + _tie_offsets(best, dense)
    return torch.from_numpy(scores)


def _lexicographic_order(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ascending lexicographic order of rows and their dense ranks."""
    order = np.lexsort(rows.T[::-1])
    ordered = rows[order]
    is_new = np.empty(order.size, dtype=bool)
    is_new[0] = False
    np.any(ordered[1:] != ordered[:-1], axis=1, out=is_new[1:])
    return order, np.cumsum(is_new)


def _tie_offsets(best: np.ndarray, dense: np.ndarray) -> np.ndarray:
    """Spread each strongest-value group by its lexicographic rank.

    ``best`` and ``dense`` are in lexicographic order, so each group of
    equal strongest values is a contiguous run. A row's offset is its rank
    within that run, scaled into the gap below the next stronger value.
    """
    starts_group = np.empty(best.size, dtype=bool)
    starts_group[0] = True
    np.not_equal(best[1:], best[:-1], out=starts_group[1:])
    starts = np.flatnonzero(starts_group)
    group = np.cumsum(starts_group) - 1
    gaps = np.append(np.diff(best[starts]), TOP_GAP)
    lowest = dense[starts]
    highest = np.append(dense[starts[1:] - 1], dense[-1])
    place = (dense - lowest[group]).astype(np.float64)
    span = (highest - lowest + 1)[group].astype(np.float64)
    offsets: np.ndarray = place / span * OFFSET_SHARE * gaps[group]
    return offsets
