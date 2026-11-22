"""Rank candidates by their sorted rule contributions, compared position by position."""

import torch
from torch import Tensor

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
    scores = torch.zeros(num_candidates, dtype=torch.float64)
    is_positive = values > 0
    candidates, values = candidates[is_positive], values[is_positive]
    if candidates.numel() == 0:
        return scores
    reached, table = _top_contributions(candidates, values, depth)
    _, order = torch.unique(table, dim=0, return_inverse=True)
    best = table[:, 0].to(torch.float64)
    scores[reached] = best + _tie_offsets(best, order)
    return scores


def _top_contributions(
    candidates: Tensor, values: Tensor, depth: int
) -> tuple[Tensor, Tensor]:
    """Return the reached candidates and their strongest values, descending."""
    by_value = torch.argsort(values, descending=True, stable=True)
    candidates, values = candidates[by_value], values[by_value]
    by_candidate = torch.argsort(candidates, stable=True)
    candidates, values = candidates[by_candidate], values[by_candidate]
    reached, local = torch.unique(candidates, return_inverse=True)
    counts = torch.bincount(local, minlength=reached.numel())
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(candidates.numel()) - starts[local]
    is_kept = positions < depth
    table = torch.zeros(reached.numel(), depth, dtype=values.dtype)
    table[local[is_kept], positions[is_kept]] = values[is_kept]
    return reached, table


def _tie_offsets(best: Tensor, order: Tensor) -> Tensor:
    """Spread each strongest-value group by its lexicographic order.

    Rows sharing a strongest value occupy a contiguous run of ``order``, so
    each row's offset is its place in that run, scaled into the gap below
    the next stronger value.
    """
    distinct, group = torch.unique(best, return_inverse=True)
    top_gap = torch.tensor([TOP_GAP], dtype=distinct.dtype)
    gaps = torch.cat([distinct[1:] - distinct[:-1], top_gap])
    num_groups = distinct.numel()
    lowest = torch.full((num_groups,), order.numel(), dtype=order.dtype)
    lowest.scatter_reduce_(0, group, order, "amin")
    highest = torch.zeros(num_groups, dtype=order.dtype)
    highest.scatter_reduce_(0, group, order, "amax")
    place = (order - lowest[group]).to(torch.float64)
    span = (highest - lowest + 1)[group].to(torch.float64)
    return place / span * OFFSET_SHARE * gaps[group]
