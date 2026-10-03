"""How body grounding counts are weighted into a rule's contribution."""

from enum import StrEnum
from typing import assert_never

import torch
from torch import Tensor

__all__ = [
    "GROUNDING_WEIGHT_OFFSET",
    "ScoringStrategy",
    "evidence_weights",
    "weight_for_candidate",
]

GROUNDING_WEIGHT_OFFSET: float = 1.0
"""Offset in ``log2(offset + groundings)``, set so one grounding weighs 1.0."""


class ScoringStrategy(StrEnum):
    """How a rule's confidence is weighted against a candidate.

    Attributes
    ----------
    NOISY_OR : str
        Each firing rule contributes its confidence once, however many
        body groundings support the candidate.
    PATH_WEIGHTED : str
        A rule's confidence is weighted by ``log2(1 + groundings)``, so a
        candidate reached by many groundings outranks one reached by a
        single grounding. A single grounding reproduces :attr:`NOISY_OR`.
    """

    NOISY_OR = "noisy_or"
    PATH_WEIGHTED = "path_weighted"


def evidence_weights(groundings: Tensor, strategy: ScoringStrategy) -> Tensor:
    """Convert body grounding counts into per-candidate evidence weights.

    Parameters
    ----------
    groundings : Tensor
        Number of body groundings supporting each candidate.
    strategy : ScoringStrategy
        The weighting scheme.

    Returns
    -------
    Tensor
        Weights parallel to ``groundings``.
    """
    match strategy:
        case ScoringStrategy.NOISY_OR:
            return torch.ones_like(groundings)
        case ScoringStrategy.PATH_WEIGHTED:
            return torch.log2(GROUNDING_WEIGHT_OFFSET + groundings)
        case _ as unreachable:
            assert_never(unreachable)


def weight_for_candidate(
    row: tuple[Tensor, Tensor],
    candidate_id: int,
    strategy: ScoringStrategy,
) -> tuple[int, float] | None:
    """Return one candidate's grounding count and evidence weight, if reached.

    Parameters
    ----------
    row : tuple[Tensor, Tensor]
        Candidate indices and grounding counts from a chain grounding.
    candidate_id : int
        The candidate to look for.
    strategy : ScoringStrategy
        The weighting scheme.

    Returns
    -------
    tuple[int, float] | None
        ``(groundings, weight)``, or ``None`` when the chain does not
        reach this candidate.
    """
    indices, groundings = row
    position = torch.nonzero(indices == candidate_id, as_tuple=True)[0]
    if position.numel() == 0:
        return None
    count = groundings[position[0]]
    weight = evidence_weights(count.reshape(1), strategy)[0]
    return int(count.item()), float(weight.item())
