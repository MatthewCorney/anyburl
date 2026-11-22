"""How body grounding counts are weighted into a rule's contribution."""

from collections.abc import Mapping
from enum import StrEnum
from typing import assert_never

import torch
from torch import Tensor

__all__ = [
    "GROUNDING_WEIGHT_OFFSET",
    "LEGACY_GROUNDING_WEIGHTINGS",
    "TIE_BREAK_WEIGHT",
    "ConfidenceAggregation",
    "GroundingWeighting",
    "evidence_weights",
    "weight_for_candidate",
]

GROUNDING_WEIGHT_OFFSET: float = 1.0
"""Offset in ``log2(offset + groundings)``, set so one grounding weighs 1.0."""

TIE_BREAK_WEIGHT: float = 1e-4
"""Scale of the noisy-or term that breaks ties under maximum aggregation."""

LEGACY_GROUNDING_WEIGHTINGS: Mapping[str, str] = {
    "noisy_or": "once",
    "path_weighted": "log_count",
}
"""Former :class:`GroundingWeighting` values, mapped to their current ones."""


class ConfidenceAggregation(StrEnum):
    """How the contributions of different body chains combine into a score.

    ``"noisy_or"`` is also a former :class:`GroundingWeighting` value, which
    still parses there as :attr:`GroundingWeighting.ONCE`.

    Attributes
    ----------
    NOISY_OR : str
        ``1 - prod(1 - contribution)``: every firing chain raises the score.
    MAXIMUM : str
        The strongest chain's contribution, with ties broken by a small
        noisy-or term. Approximates AnyBURL's ranking by the best rule,
        then the next best, so many weak rules cannot outrank one strong
        rule.
    LEXICOGRAPHIC : str
        AnyBURL's ranking: each candidate's strongest contributions, sorted
        descending, compared position by position. Scores stay within a
        small offset of the strongest contribution.
    """

    NOISY_OR = "noisy_or"
    MAXIMUM = "maximum"
    LEXICOGRAPHIC = "lexicographic"


class GroundingWeighting(StrEnum):
    """How the number of body groundings reaching a candidate weights a rule.

    The former values ``"noisy_or"`` and ``"path_weighted"`` still parse,
    to :attr:`ONCE` and :attr:`LOG_COUNT`.

    Attributes
    ----------
    ONCE : str
        Each firing rule contributes its confidence once, however many
        body groundings support the candidate.
    LOG_COUNT : str
        A rule's confidence is weighted by ``log2(1 + groundings)``, so a
        candidate reached by many groundings outranks one reached by a
        single grounding. A single grounding reproduces :attr:`ONCE`.
    """

    ONCE = "once"
    LOG_COUNT = "log_count"

    @classmethod
    def _missing_(cls, value: object) -> "GroundingWeighting | None":
        """Return the member a former value names, or ``None``."""
        match value:
            case str() if value in LEGACY_GROUNDING_WEIGHTINGS:
                return cls(LEGACY_GROUNDING_WEIGHTINGS[value])
            case _:
                return None


def evidence_weights(groundings: Tensor, weighting: GroundingWeighting) -> Tensor:
    """Convert body grounding counts into per-candidate evidence weights.

    Parameters
    ----------
    groundings : Tensor
        Number of body groundings supporting each candidate.
    weighting : GroundingWeighting
        The weighting scheme.

    Returns
    -------
    Tensor
        Weights parallel to ``groundings``.
    """
    match weighting:
        case GroundingWeighting.ONCE:
            return torch.ones_like(groundings)
        case GroundingWeighting.LOG_COUNT:
            return torch.log2(GROUNDING_WEIGHT_OFFSET + groundings)
        case _ as unreachable:
            assert_never(unreachable)


def weight_for_candidate(
    row: tuple[Tensor, Tensor],
    candidate_id: int,
    weighting: GroundingWeighting,
) -> tuple[int, float] | None:
    """Return one candidate's grounding count and evidence weight, if reached.

    Parameters
    ----------
    row : tuple[Tensor, Tensor]
        Candidate indices and grounding counts from a chain grounding.
    candidate_id : int
        The candidate to look for.
    weighting : GroundingWeighting
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
    weight = evidence_weights(count.reshape(1), weighting)[0]
    return int(count.item()), float(weight.item())
