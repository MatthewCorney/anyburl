"""Every body chain's evidence for one query, computed in one pass."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .scoring import GroundingWeighting, evidence_weights
from .sides import ChainSide, SideEvidence

__all__ = ["ChainSideBatch", "QueryChainEvidence"]

NO_RULE_BASE: float = 1.0
"""Complement factor of a chain with no rule holding for the query."""


@dataclass(frozen=True, slots=True)
class QueryChainEvidence:
    """Every body chain's evidence for one query, flattened across chains.

    Parameters
    ----------
    rows : Tensor
        Chain row of each entry.
    side : SideEvidence
        Candidate, contribution and grounding count of each entry.
    """

    rows: Tensor
    side: SideEvidence


@dataclass(frozen=True, slots=True)
class ChainSideBatch:
    """One direction's chain sides, with candidate confidences in one table.

    Parameters
    ----------
    sides : tuple[ChainSide, ...]
        The chain sides, in row order.
    num_candidates : int
        Number of entities a query ranks.
    grounded_keys : Tensor
        Ascending ``row * num_candidates + candidate`` of every candidate
        some AC1 rule is grounded on.
    grounded_values : Tensor
        Aggregated confidence of each key.
    """

    sides: tuple[ChainSide, ...]
    num_candidates: int
    grounded_keys: Tensor
    grounded_values: Tensor

    @classmethod
    def build(cls, sides: Sequence[ChainSide], num_candidates: int) -> "ChainSideBatch":
        """Index every side's candidate confidences by row and candidate.

        Parameters
        ----------
        sides : Sequence[ChainSide]
            The chain sides, in row order.
        num_candidates : int
            Number of entities a query ranks.

        Returns
        -------
        ChainSideBatch
            The sides with their combined confidence table.
        """
        keys = [
            side.candidate_ids + row * num_candidates for row, side in enumerate(sides)
        ]
        values = [side.candidate_values for side in sides]
        return cls(
            sides=tuple(sides),
            num_candidates=num_candidates,
            grounded_keys=torch.cat([torch.zeros(0, dtype=torch.long), *keys]),
            grounded_values=torch.cat([torch.zeros(0), *values]),
        )

    def evidence(
        self, query_id: int, weighting: GroundingWeighting
    ) -> QueryChainEvidence:
        """Return every chain's contribution to each candidate it reaches.

        Parameters
        ----------
        query_id : int
            The entity being queried.
        weighting : GroundingWeighting
            How grounding counts weigh on each contribution.

        Returns
        -------
        QueryChainEvidence
            One entry per chain and reached candidate, in row order.
        """
        fetched = [side.fetch(query_id) for side in self.sides]
        sizes = torch.tensor([found.numel() for found, _ in fetched], dtype=torch.long)
        rows = torch.repeat_interleave(torch.arange(len(self.sides)), sizes)
        candidates = torch.cat(
            [torch.zeros(0, dtype=torch.long), *(found for found, _ in fetched)]
        )
        counts = torch.cat([torch.zeros(0), *(count for _, count in fetched)]).float()
        weights = evidence_weights(counts, weighting)
        query_bases = self._query_bases(query_id)[rows]
        grounded_bases = 1.0 - self._grounded_confidences(rows, candidates)
        complement = query_bases**weights * grounded_bases**weights
        return QueryChainEvidence(
            rows=rows,
            side=SideEvidence(candidates, 1.0 - complement, counts),
        )

    def _query_bases(self, query_id: int) -> Tensor:
        """Return each row's ``1 - confidence`` of rules holding for the query."""
        confidences = [side.confidence_for(query_id) for side in self.sides]
        return torch.tensor(
            [
                NO_RULE_BASE if confidence is None else 1.0 - confidence
                for confidence in confidences
            ],
            dtype=torch.float64,
        ).float()

    def _grounded_confidences(self, rows: Tensor, candidates: Tensor) -> Tensor:
        """Return the confidence grounded on each entry's candidate, else 0."""
        if self.grounded_keys.numel() == 0:
            return torch.zeros(candidates.numel())
        keys = rows * self.num_candidates + candidates
        positions = torch.searchsorted(self.grounded_keys, keys)
        positions = positions.clamp(max=self.grounded_keys.numel() - 1)
        is_grounded = self.grounded_keys[positions] == keys
        return torch.where(is_grounded, self.grounded_values[positions], 0.0)
