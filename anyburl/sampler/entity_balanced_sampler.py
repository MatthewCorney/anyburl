"""Triple sampling that gives every head entity equal weight."""

import torch

from .._logging import get_logger
from ..graph import EdgeTypeTuple
from .base import BaseTripleSampler, Triple

logger = get_logger(__name__)


class EntityBalancedTripleSampler(BaseTripleSampler):
    """Samples a head entity uniformly, then one of its edges.

    Uniform edge sampling picks a head with probability proportional to its
    degree; drawing the head first spreads the sample evenly across
    entities. With several eligible edge types the budget is split evenly
    between them and balanced within each.
    """

    def sample(self) -> list[Triple]:
        """Sample triples with every head entity equally likely.

        Returns
        -------
        list[Triple]
            Sampled triples, at most ``sample_size`` of them.
        """
        wanted = self._sample_size()
        per_type = self._per_type_quota(wanted)

        triples: list[Triple] = []
        for edge_type, quota in zip(self._edge_types, per_type, strict=True):
            if quota > 0:
                triples.extend(self._sample_within(edge_type, quota))

        logger.debug("Sampled %d triples via entity-balanced strategy", len(triples))
        return triples[:wanted]

    def _per_type_quota(self, wanted: int) -> list[int]:
        """Split the sample budget evenly across eligible edge types."""
        types = len(self._edge_types)
        base, remainder = divmod(wanted, types)
        return [base + (1 if i < remainder else 0) for i in range(types)]

    def _sample_within(self, edge_type: EdgeTypeTuple, quota: int) -> list[Triple]:
        """Draw ``quota`` triples of one edge type, balanced across heads.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The edge type to sample from.
        quota : int
            How many triples to draw.

        Returns
        -------
        list[Triple]
            Sampled triples.
        """
        edge_index = self._graph.edge_index(edge_type)
        heads = edge_index[0]
        present, inverse = torch.unique(heads, return_inverse=True)

        chosen_heads = torch.randint(
            0, present.numel(), (quota,), generator=self._generator
        )

        order = torch.argsort(inverse, stable=True)
        starts = torch.searchsorted(inverse[order], torch.arange(present.numel()))
        degrees = torch.bincount(inverse, minlength=present.numel())

        offsets = (
            torch.rand(quota, generator=self._generator) * degrees[chosen_heads]
        ).long()
        edge_positions = order[starts[chosen_heads] + offsets]

        return [
            self._extract_triple(edge_type, int(position))
            for position in edge_positions.tolist()
        ]
