"""Uniform outgoing-edge-type selector for random walks."""

import torch

from ..graph import EdgeTypeTuple


class UniformEdgeSelector:
    """Select outgoing edge types uniformly at random.

    Parameters
    ----------
    generator : torch.Generator
        Source of randomness.
    """

    def __init__(self, generator: torch.Generator) -> None:
        self._generator = generator

    def select(
        self,
        _node_type: str,
        candidates: tuple[EdgeTypeTuple, ...],
    ) -> EdgeTypeTuple:
        """Select one candidate edge type uniformly at random."""
        index = int(
            torch.randint(0, len(candidates), (1,), generator=self._generator).item()
        )
        return candidates[index]
