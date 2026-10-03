"""Inverse-relation-frequency outgoing-edge-type selector for random walks."""

import torch

from ..graph import EdgeTypeTuple, HeteroGraph


class RelationWeightedEdgeSelector:
    """Select outgoing edge types with probability inverse to their edge count.

    Parameters
    ----------
    graph : HeteroGraph
        The graph whose edge counts set the weights.
    generator : torch.Generator
        Source of randomness.
    """

    def __init__(self, graph: HeteroGraph, generator: torch.Generator) -> None:
        self._generator = generator
        self._weights = {
            node_type: torch.tensor(
                [1.0 / graph.edge_count(et) for et in edge_types],
                dtype=torch.float32,
            )
            for node_type in graph.node_types
            if (edge_types := graph.outgoing_edge_types(node_type))
        }

    def select(
        self,
        node_type: str,
        candidates: tuple[EdgeTypeTuple, ...],
    ) -> EdgeTypeTuple:
        """Select one of ``node_type``'s outgoing edge types.

        Parameters
        ----------
        node_type : str
            The node type being stepped from.
        candidates : tuple[EdgeTypeTuple, ...]
            Its outgoing edge types, as returned by
            :meth:`~anyburl.graph.HeteroGraph.outgoing_edge_types`.

        Returns
        -------
        EdgeTypeTuple
            The chosen edge type.
        """
        weights = self._weights[node_type]
        index = int(torch.multinomial(weights, 1, generator=self._generator).item())
        return candidates[index]
