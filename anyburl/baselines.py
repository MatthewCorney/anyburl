"""Reference scorers that give a learned model's metrics a scale.

Each scorer satisfies :class:`~anyburl.evaluation.EntityScorer`, so
:class:`~anyburl.evaluation.LinkPredictionEvaluator` measures them exactly as
it measures a fitted model.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .graph import EdgeTypeTuple, HeteroGraph, validate_chain

__all__ = [
    "MetaPathScorer",
    "PopularityScorer",
    "RandomScorer",
]


@dataclass(frozen=True, slots=True)
class _Endpoints:
    """Candidate counts on each side of a target relation.

    Parameters
    ----------
    num_heads : int
        Number of candidate head entities.
    num_tails : int
        Number of candidate tail entities.
    """

    num_heads: int
    num_tails: int

    @classmethod
    def of(cls, graph: HeteroGraph, edge_type: EdgeTypeTuple) -> "_Endpoints":
        """Read the endpoint counts of ``edge_type`` from ``graph``."""
        src_type, _, dst_type = edge_type
        return cls(
            num_heads=graph.node_count(src_type),
            num_tails=graph.node_count(dst_type),
        )


class RandomScorer:
    """Scores candidates uniformly at random, deterministically given ``seed``.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    target_edge_type : EdgeTypeTuple
        The relation being predicted.
    seed : int
        Seed for the random scores.
    """

    def __init__(
        self,
        graph: HeteroGraph,
        target_edge_type: EdgeTypeTuple,
        *,
        seed: int = 0,
    ) -> None:
        self._endpoints = _Endpoints.of(graph, target_edge_type)
        self._generator = torch.Generator().manual_seed(seed)

    def score_tails(self, head_id: int) -> Tensor:
        """Return uniform random scores over all candidate tails."""
        del head_id
        return torch.rand(self._endpoints.num_tails, generator=self._generator)

    def score_heads(self, tail_id: int) -> Tensor:
        """Return uniform random scores over all candidate heads."""
        del tail_id
        return torch.rand(self._endpoints.num_heads, generator=self._generator)


class PopularityScorer:
    """Scores every candidate by its degree in the target relation.

    The score ignores the query, so it measures how much of the ranking is
    explained by degree alone.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    target_edge_type : EdgeTypeTuple
        The relation being predicted.
    """

    def __init__(self, graph: HeteroGraph, target_edge_type: EdgeTypeTuple) -> None:
        endpoints = _Endpoints.of(graph, target_edge_type)
        edge_index = graph.edge_index(target_edge_type)
        weights = torch.ones(edge_index.size(1))
        self._tail_degree = torch.zeros(endpoints.num_tails).scatter_add_(
            0, edge_index[1], weights
        )
        self._head_degree = torch.zeros(endpoints.num_heads).scatter_add_(
            0, edge_index[0], weights
        )

    def score_tails(self, head_id: int) -> Tensor:
        """Return each candidate tail's degree, ignoring the query."""
        del head_id
        return self._tail_degree.clone()

    def score_heads(self, tail_id: int) -> Tensor:
        """Return each candidate head's degree, ignoring the query."""
        del tail_id
        return self._head_degree.clone()


class MetaPathScorer:
    """Counts paths along one fixed edge-type chain, unweighted.

    Equivalent to a single cyclic rule scored by grounding count with its
    confidence ignored.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    chain : tuple[EdgeTypeTuple, ...]
        Edge types to traverse, head endpoint to tail endpoint.

    Raises
    ------
    InvalidRuleError
        If ``chain`` is empty or its edge types do not join end to end.
    """

    def __init__(self, graph: HeteroGraph, chain: tuple[EdgeTypeTuple, ...]) -> None:
        validate_chain(chain)
        self._matrices = tuple(graph.get_csr_matrix(et) for et in chain)
        self._num_heads = graph.node_count(chain[0][0])
        self._num_tails = graph.node_count(chain[-1][2])

    def score_tails(self, head_id: int) -> Tensor:
        """Return the number of chain paths from ``head_id`` to each tail."""
        vector = torch.zeros(1, self._num_heads)
        vector[0, head_id] = 1.0
        for matrix in self._matrices:
            vector = vector @ matrix
        return vector.squeeze(0)

    def score_heads(self, tail_id: int) -> Tensor:
        """Return the number of chain paths from each head to ``tail_id``."""
        vector = torch.zeros(self._num_tails, 1)
        vector[tail_id, 0] = 1.0
        for matrix in reversed(self._matrices):
            vector = matrix @ vector
        return vector.squeeze(1)
