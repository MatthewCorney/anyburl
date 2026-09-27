"""Reference scorers that give a learned MRR a scale to be read against.

An MRR means nothing on its own. On DBLP, ranking papers at random scores
0.0021, ranking by popularity scores 0.0120, and counting co-author paths
--- ten lines, no learning --- scores 0.1374, while the fitted rule model
scores 0.1896. The learned rules are worth +38% over the best heuristic, not
90x over nothing, and only the baselines make that visible.

Each scorer satisfies :class:`~anyburl.evaluation.EntityScorer`, so
:class:`~anyburl.evaluation.LinkPredictionEvaluator` measures them exactly as
it measures a fitted model.
"""

from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor

from ._logging import get_logger
from .graph import EdgeTypeTuple, HeteroGraph

logger = get_logger(__name__)

__all__ = [
    "MetaPathScorer",
    "PopularityScorer",
    "RandomScorer",
    "reverse_chain",
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
    """Scores candidates uniformly at random --- the true floor.

    Any model that cannot beat this is not ranking at all. Deterministic
    given ``seed`` so a reported figure can be reproduced.

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

    The score ignores the query entirely, so it measures how much of a
    benchmark is explained by "popular things are popular". A model close
    to this number has learned the degree distribution and little else.

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


def reverse_chain(chain: tuple[EdgeTypeTuple, ...]) -> tuple[EdgeTypeTuple, ...]:
    """Return the edge-type chain that walks ``chain`` backwards.

    Parameters
    ----------
    chain : tuple[EdgeTypeTuple, ...]
        Edge types in forward order.

    Returns
    -------
    tuple[EdgeTypeTuple, ...]
        The same relations reversed in order and in endpoint direction.
        Note the result names edge types that need not exist in a graph;
        :class:`MetaPathScorer` transposes matrices rather than looking
        these up.
    """
    return tuple((dst, relation, src) for src, relation, dst in reversed(chain))


class MetaPathScorer:
    """Counts paths along one fixed edge-type chain, unweighted.

    This is the strongest "obvious" heuristic for a benchmark: pick the
    meta-path a domain expert would name --- co-authorship on DBLP --- and
    rank by how many such paths connect the query to each candidate. It is
    what a learned rule set has to beat to justify itself, since it is one
    rule with confidence ignored.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    chain : tuple[EdgeTypeTuple, ...]
        Edge types to traverse, head endpoint to tail endpoint.

    Raises
    ------
    ValueError
        If ``chain`` is empty or its edge types do not join end to end.
    """

    def __init__(self, graph: HeteroGraph, chain: tuple[EdgeTypeTuple, ...]) -> None:
        if not chain:
            raise ValueError("chain must contain at least one edge type")
        for earlier, later in pairwise(chain):
            if earlier[2] != later[0]:
                raise ValueError(
                    f"chain does not join: {earlier!r} ends in {earlier[2]!r} "
                    f"but {later!r} starts at {later[0]!r}"
                )

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
