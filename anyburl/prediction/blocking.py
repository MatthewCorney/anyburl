"""Block candidates already linked to the query entity by any relation.

Some benchmarks remove every held-out triple whose two entities are already
connected in training, so on them any such candidate is known to be wrong.
:func:`linked_pair_share` measures how many held-out triples are linked;
:class:`LinkedPairBlocker` ranks linked candidates below every other one.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..core import EdgeTypeTuple, EntityScorer, Triple
from ..exceptions import ConfigurationError
from ..graph import HeteroGraph

__all__ = [
    "BLOCKED_SCORE",
    "LinkIndex",
    "LinkedEntities",
    "LinkedPairBlocker",
    "linked_pair_share",
]

BLOCKED_SCORE: float = -1.0
"""Score given to a blocked candidate, below every non-negative rule score."""


@dataclass(frozen=True, slots=True)
class LinkedEntities:
    """For each entity of one node type, the linked entities of another.

    Parameters
    ----------
    crow : Tensor
        Row offsets, one more than the number of query-type entities.
    col : Tensor
        Linked candidate-type entity ids, sorted within each row.
    """

    crow: Tensor
    col: Tensor

    def of(self, entity_id: int) -> Tensor:
        """Return the entities linked to ``entity_id``."""
        return self.col[int(self.crow[entity_id]) : int(self.crow[entity_id + 1])]


class LinkIndex:
    """Which entities are linked by any edge type, in either direction.

    Built once per graph and shared via ``graph.cached(LinkIndex)``.

    Parameters
    ----------
    graph : HeteroGraph
        The graph whose edges define the links.
    """

    def __init__(self, graph: HeteroGraph) -> None:
        self._graph = graph
        self._between: dict[tuple[str, str], LinkedEntities] = {}

    def between(self, query_type: str, candidate_type: str) -> LinkedEntities:
        """Return, per ``query_type`` entity, its linked ``candidate_type`` entities.

        Parameters
        ----------
        query_type : str
            Node type of the query entity.
        candidate_type : str
            Node type of the candidates.

        Returns
        -------
        LinkedEntities
            Linked candidates of every query-type entity.
        """
        key = (query_type, candidate_type)
        links = self._between.get(key)
        if links is None:
            links = self._build(query_type, candidate_type)
            self._between[key] = links
        return links

    def _build(self, query_type: str, candidate_type: str) -> LinkedEntities:
        """Collect edges between the two node types and index them by query entity."""
        graph = self._graph
        sources: list[Tensor] = []
        targets: list[Tensor] = []
        for edge_type in graph.edge_types:
            if graph.edge_count(edge_type) == 0:
                continue
            src_type, _, dst_type = edge_type
            edges = graph.edge_index(edge_type)
            if (src_type, dst_type) == (query_type, candidate_type):
                sources.append(edges[0])
                targets.append(edges[1])
            if (dst_type, src_type) == (query_type, candidate_type):
                sources.append(edges[1])
                targets.append(edges[0])
        num_queries = graph.node_count(query_type)
        num_candidates = graph.node_count(candidate_type)
        if not sources:
            empty = torch.zeros(0, dtype=torch.long)
            return LinkedEntities(torch.zeros(num_queries + 1, dtype=torch.long), empty)
        keys = torch.unique(torch.cat(sources) * num_candidates + torch.cat(targets))
        rows = keys // num_candidates
        crow = torch.zeros(num_queries + 1, dtype=torch.long)
        crow[1:] = torch.cumsum(torch.bincount(rows, minlength=num_queries), dim=0)
        return LinkedEntities(crow, keys % num_candidates)


class LinkedPairBlocker:
    """Wraps a scorer so candidates linked to the query entity score lowest.

    Satisfies :class:`~anyburl.core.EntityScorer`.

    Parameters
    ----------
    scorer : EntityScorer
        The scorer whose scores are blocked.
    graph : HeteroGraph
        The graph whose edges, of any relation, define linked pairs.
    edge_type : EdgeTypeTuple
        The relation being scored, which fixes the node types.
    """

    def __init__(
        self, scorer: EntityScorer, graph: HeteroGraph, edge_type: EdgeTypeTuple
    ) -> None:
        head_type, _, tail_type = edge_type
        index = graph.cached(LinkIndex)
        self._scorer = scorer
        self._tails_of = index.between(head_type, tail_type)
        self._heads_of = index.between(tail_type, head_type)

    def score_tails(self, head_id: int) -> Tensor:
        """Score tails of ``head_id``, blocking tails already linked to it.

        Parameters
        ----------
        head_id : int
            The source entity index.

        Returns
        -------
        Tensor
            The wrapped scores, ``BLOCKED_SCORE`` at linked tails.
        """
        return _blocked(self._scorer.score_tails(head_id), self._tails_of.of(head_id))

    def score_heads(self, tail_id: int) -> Tensor:
        """Score heads of ``tail_id``, blocking heads already linked to it.

        Parameters
        ----------
        tail_id : int
            The destination entity index.

        Returns
        -------
        Tensor
            The wrapped scores, ``BLOCKED_SCORE`` at linked heads.
        """
        return _blocked(self._scorer.score_heads(tail_id), self._heads_of.of(tail_id))


def _blocked(scores: Tensor, linked: Tensor) -> Tensor:
    """Return ``scores`` with ``linked`` positions set to ``BLOCKED_SCORE``."""
    if linked.numel() == 0:
        return scores
    blocked = scores.clone()
    blocked[linked] = BLOCKED_SCORE
    return blocked


def linked_pair_share(graph: HeteroGraph, triples: Sequence[Triple]) -> float:
    """Return the share of ``triples`` whose entities are already linked.

    A pair counts as linked when any edge of ``graph``, of any relation and
    in either direction, joins its head and tail.

    Parameters
    ----------
    graph : HeteroGraph
        The graph whose edges define links, usually the training graph.
    triples : Sequence[Triple]
        Held-out triples, usually the validation split.

    Returns
    -------
    float
        The linked share, in ``[0, 1]``.

    Raises
    ------
    ConfigurationError
        If ``triples`` is empty.
    """
    if not triples:
        raise ConfigurationError("linked_pair_share needs at least one triple")
    index = graph.cached(LinkIndex)
    linked = sum(
        bool(
            (
                index.between(triple.head_type, triple.tail_type).of(triple.head_id)
                == triple.tail_id
            ).any()
        )
        for triple in triples
    )
    return linked / len(triples)
