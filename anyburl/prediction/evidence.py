"""Ground learned rules and report their evidence for every candidate of a query."""

from collections.abc import Sequence
from typing import assert_never

import torch

from ..acyclic.evaluation import AcyclicIndex
from ..acyclic.scoring import AcyclicScorer, Firings
from ..core import BodyChain, EdgeTypeTuple, QueryDirection
from ..exceptions import ConfigurationError
from ..graph import HeteroGraph
from ..metrics import RuleMetrics
from ..rule import Rule, RuleType
from .grounding import build_groundings
from .records import AcyclicEvidence, ChainEvidence, PredictionConfig
from .scoring import GroundingWeighting
from .sides import ChainSide, build_chain_sides, side_evidence

__all__ = ["RuleEvidence"]


class RuleEvidence:
    """The evidence one relation's rules give for each candidate of a query.

    Rules sharing a body chain share one grounding. Scorers such as
    :class:`~anyburl.prediction.RulePredictor` and
    :class:`~anyburl.aggregation.AggregatedScorer` combine this evidence into
    a ranking. AC2 rules are ignored.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to ground rules against.
    results : Sequence[tuple[Rule, RuleMetrics]]
        Rules paired with their evaluated metrics. All rules must predict
        the same head relation.
    config : PredictionConfig | None
        Grounding weighting and unseen-negative correction.
        ``None`` uses the defaults; ``aggregation`` is not used here.

    Raises
    ------
    ConfigurationError
        If ``results`` is empty or its rules predict different relations.
    """

    def __init__(
        self,
        graph: HeteroGraph,
        results: Sequence[tuple[Rule, RuleMetrics]],
        *,
        config: PredictionConfig | None = None,
    ) -> None:
        if not results:
            raise ConfigurationError("results must not be empty")
        head_relations = {rule.head.edge_signature for rule, _ in results}
        if len(head_relations) > 1:
            raise ConfigurationError(
                f"All rules must predict the same head relation, "
                f"got {len(head_relations)} distinct: {head_relations}"
            )

        config = config if config is not None else PredictionConfig()
        self._weighting = config.grounding_weighting
        self._head_edge_type: EdgeTypeTuple = head_relations.pop()
        head_type, _, tail_type = self._head_edge_type
        self._num_heads = graph.node_count(head_type)
        self._num_tails = graph.node_count(tail_type)
        tail_sides, head_sides = build_chain_sides(
            results,
            build_groundings(graph, results),
            config.unseen_negatives,
        )
        self._tail_sides = tuple(tail_sides)
        self._head_sides = tuple(head_sides)
        self._acyclic = AcyclicScorer(
            graph.cached(AcyclicIndex), results, config.unseen_negatives
        )

    @property
    def head_edge_type(self) -> EdgeTypeTuple:
        """Return the relation the rules predict."""
        return self._head_edge_type

    @property
    def weighting(self) -> GroundingWeighting:
        """Return how grounding counts weigh on each chain's contribution."""
        return self._weighting

    @property
    def chains(self) -> tuple[tuple[BodyChain, RuleType], ...]:
        """The body chain and rule type of each :meth:`chain_evidence` row."""
        return tuple((side.chain, side.rule_type) for side in self._tail_sides)

    def num_candidates(self, direction: QueryDirection) -> int:
        """Return how many entities a query in ``direction`` ranks."""
        match direction:
            case QueryDirection.TAIL:
                return self._num_tails
            case QueryDirection.HEAD:
                return self._num_heads
            case _ as unreachable:
                assert_never(unreachable)

    def sides(self, direction: QueryDirection) -> tuple[ChainSide, ...]:
        """Return each body chain's rules as seen from ``direction``."""
        match direction:
            case QueryDirection.TAIL:
                return self._tail_sides
            case QueryDirection.HEAD:
                return self._head_sides
            case _ as unreachable:
                assert_never(unreachable)

    def acyclic_firings(self, direction: QueryDirection, query_id: int) -> Firings:
        """Return every acyclic rule firing for one query."""
        match direction:
            case QueryDirection.TAIL:
                return self._acyclic.tail_firings(query_id)
            case QueryDirection.HEAD:
                return self._acyclic.head_firings(query_id)
            case _ as unreachable:
                assert_never(unreachable)

    def chain_evidence(self, direction: QueryDirection, query_id: int) -> ChainEvidence:
        """Break a query's chain evidence down by body chain.

        Parameters
        ----------
        direction : QueryDirection
            ``TAIL`` to score tails of head ``query_id``, ``HEAD`` to score
            heads of tail ``query_id``.
        query_id : int
            The entity being queried.

        Returns
        -------
        ChainEvidence
            One row per body chain over every candidate, in the order of
            :attr:`chains`. Tail and head sides share that order.
        """
        num_candidates = self.num_candidates(direction)
        rows = [
            side_evidence(side, query_id, num_candidates, self._weighting)
            for side in self.sides(direction)
        ]
        empty = torch.zeros(0, num_candidates)
        return ChainEvidence(
            contributions=torch.stack([row[0] for row in rows]) if rows else empty,
            groundings=torch.stack([row[1] for row in rows]) if rows else empty,
        )

    def acyclic_evidence(
        self, direction: QueryDirection, query_id: int
    ) -> AcyclicEvidence:
        """Summarise the acyclic rules firing for every candidate of a query.

        Parameters
        ----------
        direction : QueryDirection
            ``TAIL`` to score tails of head ``query_id``, ``HEAD`` to score
            heads of tail ``query_id``.
        query_id : int
            The entity being queried.

        Returns
        -------
        AcyclicEvidence
            Strongest confidence, noisy-or and firing count per candidate.
        """
        candidates, confidences = self.acyclic_firings(direction, query_id)
        num_candidates = self.num_candidates(direction)
        max_confidence = torch.zeros(num_candidates)
        max_confidence.scatter_reduce_(0, candidates, confidences, "amax")
        log_complement = torch.zeros(num_candidates)
        log_complement.index_add_(0, candidates, torch.log1p(-confidences))
        counts = torch.zeros(num_candidates)
        counts.index_add_(0, candidates, torch.ones_like(confidences))
        return AcyclicEvidence(
            max_confidence=max_confidence,
            noisy_or=1.0 - torch.exp(log_complement),
            firings=counts,
        )
