"""Ground learned rules and report their evidence for every candidate of a query."""

from dataclasses import dataclass
from typing import assert_never

import torch
from torch import Tensor

from ..acyclic.evaluation import AcyclicIndex
from ..acyclic.scoring import AcyclicScorer, Firings
from ..core import BodyChain, EdgeTypeTuple, QueryDirection
from ..exceptions import ConfigurationError
from ..graph import HeteroGraph
from ..rule import RuleType
from ..rule_evaluation import RuleSet
from .batch import ChainSideBatch, QueryChainEvidence
from .grounding import build_groundings
from .records import AcyclicEvidence, ChainEvidence, PredictionConfig
from .scoring import GroundingWeighting
from .sides import ChainSide, SideEvidence, build_chain_sides

__all__ = ["RuleEvidence"]

UNSELECTED_SLOT: int = -1
"""Slot of a candidate left out of a restricted evidence query."""


@dataclass(slots=True)
class _QueryMemo:
    """What has been computed so far for the most recent query.

    Parameters
    ----------
    key : tuple[QueryDirection, int]
        The query's direction and entity.
    firings : Firings | None
        Its acyclic rule firings, once computed.
    chains : QueryChainEvidence | None
        Its chain evidence, once computed.
    reached : Tensor | None
        Its reached candidates, once computed.
    """

    key: tuple[QueryDirection, int]
    firings: Firings | None = None
    chains: QueryChainEvidence | None = None
    reached: Tensor | None = None


class RuleEvidence:
    """The evidence one relation's rules give for each candidate of a query.

    Rules sharing a body chain share one grounding. Scorers such as
    :class:`~anyburl.prediction.RulePredictor` and
    :class:`~anyburl.aggregation.AggregatedScorer` combine this evidence into
    a ranking. AC2 rules are ignored.

    The most recent query's firings, chain evidence and reached candidates
    are kept, so several scorers ranking the same query ground it once.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to ground rules against.
    results : RuleSet
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
        results: RuleSet,
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
        self._tail_batch = ChainSideBatch.build(tail_sides, self._num_tails)
        self._head_batch = ChainSideBatch.build(head_sides, self._num_heads)
        self._acyclic = AcyclicScorer(
            graph.cached(AcyclicIndex), results, config.unseen_negatives
        )
        self._memo: _QueryMemo | None = None

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
        return tuple((side.chain, side.rule_type) for side in self._tail_batch.sides)

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
        return self._batch(direction).sides

    def _batch(self, direction: QueryDirection) -> ChainSideBatch:
        """Return the chain sides scoring queries in ``direction``."""
        match direction:
            case QueryDirection.TAIL:
                return self._tail_batch
            case QueryDirection.HEAD:
                return self._head_batch
            case _ as unreachable:
                assert_never(unreachable)

    def acyclic_firings(self, direction: QueryDirection, query_id: int) -> Firings:
        """Return every acyclic rule firing for one query."""
        memo = self._memo_for(direction, query_id)
        if memo.firings is None:
            match direction:
                case QueryDirection.TAIL:
                    memo.firings = self._acyclic.tail_firings(query_id)
                case QueryDirection.HEAD:
                    memo.firings = self._acyclic.head_firings(query_id)
                case _ as unreachable:
                    assert_never(unreachable)
        return memo.firings

    def _memo_for(self, direction: QueryDirection, query_id: int) -> _QueryMemo:
        """Return this query's memo, starting a new one for a new query."""
        key = (direction, query_id)
        if self._memo is None or self._memo.key != key:
            self._memo = _QueryMemo(key)
        return self._memo

    def _query_chain_evidence(
        self, direction: QueryDirection, query_id: int
    ) -> QueryChainEvidence:
        """Return every chain's evidence for one query."""
        memo = self._memo_for(direction, query_id)
        if memo.chains is None:
            memo.chains = self._batch(direction).evidence(query_id, self._weighting)
        return memo.chains

    def chain_contributions(
        self, direction: QueryDirection, query_id: int
    ) -> SideEvidence:
        """Return every chain's contribution to each candidate it reaches.

        Parameters
        ----------
        direction : QueryDirection
            ``TAIL`` to score tails of head ``query_id``, ``HEAD`` to score
            heads of tail ``query_id``.
        query_id : int
            The entity being queried.

        Returns
        -------
        SideEvidence
            One entry per body chain and candidate it grounds, chains in the
            order of :attr:`chains`. A candidate appears once per chain.
        """
        return self._query_chain_evidence(direction, query_id).side

    def reached_candidates(self, direction: QueryDirection, query_id: int) -> Tensor:
        """Return the sorted ids of candidates some rule gives a positive score.

        Parameters
        ----------
        direction : QueryDirection
            ``TAIL`` to score tails of head ``query_id``, ``HEAD`` to score
            heads of tail ``query_id``.
        query_id : int
            The entity being queried.

        Returns
        -------
        Tensor
            Unique candidate ids, ascending.
        """
        memo = self._memo_for(direction, query_id)
        if memo.reached is None:
            acyclic_ids, _ = self.acyclic_firings(direction, query_id)
            found = self._query_chain_evidence(direction, query_id).side
            is_reached = torch.zeros(self.num_candidates(direction), dtype=torch.bool)
            is_reached[found.candidates[found.contributions > 0]] = True
            is_reached[acyclic_ids] = True
            memo.reached = is_reached.nonzero().flatten()
        return memo.reached

    def chain_evidence(
        self,
        direction: QueryDirection,
        query_id: int,
        *,
        candidates: Tensor | None = None,
    ) -> ChainEvidence:
        """Break a query's chain evidence down by body chain.

        Parameters
        ----------
        direction : QueryDirection
            ``TAIL`` to score tails of head ``query_id``, ``HEAD`` to score
            heads of tail ``query_id``.
        query_id : int
            The entity being queried.
        candidates : Tensor | None
            Unique ids of the candidates to report, in column order.
            ``None`` reports every candidate.

        Returns
        -------
        ChainEvidence
            One row per body chain, in the order of :attr:`chains`. Tail and
            head sides share that order.
        """
        slots = self._candidate_slots(direction, candidates)
        num_columns = int((slots != UNSELECTED_SLOT).sum().item())
        found = self._query_chain_evidence(direction, query_id)
        columns = slots[found.side.candidates]
        is_selected = columns != UNSELECTED_SLOT
        rows, columns = found.rows[is_selected], columns[is_selected]
        contributions = torch.zeros(len(self.sides(direction)), num_columns)
        groundings = torch.zeros_like(contributions)
        contributions[rows, columns] = found.side.contributions[is_selected]
        groundings[rows, columns] = found.side.groundings[is_selected]
        return ChainEvidence(contributions=contributions, groundings=groundings)

    def acyclic_evidence(
        self,
        direction: QueryDirection,
        query_id: int,
        *,
        candidates: Tensor | None = None,
    ) -> AcyclicEvidence:
        """Summarise the acyclic rules firing for every candidate of a query.

        Parameters
        ----------
        direction : QueryDirection
            ``TAIL`` to score tails of head ``query_id``, ``HEAD`` to score
            heads of tail ``query_id``.
        query_id : int
            The entity being queried.
        candidates : Tensor | None
            Unique ids of the candidates to report, in order. ``None``
            reports every candidate.

        Returns
        -------
        AcyclicEvidence
            Strongest confidence, noisy-or and firing count per candidate.
        """
        slots = self._candidate_slots(direction, candidates)
        num_columns = int((slots != UNSELECTED_SLOT).sum().item())
        fired, confidences = self.acyclic_firings(direction, query_id)
        columns = slots[fired]
        is_selected = columns != UNSELECTED_SLOT
        columns, confidences = columns[is_selected], confidences[is_selected]
        max_confidence = torch.zeros(num_columns)
        max_confidence.scatter_reduce_(0, columns, confidences, "amax")
        log_complement = torch.zeros(num_columns)
        log_complement.index_add_(0, columns, torch.log1p(-confidences))
        counts = torch.zeros(num_columns)
        counts.index_add_(0, columns, torch.ones_like(confidences))
        return AcyclicEvidence(
            max_confidence=max_confidence,
            noisy_or=1.0 - torch.exp(log_complement),
            firings=counts,
        )

    def _candidate_slots(
        self, direction: QueryDirection, candidates: Tensor | None
    ) -> Tensor:
        """Map each candidate id to its column, or :data:`UNSELECTED_SLOT`."""
        num_candidates = self.num_candidates(direction)
        if candidates is None:
            return torch.arange(num_candidates)
        slots = torch.full((num_candidates,), UNSELECTED_SLOT, dtype=torch.long)
        slots[candidates] = torch.arange(candidates.numel())
        return slots
