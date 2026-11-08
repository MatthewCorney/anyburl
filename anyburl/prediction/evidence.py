"""Ground learned rules and report their evidence for every candidate of a query."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

import torch
from torch import Tensor

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
from .sides import ChainSide, SideEvidence, build_chain_sides, side_evidence

__all__ = ["RuleEvidence"]

UNSELECTED_SLOT: int = -1
"""Slot of a candidate left out of a restricted evidence query."""


@dataclass(frozen=True, slots=True)
class _QueryChainEvidence:
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

    @classmethod
    def concat(cls, found: Sequence[SideEvidence]) -> "_QueryChainEvidence":
        """Flatten per-chain evidence, keeping each entry's chain row."""
        sizes = torch.tensor(
            [side.candidates.numel() for side in found], dtype=torch.long
        )
        empty = torch.zeros(0)
        return cls(
            rows=torch.repeat_interleave(torch.arange(len(found)), sizes),
            side=SideEvidence(
                candidates=torch.cat(
                    [empty.long(), *(side.candidates for side in found)]
                ),
                contributions=torch.cat(
                    [empty, *(side.contributions for side in found)]
                ),
                groundings=torch.cat([empty, *(side.groundings for side in found)]),
            ),
        )


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
        self._latest: tuple[tuple[QueryDirection, int], _QueryChainEvidence] | None = (
            None
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

    def _query_chain_evidence(
        self, direction: QueryDirection, query_id: int
    ) -> _QueryChainEvidence:
        """Return every chain's evidence for one query.

        The latest query's evidence is kept, so asking for a query's
        :meth:`reached_candidates` and then its evidence grounds it once.
        """
        key = (direction, query_id)
        if self._latest is not None and self._latest[0] == key:
            return self._latest[1]
        found = _QueryChainEvidence.concat(
            [
                side_evidence(side, query_id, self._weighting)
                for side in self.sides(direction)
            ]
        )
        self._latest = (key, found)
        return found

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
        acyclic_ids, _ = self.acyclic_firings(direction, query_id)
        found = self._query_chain_evidence(direction, query_id).side
        chain_ids = found.candidates[found.contributions > 0]
        reached: Tensor = torch.unique(torch.cat([chain_ids, acyclic_ids]))
        return reached

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
