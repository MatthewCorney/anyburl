"""Apply learned rules to score candidate entities."""

from collections import defaultdict
from collections.abc import Sequence
from typing import assert_never

import torch
from torch import Tensor
from tqdm import tqdm

from .._logging import get_logger
from ..evaluation import QueryDirection
from ..exceptions import ConfigurationError
from ..graph import EdgeTypeTuple, HeteroGraph
from ..metrics import RuleMetrics
from ..metrics.acyclic import AcyclicIndex
from ..rule import Rule, RuleType
from .acyclic import AcyclicScorer, Firings
from .grounding import BodyChainKey, build_groundings
from .records import (
    AcyclicEvidence,
    ChainEvidence,
    Prediction,
    PredictionConfig,
    RuleFiring,
)
from .scoring import (
    TIE_BREAK_WEIGHT,
    ConfidenceAggregation,
    weight_for_candidate,
)
from .sides import ChainSide, apply_side, build_chain_sides, side_evidence

__all__ = ["DEFAULT_TOP_TAILS", "RulePredictor"]

logger = get_logger(__name__)

DEFAULT_TOP_TAILS: int = 10
"""Number of tails :meth:`RulePredictor.top_tails` returns by default."""


class RulePredictor:
    """Grounds rules against a graph and aggregates their predictions.

    Rules sharing a body chain share one grounding. Each chain's
    contribution is weighted by the configured :class:`ScoringStrategy`, and
    contributions combine per :class:`ConfidenceAggregation`. Acyclic rules
    fire individually and combine the same way, but are left out of
    :meth:`chain_evidence` and :meth:`explain`. AC2 rules are ignored.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to ground rules against.
    results : Sequence[tuple[Rule, RuleMetrics]]
        Rules paired with their evaluated metrics. All rules must
        predict the same head relation.
    config : PredictionConfig | None
        Scoring, aggregation and grounding settings. ``None`` uses the
        defaults.

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

        self._graph = graph
        self._config = config if config is not None else PredictionConfig()
        self._scoring_strategy = self._config.scoring_strategy
        self._head_edge_type: EdgeTypeTuple = head_relations.pop()
        head_type, _, tail_type = self._head_edge_type
        self._num_heads = graph.node_count(head_type)
        self._num_tails = graph.node_count(tail_type)
        self._tail_sides, self._head_sides = build_chain_sides(
            results,
            build_groundings(graph, results, self._config.grounding_mode),
            self._config.unseen_negatives,
        )
        self._acyclic = AcyclicScorer(
            AcyclicIndex(graph), results, self._config.unseen_negatives
        )

    @property
    def chains(self) -> tuple[tuple[BodyChainKey, RuleType], ...]:
        """The body chain and rule type of each :meth:`chain_evidence` row."""
        return tuple((side.chain, side.rule_type) for side in self._tail_sides)

    def predict(
        self, *, filter_known: bool = False, min_score: float = 0.0
    ) -> list[Prediction]:
        """Score every head entity's candidate tails.

        Parameters
        ----------
        filter_known : bool
            If ``True``, exclude pairs already present in the graph.
        min_score : float
            Pairs scoring at or below this value are excluded.

        Returns
        -------
        list[Prediction]
            Predictions sorted by score descending.
        """
        known_tails = self._known_tails() if filter_known else {}
        predictions: list[Prediction] = []
        for head_id in tqdm(range(self._num_heads), desc="Predicting"):
            scores = self.score_tails(head_id)
            known = known_tails.get(head_id, set())
            for tail_id in torch.nonzero(scores > min_score, as_tuple=True)[0].tolist():
                if tail_id not in known:
                    predictions.append(
                        self._prediction(head_id, tail_id, float(scores[tail_id]))
                    )
        predictions.sort(key=lambda p: p.score, reverse=True)
        logger.info("Predicted %d pairs", len(predictions))
        return predictions

    def score_tails(self, head_id: int) -> Tensor:
        """Compute noisy-or scores for all candidate tails given a head.

        Parameters
        ----------
        head_id : int
            The source entity index.

        Returns
        -------
        Tensor
            1-D float tensor of shape ``(num_tails,)``.
        """
        return self._score(
            self._tail_sides,
            self._acyclic.tail_firings(head_id),
            head_id,
            self._num_tails,
        )

    def score_heads(self, tail_id: int) -> Tensor:
        """Compute noisy-or scores for all candidate heads given a tail.

        Parameters
        ----------
        tail_id : int
            The destination entity index.

        Returns
        -------
        Tensor
            1-D float tensor of shape ``(num_heads,)``.
        """
        return self._score(
            self._head_sides,
            self._acyclic.head_firings(tail_id),
            tail_id,
            self._num_heads,
        )

    def chain_evidence(self, direction: QueryDirection, query_id: int) -> ChainEvidence:
        """Break a query's score down into each body chain's contribution.

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
        match direction:
            case QueryDirection.TAIL:
                sides, num_candidates = self._tail_sides, self._num_tails
            case QueryDirection.HEAD:
                sides, num_candidates = self._head_sides, self._num_heads
            case _ as unreachable:
                assert_never(unreachable)

        rows = [
            side_evidence(side, query_id, num_candidates, self._scoring_strategy)
            for side in sides
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
        match direction:
            case QueryDirection.TAIL:
                firings = self._acyclic.tail_firings(query_id)
                num_candidates = self._num_tails
            case QueryDirection.HEAD:
                firings = self._acyclic.head_firings(query_id)
                num_candidates = self._num_heads
            case _ as unreachable:
                assert_never(unreachable)
        candidates, confidences = firings
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

    def explain(self, head_id: int, tail_id: int) -> tuple[RuleFiring, ...]:
        """Report which rules produce the score for one predicted pair.

        Parameters
        ----------
        head_id : int
            The source entity.
        tail_id : int
            The destination entity.

        Returns
        -------
        tuple[RuleFiring, ...]
            One entry per firing rule group, strongest contribution first.
            Empty when no rule connects the pair.
        """
        firings: list[RuleFiring] = []
        for side in self._tail_sides:
            weight = weight_for_candidate(
                side.fetch(head_id), tail_id, self._scoring_strategy
            )
            if weight is None:
                continue
            groundings, evidence = weight
            for confidence in (
                side.confidence_for(head_id),
                side.candidate_confidences.get(tail_id),
            ):
                if confidence is not None:
                    firings.append(
                        RuleFiring(
                            chain=side.chain,
                            rule_type=side.rule_type,
                            confidence=confidence,
                            groundings=groundings,
                            contribution=1.0 - (1.0 - confidence) ** evidence,
                        )
                    )
        firings.sort(key=lambda firing: firing.contribution, reverse=True)
        return tuple(firings)

    def top_tails(
        self, head_id: int, *, limit: int = DEFAULT_TOP_TAILS
    ) -> list[Prediction]:
        """Return the best-scoring tails for one head, with explanations.

        Parameters
        ----------
        head_id : int
            The source entity.
        limit : int
            Maximum number of predictions to return.

        Returns
        -------
        list[Prediction]
            Predictions sorted by score descending, each carrying the rule
            firings that produced it. Candidates scoring zero are omitted.

        Raises
        ------
        ConfigurationError
            If ``limit`` is not positive.
        """
        if limit < 1:
            raise ConfigurationError(f"limit must be positive, got {limit}")

        scores = self.score_tails(head_id)
        ranked = torch.argsort(scores, descending=True)[:limit].tolist()
        return [
            self._prediction(
                head_id,
                tail_id,
                float(scores[tail_id]),
                explanations=self.explain(head_id, tail_id),
            )
            for tail_id in ranked
            if float(scores[tail_id]) > 0.0
        ]

    def _score(
        self,
        sides: Sequence[ChainSide],
        firings: Firings,
        query_id: int,
        num_candidates: int,
    ) -> Tensor:
        """Combine every chain side's and acyclic rule's contribution."""
        candidates, confidences = firings
        log_complement = torch.zeros(num_candidates)
        log_complement.index_add_(0, candidates, torch.log1p(-confidences))
        complement = torch.exp(log_complement)
        match self._config.aggregation:
            case ConfidenceAggregation.NOISY_OR:
                for side in sides:
                    apply_side(complement, side, query_id, self._scoring_strategy)
                return torch.ones_like(complement) - complement
            case ConfidenceAggregation.MAXIMUM:
                best = torch.zeros(num_candidates)
                best.scatter_reduce_(0, candidates, confidences, "amax")
                for side in sides:
                    side_complement = torch.ones(num_candidates)
                    apply_side(side_complement, side, query_id, self._scoring_strategy)
                    torch.maximum(best, 1.0 - side_complement, out=best)
                    complement *= side_complement
                return best + TIE_BREAK_WEIGHT * (1.0 - complement)
            case _ as unreachable:
                assert_never(unreachable)

    def _prediction(
        self,
        head_id: int,
        tail_id: int,
        score: float,
        *,
        explanations: tuple[RuleFiring, ...] = (),
    ) -> Prediction:
        """Build a :class:`Prediction` for the head relation."""
        head_type, relation, tail_type = self._head_edge_type
        return Prediction(
            head_id=head_id,
            tail_id=tail_id,
            head_type=head_type,
            tail_type=tail_type,
            relation=relation,
            score=score,
            explanations=explanations,
        )

    def _known_tails(self) -> dict[int, set[int]]:
        """Return the known tails of each head under the head relation."""
        edge_index = self._graph.edge_index(self._head_edge_type)
        known: dict[int, set[int]] = defaultdict(set)
        for head, tail in zip(
            edge_index[0].tolist(), edge_index[1].tolist(), strict=True
        ):
            known[head].add(tail)
        return dict(known)
