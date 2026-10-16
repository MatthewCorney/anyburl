"""Apply learned rules to score candidate entities."""

from collections import defaultdict
from collections.abc import Sequence
from typing import assert_never

import torch
from torch import Tensor

from .._logging import get_logger, progress
from ..core import QueryDirection
from ..exceptions import ConfigurationError
from ..graph import HeteroGraph
from ..metrics import RuleMetrics
from ..rule import Rule
from .evidence import RuleEvidence
from .records import Prediction, PredictionConfig, RuleFiring
from .scoring import TIE_BREAK_WEIGHT, ConfidenceAggregation, weight_for_candidate
from .sides import apply_side

__all__ = ["DEFAULT_TOP_TAILS", "RulePredictor", "combine_evidence"]

logger = get_logger(__name__)

DEFAULT_TOP_TAILS: int = 10
"""Number of tails :meth:`RulePredictor.top_tails` returns by default."""


class RulePredictor:
    """Scores candidates by combining rule evidence with a fixed aggregation.

    Contributions are weighted by the configured :class:`GroundingWeighting`
    and combined per :class:`ConfidenceAggregation`. Acyclic rules fire
    individually and combine the same way, but are left out of
    :meth:`explain`. AC2 rules are ignored.

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
        self._graph = graph
        self._config = config if config is not None else PredictionConfig()
        self._evidence = RuleEvidence(graph, results, config=self._config)

    @property
    def evidence(self) -> RuleEvidence:
        """Return the rule evidence this predictor scores from."""
        return self._evidence

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
        num_heads = self._evidence.num_candidates(QueryDirection.HEAD)
        for head_id in progress(range(num_heads), desc="Predicting"):
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
        return combine_evidence(
            self._evidence, QueryDirection.TAIL, head_id, self._config.aggregation
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
        return combine_evidence(
            self._evidence, QueryDirection.HEAD, tail_id, self._config.aggregation
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
        for side in self._evidence.sides(QueryDirection.TAIL):
            weight = weight_for_candidate(
                side.fetch(head_id), tail_id, self._evidence.weighting
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

    def _prediction(
        self,
        head_id: int,
        tail_id: int,
        score: float,
        *,
        explanations: tuple[RuleFiring, ...] = (),
    ) -> Prediction:
        """Build a :class:`Prediction` for the head relation."""
        head_type, relation, tail_type = self._evidence.head_edge_type
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
        edge_index = self._graph.edge_index(self._evidence.head_edge_type)
        known: dict[int, set[int]] = defaultdict(set)
        for head, tail in zip(
            edge_index[0].tolist(), edge_index[1].tolist(), strict=True
        ):
            known[head].add(tail)
        return dict(known)


def combine_evidence(
    evidence: RuleEvidence,
    direction: QueryDirection,
    query_id: int,
    aggregation: ConfidenceAggregation,
) -> Tensor:
    """Combine every chain's and acyclic rule's contribution into scores.

    Parameters
    ----------
    evidence : RuleEvidence
        The rules' evidence.
    direction : QueryDirection
        ``TAIL`` to score tails of head ``query_id``, ``HEAD`` to score
        heads of tail ``query_id``.
    query_id : int
        The entity being queried.
    aggregation : ConfidenceAggregation
        How contributions combine.

    Returns
    -------
    Tensor
        One score per candidate.
    """
    sides = evidence.sides(direction)
    candidates, confidences = evidence.acyclic_firings(direction, query_id)
    num_candidates = evidence.num_candidates(direction)
    weighting = evidence.weighting
    log_complement = torch.zeros(num_candidates)
    log_complement.index_add_(0, candidates, torch.log1p(-confidences))
    complement = torch.exp(log_complement)
    match aggregation:
        case ConfidenceAggregation.NOISY_OR:
            for side in sides:
                apply_side(complement, side, query_id, weighting)
            return torch.ones_like(complement) - complement
        case ConfidenceAggregation.MAXIMUM:
            best = torch.zeros(num_candidates)
            best.scatter_reduce_(0, candidates, confidences, "amax")
            for side in sides:
                side_complement = torch.ones(num_candidates)
                apply_side(side_complement, side, query_id, weighting)
                torch.maximum(best, 1.0 - side_complement, out=best)
                complement *= side_complement
            return best + TIE_BREAK_WEIGHT * (1.0 - complement)
        case _ as unreachable:
            assert_never(unreachable)
