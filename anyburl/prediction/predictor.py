"""Apply learned rules to score candidate entities."""

from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import Tensor
from tqdm import tqdm

from .._logging import get_logger
from ..exceptions import ConfigurationError
from ..graph import EdgeTypeTuple, HeteroGraph
from ..metrics import RuleMetrics
from ..rule import Rule
from .grounding import GroundingMode, build_groundings
from .records import Prediction, RuleFiring
from .scoring import ScoringStrategy, weight_for_candidate
from .sides import ChainSide, apply_side, build_chain_sides

__all__ = ["DEFAULT_TOP_TAILS", "RulePredictor"]

logger = get_logger(__name__)

DEFAULT_TOP_TAILS: int = 10
"""Number of tails :meth:`RulePredictor.top_tails` returns by default."""


class RulePredictor:
    """Grounds rules against a graph and aggregates their predictions.

    Rules sharing a body chain share one grounding. A candidate's score is
    the noisy-or of the confidences of every rule reaching it, weighted by
    the configured :class:`ScoringStrategy`. AC2 rules are ignored.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to ground rules against.
    results : Sequence[tuple[Rule, RuleMetrics]]
        Rules paired with their evaluated metrics. All rules must
        predict the same head relation.
    scoring_strategy : ScoringStrategy
        How rule confidences are weighted against a candidate.
    grounding_mode : GroundingMode
        Whether to materialise chain products up front, ground each chain
        per query, or decide per chain by size.

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
        scoring_strategy: ScoringStrategy = ScoringStrategy.PATH_WEIGHTED,
        grounding_mode: GroundingMode = GroundingMode.AUTO,
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
        self._scoring_strategy = scoring_strategy
        self._head_edge_type: EdgeTypeTuple = head_relations.pop()
        head_type, _, tail_type = self._head_edge_type
        self._num_heads = graph.node_count(head_type)
        self._num_tails = graph.node_count(tail_type)
        self._tail_sides, self._head_sides = build_chain_sides(
            results, build_groundings(graph, results, grounding_mode)
        )

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
        return self._score(self._tail_sides, head_id, self._num_tails)

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
        return self._score(self._head_sides, tail_id, self._num_heads)

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
        self, sides: Sequence[ChainSide], query_id: int, num_candidates: int
    ) -> Tensor:
        """Return ``1 - prod(1 - contribution)`` over every chain side."""
        complement = torch.ones(num_candidates)
        for side in sides:
            apply_side(complement, side, query_id, self._scoring_strategy)
        return 1.0 - complement

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
