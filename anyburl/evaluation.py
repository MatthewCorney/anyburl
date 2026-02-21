"""Link prediction evaluation: MRR and Hits@K metrics."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ._logging import get_logger
from .graph import HeteroGraph
from .prediction import RulePredictor
from .sampler import Triple

logger = get_logger(__name__)

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 10)
"""Default Hits@K thresholds for link prediction evaluation."""

FILTERED_SCORE: float = -1.0
"""Score assigned to known triples during filtered evaluation."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration for link prediction evaluation.

    Parameters
    ----------
    k_values : tuple[int, ...]
        Hits@K thresholds to compute.
    filter_known : bool
        If ``True``, filter out known triples when computing ranks
        (the standard "filtered" setting).
    """

    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    filter_known: bool = True


@dataclass(frozen=True, slots=True)
class LinkPredictionMetrics:
    """Aggregated link prediction evaluation results.

    Parameters
    ----------
    mrr : float
        Mean Reciprocal Rank across all queries.
    hits_at_k : dict[int, float]
        Hits@K for each threshold in the config.
    num_queries : int
        Total number of queries evaluated (2 per triple:
        tail prediction + head prediction).
    """

    mrr: float
    hits_at_k: dict[int, float] = field(default_factory=dict)
    num_queries: int = 0


class LinkPredictionEvaluator:
    """Evaluates link prediction quality using MRR and Hits@K.

    For each test triple ``(h, r, t)``:

    - **Tail prediction**: score all candidate tails given ``h``,
      compute the filtered rank of ``t``.
    - **Head prediction**: score all candidate heads given ``t``,
      compute the filtered rank of ``h``.

    Filtered rank excludes known triples (except the query target)
    so that a model is not penalized for predicting valid triples.

    Parameters
    ----------
    predictor : RulePredictor
        A pre-built predictor with cached chain products.
    graph : HeteroGraph
        The knowledge graph (used for filtering known triples).
    config : EvaluationConfig
        Evaluation configuration.
    """

    def __init__(
        self,
        predictor: RulePredictor,
        graph: HeteroGraph,
        config: EvaluationConfig,
    ) -> None:
        self._predictor = predictor
        self._graph = graph
        self._config = config

    def evaluate(
        self,
        test_triples: Sequence[Triple],
    ) -> LinkPredictionMetrics:
        """Evaluate link prediction on a set of test triples.

        Parameters
        ----------
        test_triples : Sequence[Triple]
            Test triples to evaluate. Each triple contributes two
            queries (tail prediction and head prediction).

        Returns
        -------
        LinkPredictionMetrics
            Aggregated MRR and Hits@K metrics.
        """
        if not test_triples:
            return LinkPredictionMetrics(
                mrr=0.0,
                hits_at_k=dict.fromkeys(self._config.k_values, 0.0),
                num_queries=0,
            )

        head_edge_type = (
            test_triples[0].head_type,
            test_triples[0].relation,
            test_triples[0].tail_type,
        )

        head_to_tails, tail_to_heads = self._build_adjacency_index(head_edge_type)

        ranks: list[int] = []

        for triple in test_triples:
            tail_rank = self._compute_tail_rank(
                triple.head_id, triple.tail_id, head_to_tails
            )
            ranks.append(tail_rank)

            head_rank = self._compute_head_rank(
                triple.head_id, triple.tail_id, tail_to_heads
            )
            ranks.append(head_rank)

        return self._aggregate_ranks(ranks)

    def _compute_tail_rank(
        self,
        head_id: int,
        true_tail_id: int,
        head_to_tails: dict[int, set[int]],
    ) -> int:
        """Compute the filtered rank of the true tail entity.

        Parameters
        ----------
        head_id : int
            The source entity.
        true_tail_id : int
            The correct tail entity.
        head_to_tails : dict[int, set[int]]
            Known tails per head for filtering.

        Returns
        -------
        int
            The rank of the true tail (1-based).
        """
        scores = self._predictor.score_tails(head_id)

        if self._config.filter_known:
            scores = self._filter_tail_scores(
                scores, true_tail_id, head_to_tails.get(head_id, set())
            )

        true_score = float(scores[true_tail_id].item())
        return int((scores > true_score).sum().item()) + 1

    def _compute_head_rank(
        self,
        true_head_id: int,
        tail_id: int,
        tail_to_heads: dict[int, set[int]],
    ) -> int:
        """Compute the filtered rank of the true head entity.

        Parameters
        ----------
        true_head_id : int
            The correct head entity.
        tail_id : int
            The destination entity.
        tail_to_heads : dict[int, set[int]]
            Known heads per tail for filtering.

        Returns
        -------
        int
            The rank of the true head (1-based).
        """
        scores = self._predictor.score_heads(tail_id)

        if self._config.filter_known:
            scores = self._filter_head_scores(
                scores, true_head_id, tail_to_heads.get(tail_id, set())
            )

        true_score = float(scores[true_head_id].item())
        return int((scores > true_score).sum().item()) + 1

    @staticmethod
    def _filter_tail_scores(
        scores: torch.Tensor,
        true_tail_id: int,
        known_tails: set[int],
    ) -> torch.Tensor:
        """Set scores of known tails (except the target) to -1.

        Parameters
        ----------
        scores : Tensor
            1-D score tensor for all tails.
        true_tail_id : int
            The target tail (preserved).
        known_tails : set[int]
            Known tail IDs for the query head.

        Returns
        -------
        Tensor
            Filtered scores (clone of input).
        """
        filtered = scores.clone()
        for t in known_tails:
            if t != true_tail_id:
                filtered[t] = FILTERED_SCORE
        return filtered

    @staticmethod
    def _filter_head_scores(
        scores: torch.Tensor,
        true_head_id: int,
        known_heads: set[int],
    ) -> torch.Tensor:
        """Set scores of known heads (except the target) to -1.

        Parameters
        ----------
        scores : Tensor
            1-D score tensor for all heads.
        true_head_id : int
            The target head (preserved).
        known_heads : set[int]
            Known head IDs for the query tail.

        Returns
        -------
        Tensor
            Filtered scores (clone of input).
        """
        filtered = scores.clone()
        for h in known_heads:
            if h != true_head_id:
                filtered[h] = FILTERED_SCORE
        return filtered

    def _build_adjacency_index(
        self,
        edge_type: tuple[str, str, str],
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        """Build adjacency indices for fast per-entity filtering.

        Parameters
        ----------
        edge_type : tuple[str, str, str]
            The edge type to extract pairs from.

        Returns
        -------
        tuple[dict[int, set[int]], dict[int, set[int]]]
            ``(head_to_tails, tail_to_heads)`` mappings.
        """
        ei = self._graph.edge_index(edge_type)
        sources = ei[0].tolist()
        destinations = ei[1].tolist()

        head_to_tails: dict[int, set[int]] = defaultdict(set)
        tail_to_heads: dict[int, set[int]] = defaultdict(set)
        for h, t in zip(sources, destinations, strict=True):
            head_to_tails[h].add(t)
            tail_to_heads[t].add(h)

        return dict(head_to_tails), dict(tail_to_heads)

    def _aggregate_ranks(self, ranks: list[int]) -> LinkPredictionMetrics:
        """Aggregate ranks into MRR and Hits@K metrics.

        Parameters
        ----------
        ranks : list[int]
            List of 1-based ranks.

        Returns
        -------
        LinkPredictionMetrics
            Aggregated metrics.
        """
        num_queries = len(ranks)
        mrr = sum(1.0 / r for r in ranks) / num_queries

        hits_at_k: dict[int, float] = {}
        for k in self._config.k_values:
            hits = sum(1 for r in ranks if r <= k)
            hits_at_k[k] = hits / num_queries

        return LinkPredictionMetrics(
            mrr=mrr,
            hits_at_k=hits_at_k,
            num_queries=num_queries,
        )
