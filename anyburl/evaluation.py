"""Link prediction evaluation: MRR and Hits@K metrics."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, assert_never

import torch

from ._logging import get_logger
from .graph import EdgeTypeTuple, HeteroGraph
from .sampler import Triple

logger = get_logger(__name__)

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 10)
"""Default Hits@K thresholds for link prediction evaluation."""

FILTERED_SCORE: float = -1.0
"""Score assigned to known triples during filtered evaluation."""

TIE_MIDPOINT_FRACTION: float = 0.5
"""Share of a tie group counted as ranking above the target under AVERAGE."""


class TieHandling(StrEnum):
    """How to rank a target that scores equal to other candidates.

    Rule-based scores are coarse and ties are common, so this choice can
    move the reported metrics substantially.

    Attributes
    ----------
    OPTIMISTIC : str
        Count only strictly better candidates, placing the target at the
        top of its tie group. Inflates MRR when ties are large.
    AVERAGE : str
        Place the target at the midpoint of its tie group. The standard
        choice in the link prediction literature, and the default.
    PESSIMISTIC : str
        Place the target at the bottom of its tie group.
    """

    OPTIMISTIC = "optimistic"
    AVERAGE = "average"
    PESSIMISTIC = "pessimistic"


def rank_with_ties(
    scores: torch.Tensor,
    target_index: int,
    tie_handling: TieHandling = TieHandling.AVERAGE,
) -> float:
    """Return the 1-based rank of ``target_index`` under a tie policy.

    Parameters
    ----------
    scores : Tensor
        1-D score tensor over all candidates.
    target_index : int
        Index of the entity being ranked.
    tie_handling : TieHandling
        Policy for candidates scoring exactly equal to the target.

    Returns
    -------
    float
        The rank, fractional under :attr:`TieHandling.AVERAGE`.
    """
    target_score = float(scores[target_index].item())
    num_better = int((scores > target_score).sum().item())
    num_tied = int((scores == target_score).sum().item()) - 1

    match tie_handling:
        case TieHandling.OPTIMISTIC:
            return float(num_better + 1)
        case TieHandling.AVERAGE:
            return num_better + 1 + num_tied * TIE_MIDPOINT_FRACTION
        case TieHandling.PESSIMISTIC:
            return float(num_better + num_tied + 1)
        case _ as unreachable:
            assert_never(unreachable)


class EntityScorer(Protocol):
    """Scores candidate entities for a link prediction query.

    Satisfied by :class:`~anyburl.prediction.RulePredictor` and by the
    scorers in :mod:`anyburl.baselines`.
    """

    def score_tails(self, head_id: int) -> torch.Tensor:
        """Return scores over all candidate tails for ``head_id``."""
        ...

    def score_heads(self, tail_id: int) -> torch.Tensor:
        """Return scores over all candidate heads for ``tail_id``."""
        ...


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
    tie_handling : TieHandling
        How to rank a target tied with other candidates.
    """

    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    filter_known: bool = True
    tie_handling: TieHandling = TieHandling.AVERAGE


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
    predictor : EntityScorer
        The scorer being evaluated.
    graph : HeteroGraph
        The knowledge graph (used for filtering known triples).
    config : EvaluationConfig
        Evaluation configuration.
    """

    def __init__(
        self,
        predictor: EntityScorer,
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

        first = test_triples[0]
        head_to_tails, tail_to_heads = self._build_adjacency_index(
            (first.head_type, first.relation, first.tail_type)
        )

        ranks: list[float] = []
        for triple in test_triples:
            ranks.append(
                self._rank(
                    self._predictor.score_tails(triple.head_id),
                    triple.tail_id,
                    head_to_tails.get(triple.head_id, set()),
                )
            )
            ranks.append(
                self._rank(
                    self._predictor.score_heads(triple.tail_id),
                    triple.head_id,
                    tail_to_heads.get(triple.tail_id, set()),
                )
            )

        return self._aggregate_ranks(ranks)

    def _rank(self, scores: torch.Tensor, target: int, known: set[int]) -> float:
        """Rank ``target`` among ``scores``, filtering ``known`` if configured.

        Parameters
        ----------
        scores : Tensor
            1-D score tensor over all candidates.
        target : int
            The correct candidate.
        known : set[int]
            Candidates already known to be true for this query.

        Returns
        -------
        float
            The 1-based rank of ``target``.
        """
        if self._config.filter_known:
            scores = _filter_known(scores, target, known)
        return rank_with_ties(scores, target, self._config.tie_handling)

    def _build_adjacency_index(
        self,
        edge_type: EdgeTypeTuple,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        """Build ``(head_to_tails, tail_to_heads)`` maps of known triples."""
        ei = self._graph.edge_index(edge_type)
        head_to_tails: dict[int, set[int]] = defaultdict(set)
        tail_to_heads: dict[int, set[int]] = defaultdict(set)
        for head, tail in zip(ei[0].tolist(), ei[1].tolist(), strict=True):
            head_to_tails[head].add(tail)
            tail_to_heads[tail].add(head)
        return dict(head_to_tails), dict(tail_to_heads)

    def _aggregate_ranks(self, ranks: list[float]) -> LinkPredictionMetrics:
        """Aggregate 1-based ranks into MRR and Hits@K metrics."""
        num_queries = len(ranks)
        return LinkPredictionMetrics(
            mrr=sum(1.0 / rank for rank in ranks) / num_queries,
            hits_at_k={
                k: sum(1 for rank in ranks if rank <= k) / num_queries
                for k in self._config.k_values
            },
            num_queries=num_queries,
        )


def _filter_known(scores: torch.Tensor, target: int, known: set[int]) -> torch.Tensor:
    """Return a copy of ``scores`` with known candidates other than ``target`` sunk.

    Parameters
    ----------
    scores : Tensor
        1-D score tensor over all candidates.
    target : int
        The candidate being ranked, which keeps its score.
    known : set[int]
        Candidates already known to be true for this query.

    Returns
    -------
    Tensor
        Filtered scores.
    """
    filtered = scores.clone()
    others = [candidate for candidate in known if candidate != target]
    if others:
        filtered[others] = FILTERED_SCORE
    return filtered
