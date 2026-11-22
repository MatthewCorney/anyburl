"""Link prediction evaluation: MRR and Hits@K metrics."""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import assert_never

import torch

from ._logging import get_logger
from .core import EdgeTypeTuple, EntityScorer, QueryDirection, Triple
from .exceptions import ConfigurationError
from .graph import HeteroGraph

__all__ = [
    "DEFAULT_K_VALUES",
    "EvaluationConfig",
    "LinkPredictionEvaluator",
    "LinkPredictionMetrics",
    "QueryRank",
    "TieHandling",
    "rank_queries_by_edge_type",
    "rank_queries_with_each",
    "rank_with_ties",
    "summarize_ranks",
]

logger = get_logger(__name__)

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 10)
"""Default Hits@K thresholds for link prediction evaluation."""

FILTERED_SCORE: float = float("-inf")
"""Score assigned to known triples during filtered evaluation.

Negative infinity, so filtering works for scorers whose scores are unbounded,
such as a learned aggregator.
"""

TIE_MIDPOINT_FRACTION: float = 0.5
"""Share of a tie group counted as ranking above the target under AVERAGE."""

SOLE_SCORER: str = "scorer"
"""Label of the one scorer :meth:`LinkPredictionEvaluator.rank_queries` ranks."""


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


@dataclass(frozen=True, slots=True)
class QueryRank:
    """The rank of one query's correct answer.

    Parameters
    ----------
    direction : QueryDirection
        Which endpoint was ranked.
    target_id : int
        The correct entity, local to its node type.
    rank : float
        Its 1-based rank, fractional under :attr:`TieHandling.AVERAGE`.
    """

    direction: QueryDirection
    target_id: int
    rank: float


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
        query_ranks = self.rank_queries(test_triples)
        return summarize_ranks(
            [query.rank for query in query_ranks], self._config.k_values
        )

    def rank_queries(self, test_triples: Sequence[Triple]) -> tuple[QueryRank, ...]:
        """Rank the correct answer of every query, without aggregating.

        Parameters
        ----------
        test_triples : Sequence[Triple]
            Test triples, all of one relation. Each yields a tail query
            followed by a head query.

        Returns
        -------
        tuple[QueryRank, ...]
            One rank per query, two per triple.

        Raises
        ------
        ConfigurationError
            If the triples span more than one edge type.
        """
        return rank_queries_with_each(
            {SOLE_SCORER: self._predictor}, self._graph, test_triples, self._config
        )[SOLE_SCORER]


@dataclass(frozen=True, slots=True)
class _Query:
    """One query of a test triple, with the answers filtered from it."""

    direction: QueryDirection
    entity_id: int
    target_id: int
    known: set[int]

    def scores(self, scorer: EntityScorer) -> torch.Tensor:
        """Return ``scorer``'s score for every candidate of this query."""
        match self.direction:
            case QueryDirection.TAIL:
                return scorer.score_tails(self.entity_id)
            case QueryDirection.HEAD:
                return scorer.score_heads(self.entity_id)
            case _ as unreachable:
                assert_never(unreachable)

    def rank(self, scores: torch.Tensor, config: EvaluationConfig) -> float:
        """Rank the target among ``scores``, filtering known answers if configured."""
        if config.filter_known:
            scores = _filter_known(scores, self.target_id, self.known)
        return rank_with_ties(scores, self.target_id, config.tie_handling)


def rank_queries_with_each(
    scorers: Mapping[str, EntityScorer],
    known: HeteroGraph,
    test_triples: Sequence[Triple],
    config: EvaluationConfig,
) -> dict[str, tuple[QueryRank, ...]]:
    """Rank every query of one relation's triples under each of several scorers.

    Each query is scored by every scorer before the next query, and queries
    are visited grouped by direction and entity. Scorers sharing a
    :class:`~anyburl.prediction.RuleEvidence` then ground each query once.

    Parameters
    ----------
    scorers : Mapping[str, EntityScorer]
        The scorers, by label.
    known : HeteroGraph
        The known triples filtered from each query's candidates.
    test_triples : Sequence[Triple]
        Test triples, all of one relation. Each yields a tail query
        followed by a head query.
    config : EvaluationConfig
        Evaluation configuration.

    Returns
    -------
    dict[str, tuple[QueryRank, ...]]
        Each scorer's ranks, two per triple, in triple order.

    Raises
    ------
    ConfigurationError
        If the triples span more than one edge type.
    """
    queries = _queries_of(known, test_triples)
    visit_order = sorted(
        range(len(queries)),
        key=lambda index: (queries[index].direction, queries[index].entity_id),
    )
    ranks = {label: [0.0] * len(queries) for label in scorers}
    for index in visit_order:
        query = queries[index]
        for label, scorer in scorers.items():
            ranks[label][index] = query.rank(query.scores(scorer), config)
    return {
        label: tuple(
            QueryRank(query.direction, query.target_id, rank)
            for query, rank in zip(queries, label_ranks, strict=True)
        )
        for label, label_ranks in ranks.items()
    }


def _queries_of(known: HeteroGraph, test_triples: Sequence[Triple]) -> list[_Query]:
    """Return each triple's tail query then head query."""
    if not test_triples:
        return []
    edge_types = {triple.edge_type for triple in test_triples}
    if len(edge_types) > 1:
        raise ConfigurationError(
            f"test triples must share one edge type, got {sorted(edge_types)}"
        )
    head_to_tails, tail_to_heads = _adjacency_index(known, edge_types.pop())
    queries: list[_Query] = []
    for triple in test_triples:
        queries.append(
            _Query(
                QueryDirection.TAIL,
                triple.head_id,
                triple.tail_id,
                head_to_tails.get(triple.head_id, set()),
            )
        )
        queries.append(
            _Query(
                QueryDirection.HEAD,
                triple.tail_id,
                triple.head_id,
                tail_to_heads.get(triple.tail_id, set()),
            )
        )
    return queries


def _adjacency_index(
    graph: HeteroGraph, edge_type: EdgeTypeTuple
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Build ``(head_to_tails, tail_to_heads)`` maps of known triples."""
    ei = graph.edge_index(edge_type)
    head_to_tails: dict[int, set[int]] = defaultdict(set)
    tail_to_heads: dict[int, set[int]] = defaultdict(set)
    for head, tail in zip(ei[0].tolist(), ei[1].tolist(), strict=True):
        head_to_tails[head].add(tail)
        tail_to_heads[tail].add(head)
    return dict(head_to_tails), dict(tail_to_heads)


def rank_queries_by_edge_type(
    scorer_for: Callable[[EdgeTypeTuple], EntityScorer],
    known: HeteroGraph,
    test_triples: Sequence[Triple],
    config: EvaluationConfig,
) -> tuple[QueryRank, ...]:
    """Rank test triples of any number of relations, each with its own scorer.

    Parameters
    ----------
    scorer_for : Callable[[EdgeTypeTuple], EntityScorer]
        Returns the scorer for a relation.
    known : HeteroGraph
        The known triples filtered from each query's candidates.
    test_triples : Sequence[Triple]
        Test triples of one or more relations.
    config : EvaluationConfig
        Evaluation configuration.

    Returns
    -------
    tuple[QueryRank, ...]
        Two ranks per triple, grouped by relation in order of first
        appearance and in input order within a relation.
    """
    by_edge_type: dict[EdgeTypeTuple, list[Triple]] = defaultdict(list)
    for triple in test_triples:
        by_edge_type[triple.edge_type].append(triple)
    return tuple(
        query
        for edge_type, triples in by_edge_type.items()
        for query in LinkPredictionEvaluator(
            scorer_for(edge_type), known, config
        ).rank_queries(triples)
    )


def summarize_ranks(
    ranks: Sequence[float], k_values: tuple[int, ...]
) -> LinkPredictionMetrics:
    """Aggregate 1-based ranks into MRR and Hits@K.

    Parameters
    ----------
    ranks : Sequence[float]
        One rank per query.
    k_values : tuple[int, ...]
        Hits@K thresholds.

    Returns
    -------
    LinkPredictionMetrics
        The metrics, all zero when ``ranks`` is empty.
    """
    num_queries = len(ranks)
    if num_queries == 0:
        return LinkPredictionMetrics(mrr=0.0, hits_at_k=dict.fromkeys(k_values, 0.0))
    return LinkPredictionMetrics(
        mrr=sum(1.0 / rank for rank in ranks) / num_queries,
        hits_at_k={
            k: sum(1 for rank in ranks if rank <= k) / num_queries for k in k_values
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
