"""Tests for degree-stratified link prediction metrics."""

import pytest
import torch

from anyburl.baselines import PopularityScorer
from anyburl.evaluation import (
    EvaluationConfig,
    LinkPredictionEvaluator,
    QueryDirection,
    QueryRank,
    summarize_ranks,
)
from anyburl.exceptions import ConfigurationError
from anyburl.graph import HeteroGraph
from anyburl.sampler import Triple
from anyburl.stratification import (
    RelationDegrees,
    StratificationConfig,
    relation_degrees,
    stratify_by_degree,
)

TARGET = ("person", "lives_in", "city")
K_VALUES = (1, 3)


def _tail_queries(*pairs: tuple[int, float]) -> list[QueryRank]:
    return [QueryRank(QueryDirection.TAIL, target, rank) for target, rank in pairs]


def _degrees(tail: list[float]) -> RelationDegrees:
    return RelationDegrees(head=torch.zeros(1), tail=torch.tensor(tail))


class TestRelationDegrees:
    def test_counts_head_and_tail_edges(self, evaluator_graph: HeteroGraph) -> None:
        """lives_in: 0->0, 1->1, 2->2, 3->0."""
        degrees = relation_degrees(evaluator_graph, TARGET)

        assert degrees.head.tolist() == [1.0, 1.0, 1.0, 1.0]
        assert degrees.tail.tolist() == [2.0, 1.0, 1.0]

    def test_looks_up_the_ranked_endpoint(self) -> None:
        degrees = RelationDegrees(head=torch.tensor([5.0]), tail=torch.tensor([7.0]))

        assert degrees.of_target(QueryRank(QueryDirection.HEAD, 0, 1.0)) == 5
        assert degrees.of_target(QueryRank(QueryDirection.TAIL, 0, 1.0)) == 7


class TestSummarizeRanks:
    def test_computes_mrr_and_hits(self) -> None:
        metrics = summarize_ranks([1.0, 2.0, 4.0], K_VALUES)

        assert metrics.mrr == pytest.approx((1 + 0.5 + 0.25) / 3)
        assert metrics.hits_at_k == {1: pytest.approx(1 / 3), 3: pytest.approx(2 / 3)}
        assert metrics.num_queries == 3

    def test_empty_ranks_give_zeros(self) -> None:
        metrics = summarize_ranks([], K_VALUES)

        assert metrics.mrr == 0.0
        assert metrics.num_queries == 0
        assert metrics.hits_at_k == {1: 0.0, 3: 0.0}


class TestRankQueries:
    def test_matches_aggregate_evaluate(self, evaluator_graph: HeteroGraph) -> None:
        evaluator = LinkPredictionEvaluator(
            PopularityScorer(evaluator_graph, TARGET),
            evaluator_graph,
            EvaluationConfig(k_values=K_VALUES),
        )
        triples = [
            Triple(0, 0, "person", "city", "lives_in"),
            Triple(1, 1, "person", "city", "lives_in"),
        ]

        query_ranks = evaluator.rank_queries(triples)

        assert [query.direction for query in query_ranks] == [
            QueryDirection.TAIL,
            QueryDirection.HEAD,
        ] * 2
        assert [query.target_id for query in query_ranks] == [0, 0, 1, 1]
        assert summarize_ranks(
            [query.rank for query in query_ranks], K_VALUES
        ) == evaluator.evaluate(triples)

    def test_empty_triples_give_no_ranks(self, evaluator_graph: HeteroGraph) -> None:
        evaluator = LinkPredictionEvaluator(
            PopularityScorer(evaluator_graph, TARGET),
            evaluator_graph,
            EvaluationConfig(),
        )

        assert evaluator.rank_queries([]) == ()


class TestStratifyByDegree:
    def test_splits_into_equal_frequency_buckets(self) -> None:
        degrees = _degrees([1.0, 2.0, 3.0, 4.0])
        queries = _tail_queries((0, 1.0), (1, 2.0), (2, 4.0), (3, 10.0))

        buckets = stratify_by_degree(
            queries, degrees, StratificationConfig(num_buckets=2, k_values=K_VALUES)
        )

        assert [(b.min_degree, b.max_degree) for b in buckets] == [(1, 2), (3, 4)]
        assert buckets[0].metrics.mrr == pytest.approx(0.75)
        assert buckets[1].metrics.mrr == pytest.approx(0.175)

    def test_never_splits_one_degree_across_buckets(self) -> None:
        degrees = _degrees([3.0, 3.0, 3.0, 9.0])
        queries = _tail_queries((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0))

        buckets = stratify_by_degree(
            queries, degrees, StratificationConfig(num_buckets=4)
        )

        assert [(b.min_degree, b.max_degree) for b in buckets] == [(3, 3), (9, 9)]
        assert [b.metrics.num_queries for b in buckets] == [3, 1]

    def test_unseen_targets_get_their_own_bucket(self) -> None:
        degrees = _degrees([0.0, 1.0, 2.0, 5.0])
        queries = _tail_queries((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0))

        buckets = stratify_by_degree(
            queries, degrees, StratificationConfig(num_buckets=2)
        )

        assert buckets[0].min_degree == buckets[0].max_degree == 0

    def test_buckets_each_direction_separately(self) -> None:
        degrees = RelationDegrees(head=torch.tensor([4.0]), tail=torch.tensor([1.0]))
        queries = [
            QueryRank(QueryDirection.TAIL, 0, 1.0),
            QueryRank(QueryDirection.HEAD, 0, 2.0),
        ]

        buckets = stratify_by_degree(
            queries, degrees, StratificationConfig(num_buckets=1)
        )

        assert [(b.direction, b.max_degree) for b in buckets] == [
            (QueryDirection.TAIL, 1),
            (QueryDirection.HEAD, 4),
        ]

    def test_bucket_counts_sum_to_all_queries(self) -> None:
        degrees = _degrees([float(d) for d in range(10)])
        queries = _tail_queries(*((target, 1.0 + target) for target in range(10)))

        buckets = stratify_by_degree(
            queries, degrees, StratificationConfig(num_buckets=3)
        )

        assert sum(b.metrics.num_queries for b in buckets) == len(queries)

    def test_rejects_zero_buckets(self) -> None:
        with pytest.raises(ConfigurationError):
            StratificationConfig(num_buckets=0)
