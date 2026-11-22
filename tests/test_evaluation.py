"""Tests for LinkPredictionEvaluator."""

import pytest
import torch

from anyburl.core import Triple
from anyburl.evaluation import (
    EvaluationConfig,
    LinkPredictionEvaluator,
    TieHandling,
    rank_queries_with_each,
    rank_with_ties,
)
from anyburl.exceptions import ConfigurationError
from anyburl.graph import HeteroGraph
from anyburl.prediction import RulePredictor
from anyburl.rule import Rule
from anyburl.rule_evaluation import RuleSet
from tests.rules import metrics_with


def test_perfect_mrr(
    perfect_eval_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """With a perfect rule, all triples should rank 1 -> MRR = 1.0."""
    results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.9))])
    predictor = RulePredictor(perfect_eval_graph, results)

    config = EvaluationConfig(k_values=(1, 3, 10), filter_known=True)
    evaluator = LinkPredictionEvaluator(predictor, perfect_eval_graph, config)

    test_triples = [
        Triple(
            head_id=0,
            tail_id=1,
            head_type="person",
            tail_type="city",
            relation="lives_in",
        ),
    ]

    metrics = evaluator.evaluate(test_triples)

    assert metrics.mrr == pytest.approx(1.0)
    assert metrics.hits_at_k[1] == pytest.approx(1.0)
    assert metrics.hits_at_k[3] == pytest.approx(1.0)
    assert metrics.num_queries == 2


def test_perfect_hits_at_k(
    perfect_eval_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """Perfect prediction should give Hits@K = 1.0 for all K."""
    results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.9))])
    predictor = RulePredictor(perfect_eval_graph, results)

    config = EvaluationConfig(k_values=(1, 3), filter_known=True)
    evaluator = LinkPredictionEvaluator(predictor, perfect_eval_graph, config)

    test_triples = [
        Triple(
            head_id=0,
            tail_id=1,
            head_type="person",
            tail_type="city",
            relation="lives_in",
        ),
        Triple(
            head_id=1,
            tail_id=2,
            head_type="person",
            tail_type="city",
            relation="lives_in",
        ),
    ]

    metrics = evaluator.evaluate(test_triples)

    assert metrics.hits_at_k[1] == pytest.approx(1.0)
    assert metrics.num_queries == 4


def test_filtered_excludes_known_triples(
    perfect_eval_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """Filtered setting should not count known triples as competitors."""
    results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.9))])
    predictor = RulePredictor(perfect_eval_graph, results)

    filtered_config = EvaluationConfig(k_values=(1,), filter_known=True)
    filtered_eval = LinkPredictionEvaluator(
        predictor, perfect_eval_graph, filtered_config
    )

    raw_config = EvaluationConfig(k_values=(1,), filter_known=False)
    raw_eval = LinkPredictionEvaluator(predictor, perfect_eval_graph, raw_config)

    test_triples = [
        Triple(
            head_id=0,
            tail_id=1,
            head_type="person",
            tail_type="city",
            relation="lives_in",
        ),
    ]

    filtered_metrics = filtered_eval.evaluate(test_triples)
    raw_metrics = raw_eval.evaluate(test_triples)

    assert filtered_metrics.mrr >= raw_metrics.mrr


class _NegativeScorer:
    """Scores every query ``[-10.0, -0.5, -5.0]``, all below zero."""

    def score_tails(self, head_id: int) -> torch.Tensor:
        return torch.tensor([-10.0, -0.5, -5.0])

    def score_heads(self, tail_id: int) -> torch.Tensor:
        return torch.tensor([-10.0, -0.5, -5.0])


def test_filtering_sinks_known_answers_below_negative_scores(
    perfect_eval_graph: HeteroGraph,
) -> None:
    """A filtered known answer must not outrank a target scoring below zero."""
    evaluator = LinkPredictionEvaluator(
        _NegativeScorer(), perfect_eval_graph, EvaluationConfig(filter_known=True)
    )
    test_triple = Triple(
        head_id=0, tail_id=2, head_type="person", tail_type="city", relation="lives_in"
    )

    tail_query, _ = evaluator.rank_queries([test_triple])

    assert tail_query.rank == 1.0


def test_averaged_metrics(
    perfect_eval_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """Metrics should be averaged across all queries."""
    results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.9))])
    predictor = RulePredictor(perfect_eval_graph, results)

    config = EvaluationConfig(k_values=(1, 3), filter_known=True)
    evaluator = LinkPredictionEvaluator(predictor, perfect_eval_graph, config)

    test_triples = [
        Triple(
            head_id=0,
            tail_id=1,
            head_type="person",
            tail_type="city",
            relation="lives_in",
        ),
        Triple(
            head_id=1,
            tail_id=2,
            head_type="person",
            tail_type="city",
            relation="lives_in",
        ),
        Triple(
            head_id=2,
            tail_id=0,
            head_type="person",
            tail_type="city",
            relation="lives_in",
        ),
    ]

    metrics = evaluator.evaluate(test_triples)

    assert metrics.num_queries == 6
    assert 0.0 <= metrics.mrr <= 1.0


def test_empty_triples_returns_zeros(
    perfect_eval_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """Empty test set should return zero metrics."""
    results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.9))])
    predictor = RulePredictor(perfect_eval_graph, results)

    config = EvaluationConfig(k_values=(1, 3, 10), filter_known=True)
    evaluator = LinkPredictionEvaluator(predictor, perfect_eval_graph, config)

    metrics = evaluator.evaluate([])

    assert metrics.mrr == 0.0
    assert metrics.num_queries == 0
    assert all(v == 0.0 for v in metrics.hits_at_k.values())


@pytest.mark.parametrize(
    ("tie_handling", "expected"),
    [
        (TieHandling.OPTIMISTIC, 2.0),
        (TieHandling.AVERAGE, 3.0),
        (TieHandling.PESSIMISTIC, 4.0),
    ],
)
def test_rank_with_ties_policies(tie_handling: TieHandling, expected: float) -> None:
    """One candidate ranks above the target; two more tie with it."""
    scores = torch.tensor([0.5, 0.5, 0.5, 0.9])

    assert rank_with_ties(scores, 0, tie_handling) == expected


def test_rank_with_ties_untied_target_is_unaffected_by_policy() -> None:
    scores = torch.tensor([0.9, 0.5, 0.1])

    ranks = {rank_with_ties(scores, 0, policy) for policy in TieHandling}

    assert ranks == {1.0}


def test_evaluation_config_defaults_to_average_tie_handling() -> None:
    """Optimistic ranking inflates MRR when scores are coarse, so is not default."""
    assert EvaluationConfig().tie_handling is TieHandling.AVERAGE


def test_rank_queries_rejects_mixed_edge_types(
    perfect_eval_graph: HeteroGraph,
) -> None:
    """Triples of different edge types would be filtered against the wrong edges."""
    evaluator = LinkPredictionEvaluator(
        _NegativeScorer(), perfect_eval_graph, EvaluationConfig()
    )
    triples = [
        Triple(
            head_id=0, tail_id=2, head_type="person", tail_type="city", relation=name
        )
        for name in ("lives_in", "born_in")
    ]

    with pytest.raises(ConfigurationError, match="one edge type"):
        evaluator.rank_queries(triples)


class _RecordingScorer:
    """Scores every candidate zero and records which queries it was asked."""

    def __init__(self, label: str, calls: list[tuple[str, str, int]]) -> None:
        self._label = label
        self._calls = calls

    def score_tails(self, head_id: int) -> torch.Tensor:
        self._calls.append((self._label, "tail", head_id))
        return torch.zeros(3)

    def score_heads(self, tail_id: int) -> torch.Tensor:
        self._calls.append((self._label, "head", tail_id))
        return torch.zeros(3)


def _lives_in(head_id: int, tail_id: int) -> Triple:
    return Triple(
        head_id=head_id,
        tail_id=tail_id,
        head_type="person",
        tail_type="city",
        relation="lives_in",
    )


class TestRankQueriesWithEach:
    def test_matches_ranking_each_scorer_alone(
        self, perfect_eval_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        predictor = RulePredictor(
            perfect_eval_graph,
            RuleSet.of([(cyclic_rule, metrics_with(confidence=0.9))]),
        )
        scorers = {"rules": predictor, "negative": _NegativeScorer()}
        triples = [_lives_in(2, 0), _lives_in(0, 1), _lives_in(1, 2), _lives_in(0, 2)]
        config = EvaluationConfig()

        ranked = rank_queries_with_each(scorers, perfect_eval_graph, triples, config)

        for label, scorer in scorers.items():
            alone = LinkPredictionEvaluator(scorer, perfect_eval_graph, config)
            assert ranked[label] == alone.rank_queries(triples)

    def test_scores_each_query_with_every_scorer_in_turn(
        self, perfect_eval_graph: HeteroGraph
    ) -> None:
        calls: list[tuple[str, str, int]] = []
        scorers = {
            label: _RecordingScorer(label, calls) for label in ("first", "second")
        }
        triples = [_lives_in(2, 0), _lives_in(0, 1), _lives_in(2, 1)]

        rank_queries_with_each(scorers, perfect_eval_graph, triples, EvaluationConfig())

        queries = [(direction, entity) for _, direction, entity in calls[::2]]
        assert [label for label, _, _ in calls] == ["first", "second"] * 6
        assert calls[1::2] == [("second", *query) for query in queries]
        assert queries == sorted(queries)

    def test_empty_triples_give_no_ranks(self, perfect_eval_graph: HeteroGraph) -> None:
        ranked = rank_queries_with_each(
            {"negative": _NegativeScorer()}, perfect_eval_graph, [], EvaluationConfig()
        )

        assert ranked == {"negative": ()}
