"""Tests for LinkPredictionEvaluator."""

import pytest

from anyburl.evaluation import (
    EvaluationConfig,
    LinkPredictionEvaluator,
    LinkPredictionMetrics,
)
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleMetrics
from anyburl.prediction import RulePredictor
from anyburl.rule import Rule
from anyburl.sampler import Triple


def _metrics(*, confidence: float) -> RuleMetrics:
    return RuleMetrics(
        support=1, confidence=confidence, head_coverage=0.33, num_predictions=3
    )


def test_perfect_mrr(
    perfect_eval_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """With a perfect rule, all triples should rank 1 -> MRR = 1.0."""
    results = [(cyclic_rule, _metrics(confidence=0.9))]
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
    results = [(cyclic_rule, _metrics(confidence=0.9))]
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
    results = [(cyclic_rule, _metrics(confidence=0.9))]
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


def test_averaged_metrics(
    perfect_eval_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """Metrics should be averaged across all queries."""
    results = [(cyclic_rule, _metrics(confidence=0.9))]
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
    results = [(cyclic_rule, _metrics(confidence=0.9))]
    predictor = RulePredictor(perfect_eval_graph, results)

    config = EvaluationConfig(k_values=(1, 3, 10), filter_known=True)
    evaluator = LinkPredictionEvaluator(predictor, perfect_eval_graph, config)

    metrics = evaluator.evaluate([])

    assert metrics.mrr == 0.0
    assert metrics.num_queries == 0
    assert all(v == 0.0 for v in metrics.hits_at_k.values())


def test_link_prediction_metrics_values() -> None:
    metrics = LinkPredictionMetrics(
        mrr=0.5,
        hits_at_k={1: 0.3, 3: 0.5},
        num_queries=10,
    )
    assert metrics.mrr == 0.5
    assert metrics.hits_at_k[1] == 0.3
    assert metrics.num_queries == 10


def test_link_prediction_metrics_frozen() -> None:
    metrics = LinkPredictionMetrics(mrr=0.5)
    with pytest.raises(AttributeError):
        metrics.mrr = 0.8  # type: ignore[misc]
