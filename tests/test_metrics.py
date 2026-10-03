"""Tests for RuleEvaluator."""

import logging

import pytest

from anyburl.exceptions import GraphSchemaError
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleEvaluator, RuleMetrics, aggregate_confidence
from anyburl.rule import Atom, Rule, RuleConfig, RuleType, Term
from tests.rules import ac2_born_in, lives_in, object_grounded, subject_grounded

PERMISSIVE = RuleConfig(min_support=1, min_confidence=0.0, min_head_coverage=0.0)
WARNING_TEXT = "not comparable across rule types"


@pytest.mark.parametrize(
    ("confidences", "expected"),
    [
        ([], 0.0),
        ([0.5], 0.5),
        ([0.5, 0.5], 0.75),
    ],
)
def test_aggregate_confidence(confidences: list[float], expected: float) -> None:
    assert aggregate_confidence(confidences) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("num_predictions", "expected"),
    [(0, True), (4, False)],
)
def test_rule_metrics_is_trivial(num_predictions: int, expected: bool) -> None:
    metrics = RuleMetrics(
        support=0,
        confidence=0.0,
        head_coverage=0.0,
        num_predictions=num_predictions,
    )
    assert metrics.is_trivial is expected


@pytest.mark.parametrize(
    ("support", "confidence", "head_coverage", "passes"),
    [
        (2, 0.5, 0.5, True),
        (0, 0.5, 0.5, False),
        (2, 0.0, 0.5, False),
        (2, 0.5, 0.0, False),
    ],
)
def test_rule_metrics_passes_thresholds(
    support: int,
    confidence: float,
    head_coverage: float,
    passes: bool,
) -> None:
    metrics = RuleMetrics(
        support=support,
        confidence=confidence,
        head_coverage=head_coverage,
        num_predictions=4,
    )
    result = metrics.passes_thresholds(
        min_support=1,
        min_confidence=0.1,
        min_head_coverage=0.1,
    )
    assert result is passes


def test_evaluate_cyclic_hand_computed(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """Verify metrics against hand-computed values.

    born_in @ near predicts: (0,1), (1,2), (2,1), (3,0) = 4 predictions.
    lives_in actual: (0,0), (1,1), (2,2), (3,0).
    Intersection: (3,0) = 1 support.
    confidence = 1/4 = 0.25, head_coverage = 1/4 = 0.25.
    """
    metrics = RuleEvaluator(evaluator_graph, PERMISSIVE).evaluate(cyclic_rule)

    assert metrics.num_predictions == 4
    assert metrics.support == 1
    assert metrics.confidence == 0.25
    assert metrics.head_coverage == 0.25


def test_evaluate_ac1_rule(evaluator_graph: HeteroGraph) -> None:
    metrics = RuleEvaluator(evaluator_graph, PERMISSIVE).evaluate(subject_grounded(0))

    assert metrics.num_predictions == 1
    assert metrics.support == 0
    assert metrics.confidence == 0.0


@pytest.mark.parametrize(
    "rule",
    [
        ac2_born_in(),
        Rule(
            head=lives_in(
                Term.variable("X", node_type="person"),
                Term.variable("Y", node_type="city"),
            ),
            body=(
                Atom(
                    relation="near",
                    subject=Term.variable("Z0", node_type="city"),
                    object_=Term.variable("Y", node_type="city"),
                ),
            ),
            rule_type=RuleType.AC2,
        ),
    ],
    ids=["subject_connected", "object_connected"],
)
def test_evaluate_ac2(evaluator_graph: HeteroGraph, rule: Rule) -> None:
    """The connected side reaches every entity of its type: 4 x 3 predictions.

    All 4 ``lives_in`` triples have a connected endpoint, so support is 4.
    """
    metrics = RuleEvaluator(evaluator_graph, PERMISSIVE).evaluate(rule)

    assert metrics.num_predictions == 12
    assert metrics.support == 4
    assert metrics.confidence == pytest.approx(1.0 / 3.0)
    assert metrics.head_coverage == pytest.approx(1.0)


def test_evaluate_batch_filters(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    config = RuleConfig(min_support=1, min_confidence=0.1, min_head_coverage=0.1)
    evaluator = RuleEvaluator(evaluator_graph, config)
    ac1_rule = subject_grounded(0)

    results = evaluator.evaluate_batch([cyclic_rule, ac1_rule])

    passing_rules = [r for r, _ in results]
    assert cyclic_rule in passing_rules
    assert ac1_rule not in passing_rules


def test_evaluate_batch_max_results(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    evaluator = RuleEvaluator(evaluator_graph, PERMISSIVE)

    results = evaluator.evaluate_batch(
        [cyclic_rule, cyclic_rule, cyclic_rule], max_results=1
    )
    assert len(results) == 1


def test_batch_evaluation_matches_single_evaluation(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    evaluator = RuleEvaluator(evaluator_graph, PERMISSIVE)
    rules = [cyclic_rule, subject_grounded(3), object_grounded(0)]

    batch = dict(evaluator.evaluate_batch(rules))

    assert batch == {rule: evaluator.evaluate(rule) for rule in rules}


def test_unknown_edge_type_raises(evaluator_graph: HeteroGraph) -> None:
    rule = Rule(
        head=lives_in(
            Term.variable("X", node_type="person"), Term.variable("Y", node_type="city")
        ),
        body=(
            Atom(
                relation="visited",
                subject=Term.variable("X", node_type="person"),
                object_=Term.variable("Y", node_type="city"),
            ),
        ),
        rule_type=RuleType.CYCLIC,
    )

    with pytest.raises(GraphSchemaError, match="No edge type matches"):
        RuleEvaluator(evaluator_graph, PERMISSIVE).evaluate(rule)


def test_warns_when_head_coverage_alone_eliminates_a_type(
    evaluator_graph: HeteroGraph, cyclic_rule: Rule, caplog: pytest.LogCaptureFixture
) -> None:
    """Rules clearing support and confidence but not head coverage."""
    config = RuleConfig(min_support=1, min_confidence=0.0, min_head_coverage=0.99)
    evaluator = RuleEvaluator(evaluator_graph, config)

    with caplog.at_level(logging.WARNING, logger="anyburl"):
        evaluator.evaluate_batch([cyclic_rule])

    assert WARNING_TEXT in caplog.text


def test_does_not_warn_when_rules_are_simply_poor(
    evaluator_graph: HeteroGraph, cyclic_rule: Rule, caplog: pytest.LogCaptureFixture
) -> None:
    """A type failing on support is not a threshold-scale problem."""
    config = RuleConfig(min_support=10_000, min_confidence=0.0, min_head_coverage=0.99)
    evaluator = RuleEvaluator(evaluator_graph, config)

    with caplog.at_level(logging.WARNING, logger="anyburl"):
        evaluator.evaluate_batch([cyclic_rule])

    assert WARNING_TEXT not in caplog.text


def test_does_not_warn_without_a_head_coverage_floor(
    evaluator_graph: HeteroGraph, cyclic_rule: Rule, caplog: pytest.LogCaptureFixture
) -> None:
    evaluator = RuleEvaluator(evaluator_graph, PERMISSIVE)

    with caplog.at_level(logging.WARNING, logger="anyburl"):
        evaluator.evaluate_batch([cyclic_rule])

    assert WARNING_TEXT not in caplog.text
