"""Tests for RuleEvaluator."""

import pytest

from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleEvaluator, RuleMetrics, aggregate_confidence
from anyburl.rule import Atom, Rule, RuleConfig, RuleType, Term


def _make_ac1_rule_subject_grounded() -> Rule:
    """Create: lives_in(person:0, Y) :- born_in(X, Z0), near(Z0, Y).

    X is a free variable in the body. The evaluator pins X to person:0
    (the head constant) when computing the forward chain.
    """
    head = Atom(
        relation="lives_in",
        subject=Term.constant(0, node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    body = (
        Atom(
            relation="born_in",
            subject=Term.variable("X", node_type="person"),
            object_=Term.variable("Z0", node_type="city"),
        ),
        Atom(
            relation="near",
            subject=Term.variable("Z0", node_type="city"),
            object_=Term.variable("Y", node_type="city"),
        ),
    )
    return Rule(head=head, body=body, rule_type=RuleType.AC1)


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
    config = RuleConfig(min_support=1, min_confidence=0.0, min_head_coverage=0.0)
    evaluator = RuleEvaluator(evaluator_graph, config)
    metrics = evaluator.evaluate(cyclic_rule)

    assert metrics.num_predictions == 4
    assert metrics.support == 1
    assert metrics.confidence == 0.25
    assert metrics.head_coverage == 0.25


def test_evaluate_ac1_rule(evaluator_graph: HeteroGraph) -> None:
    config = RuleConfig(min_support=1, min_confidence=0.0, min_head_coverage=0.0)
    evaluator = RuleEvaluator(evaluator_graph, config)
    rule = _make_ac1_rule_subject_grounded()
    metrics = evaluator.evaluate(rule)

    assert isinstance(metrics, RuleMetrics)
    assert metrics.num_predictions == 1
    assert metrics.support == 0
    assert metrics.confidence == 0.0


def test_evaluate_ac2_subject_connected(evaluator_graph: HeteroGraph) -> None:
    """Evaluate AC2 rule: lives_in(X, Y) :- born_in(X, Z0).

    born_in has 4 persons with edges -> connected_count=4.
    disconnected type=city, 3 cities -> num_predictions=12.
    All 4 lives_in triples have subjects in connected set -> support=4.
    confidence=4/12=1/3, head_coverage=4/4=1.0.
    """
    config = RuleConfig(min_support=1, min_confidence=0.0, min_head_coverage=0.0)
    evaluator = RuleEvaluator(evaluator_graph, config)

    head = Atom(
        relation="lives_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    body = (
        Atom(
            relation="born_in",
            subject=Term.variable("X", node_type="person"),
            object_=Term.variable("Z0", node_type="city"),
        ),
    )
    rule = Rule(head=head, body=body, rule_type=RuleType.AC2)
    metrics = evaluator.evaluate(rule)

    assert metrics.num_predictions == 12
    assert metrics.support == 4
    assert metrics.confidence == pytest.approx(1.0 / 3.0)
    assert metrics.head_coverage == pytest.approx(1.0)


def test_evaluate_ac2_object_connected(evaluator_graph: HeteroGraph) -> None:
    """Evaluate AC2 rule: lives_in(X, Y) :- near(Z0, Y).

    near targets all 3 cities -> connected_count=3.
    disconnected type=person, 4 persons -> num_predictions=12.
    All 4 lives_in triples have targets in connected set -> support=4.
    confidence=4/12=1/3, head_coverage=4/4=1.0.
    """
    config = RuleConfig(min_support=1, min_confidence=0.0, min_head_coverage=0.0)
    evaluator = RuleEvaluator(evaluator_graph, config)

    head = Atom(
        relation="lives_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    body = (
        Atom(
            relation="near",
            subject=Term.variable("Z0", node_type="city"),
            object_=Term.variable("Y", node_type="city"),
        ),
    )
    rule = Rule(head=head, body=body, rule_type=RuleType.AC2)
    metrics = evaluator.evaluate(rule)

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
    ac1_rule = _make_ac1_rule_subject_grounded()

    results = evaluator.evaluate_batch([cyclic_rule, ac1_rule])

    passing_rules = [r for r, _ in results]
    assert cyclic_rule in passing_rules
    assert ac1_rule not in passing_rules


def test_evaluate_batch_max_results(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    config = RuleConfig(min_support=1, min_confidence=0.0, min_head_coverage=0.0)
    evaluator = RuleEvaluator(evaluator_graph, config)

    results = evaluator.evaluate_batch(
        [cyclic_rule, cyclic_rule, cyclic_rule], max_results=1
    )
    assert len(results) == 1


def _make_ac1_rule_object_grounded() -> Rule:
    """Create: lives_in(X, city:1) :- born_in(X, Z0), near(Z0, Y).

    Y is a free variable in the body. The evaluator pins Y to city:1
    (the head constant) via the backward/transposed chain.
    """
    head = Atom(
        relation="lives_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.constant(1, node_type="city"),
    )
    body = (
        Atom(
            relation="born_in",
            subject=Term.variable("X", node_type="person"),
            object_=Term.variable("Z0", node_type="city"),
        ),
        Atom(
            relation="near",
            subject=Term.variable("Z0", node_type="city"),
            object_=Term.variable("Y", node_type="city"),
        ),
    )
    return Rule(head=head, body=body, rule_type=RuleType.AC1)


def test_grouped_batch_matches_single_evaluate(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """Grouped batch evaluation must equal per-rule evaluate() exactly.

    Covers cyclic, subject-grounded AC1 (forward chain), and
    object-grounded AC1 (transposed chain) rule types.
    """
    config = RuleConfig(min_support=1, min_confidence=0.0, min_head_coverage=0.0)
    evaluator = RuleEvaluator(evaluator_graph, config)
    rules = [
        cyclic_rule,
        _make_ac1_rule_subject_grounded(),
        _make_ac1_rule_object_grounded(),
    ]

    grouped = evaluator._compute_all_metrics(rules)

    for rule in rules:
        assert grouped[rule] == evaluator.evaluate(rule)
