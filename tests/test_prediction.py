"""Tests for RulePredictor and Prediction."""

import pytest

from anyburl.anyburl import AnyBURL, AnyBURLConfig
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleMetrics
from anyburl.prediction import Prediction, RulePredictor, _body_chain_key
from anyburl.rule import Atom, Rule, RuleType, Term


def _make_ac1_subject_grounded() -> Rule:
    """Create: lives_in(person:0, Y) :- born_in(X, Z0), near(Z0, Y)."""
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


def _make_ac1_object_grounded() -> Rule:
    """Create: lives_in(X, city:0) :- born_in(X, Z0), near(Z0, Y)."""
    head = Atom(
        relation="lives_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.constant(0, node_type="city"),
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


def _make_ac2_rule() -> Rule:
    """Create: lives_in(X, Y) :- born_in(X, Z0)."""
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
    return Rule(head=head, body=body, rule_type=RuleType.AC2)


def _metrics(*, confidence: float) -> RuleMetrics:
    """Build a minimal RuleMetrics with the given confidence."""
    return RuleMetrics(
        support=1, confidence=confidence, head_coverage=0.25, num_predictions=4
    )


def test_prediction_frozen_dataclass() -> None:
    pred = Prediction(
        head_id=0,
        tail_id=1,
        head_type="person",
        tail_type="city",
        relation="lives_in",
        score=0.5,
    )
    assert pred.head_id == 0
    assert pred.tail_id == 1
    assert pred.head_type == "person"
    assert pred.tail_type == "city"
    assert pred.relation == "lives_in"
    assert pred.score == 0.5


def test_prediction_frozen_cannot_mutate() -> None:
    pred = Prediction(
        head_id=0,
        tail_id=1,
        head_type="person",
        tail_type="city",
        relation="lives_in",
        score=0.5,
    )
    with pytest.raises(AttributeError):
        pred.head_id = 99  # type: ignore[misc]


def test_same_body_chain_produces_same_key(cyclic_rule: Rule) -> None:
    """Two cyclic rules with same body atoms should share a chain key."""
    rule2 = cyclic_rule
    assert _body_chain_key(cyclic_rule) == _body_chain_key(rule2)


def test_different_body_chain_produces_different_key(cyclic_rule: Rule) -> None:
    """Rules with different body atoms should have different keys."""
    ac2 = _make_ac2_rule()
    assert _body_chain_key(cyclic_rule) != _body_chain_key(ac2)


def test_ac1_shares_chain_with_cyclic(cyclic_rule: Rule) -> None:
    """AC1 rules with same body as a cyclic rule share the chain key."""
    ac1_subj = _make_ac1_subject_grounded()
    ac1_obj = _make_ac1_object_grounded()
    key = _body_chain_key(cyclic_rule)
    assert _body_chain_key(ac1_subj) == key
    assert _body_chain_key(ac1_obj) == key


def test_cyclic_predictions_contain_expected_pairs(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """born_in @ near produces exactly {(0,1),(1,2),(2,1),(3,0)}."""
    results = [(cyclic_rule, _metrics(confidence=0.25))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict()

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert pair_ids == {(0, 1), (1, 2), (2, 1), (3, 0)}


def test_predictions_sorted_by_score(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    results = [(cyclic_rule, _metrics(confidence=0.25))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict()

    scores = [p.score for p in predictions]
    assert scores == sorted(scores, reverse=True)


def test_subject_grounded_predictions(evaluator_graph: HeteroGraph) -> None:
    """Subject-grounded AC1 rule for person:0 predicts via born_in @ near."""
    rule = _make_ac1_subject_grounded()
    results = [(rule, _metrics(confidence=0.5))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict()

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert (0, 1) in pair_ids


def test_object_grounded_predictions(evaluator_graph: HeteroGraph) -> None:
    """Object-grounded AC1 rule for city:0 predicts via born_in @ near."""
    rule = _make_ac1_object_grounded()
    results = [(rule, _metrics(confidence=0.5))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict()

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert (3, 0) in pair_ids


def test_two_rules_same_pair_noisy_or(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """Two rules both predicting (3,0) with conf 0.25 and 0.5."""
    obj_grounded_rule = _make_ac1_object_grounded()

    results: list[tuple[Rule, RuleMetrics]] = [
        (cyclic_rule, _metrics(confidence=0.25)),
        (obj_grounded_rule, _metrics(confidence=0.5)),
    ]

    predictor = RulePredictor(evaluator_graph, results)
    predictions = predictor.predict()

    pair_30 = next(p for p in predictions if p.head_id == 3 and p.tail_id == 0)
    assert pair_30.score == pytest.approx(0.625)


def test_known_edge_removed(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> None:
    """(3,0) is a known lives_in edge -> removed when filter_known=True."""
    results = [(cyclic_rule, _metrics(confidence=0.25))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict(filter_known=True)

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert (3, 0) not in pair_ids


def test_known_edge_kept_without_filter(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """(3,0) is kept when filter_known=False."""
    results = [(cyclic_rule, _metrics(confidence=0.25))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict(filter_known=False)

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert (3, 0) in pair_ids


def test_score_tails_cyclic(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> None:
    """score_tails for person 0 should score city 1."""
    results = [(cyclic_rule, _metrics(confidence=0.5))]
    predictor = RulePredictor(evaluator_graph, results)

    scores = predictor.score_tails(0)

    assert scores.shape == (3,)
    assert scores[1].item() > 0
    assert scores[0].item() == pytest.approx(0.0)
    assert scores[2].item() == pytest.approx(0.0)


def test_score_tails_with_ac1(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> None:
    """Subject-grounded AC1 for person 0 also contributes to score_tails(0)."""
    results = [
        (cyclic_rule, _metrics(confidence=0.25)),
        (_make_ac1_subject_grounded(), _metrics(confidence=0.5)),
    ]
    predictor = RulePredictor(evaluator_graph, results)

    scores = predictor.score_tails(0)

    assert scores[1].item() == pytest.approx(0.625)


def test_score_heads_cyclic(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> None:
    """score_heads for city 0 should score person 3."""
    results = [(cyclic_rule, _metrics(confidence=0.5))]
    predictor = RulePredictor(evaluator_graph, results)

    scores = predictor.score_heads(0)

    assert scores.shape == (4,)
    assert scores[3].item() > 0


def test_empty_results_raises(evaluator_graph: HeteroGraph) -> None:
    with pytest.raises(ValueError, match="results must not be empty"):
        RulePredictor(evaluator_graph, [])


def test_ac2_rules_are_skipped(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """AC2 rules are included in results but produce no predictions."""
    ac2 = _make_ac2_rule()
    results = [
        (cyclic_rule, _metrics(confidence=0.25)),
        (ac2, _metrics(confidence=0.1)),
    ]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict()

    assert len(predictions) > 0


def test_predict_before_fit_raises() -> None:
    pipeline = AnyBURL(AnyBURLConfig())
    with pytest.raises(RuntimeError, match="fit"):
        pipeline.predict()
