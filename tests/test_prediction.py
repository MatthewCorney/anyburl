"""Tests for RulePredictor and Prediction."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.anyburl import AnyBURL, AnyBURLConfig
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleMetrics
from anyburl.prediction import (
    GroundingMode,
    Prediction,
    RuleFiring,
    RulePredictor,
    ScoringStrategy,
)
from anyburl.rule import Atom, Rule, RuleType, Term


def _make_ac1_subject_grounded() -> Rule:
    """Create: lives_in(person:0, Y) :- born_in(X, Z0), near(Z0, Y).

    X is a free variable. The evaluator pins X to person:0 when computing
    the forward chain, which is the intended AC1 semantics.
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


def test_prediction_frozen_cannot_mutate() -> None:
    pred = Prediction(
        head_id=0,
        tail_id=1,
        head_type="person",
        tail_type="city",
        relation="lives_in",
        score=0.5,
    )
    with pytest.raises(AttributeError, match="cannot assign to field"):
        pred.head_id = 99  # type: ignore[misc]


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

    cyclic_only = RulePredictor(
        evaluator_graph, [(cyclic_rule, _metrics(confidence=0.25))]
    )
    expected = cyclic_only.predict()

    assert {(p.head_id, p.tail_id) for p in predictions} == {
        (p.head_id, p.tail_id) for p in expected
    }


def test_predict_before_fit_raises() -> None:
    pipeline = AnyBURL(AnyBURLConfig())
    with pytest.raises(RuntimeError, match="fit"):
        pipeline.predict()


@pytest.fixture
def multi_grounding_graph() -> HeteroGraph:
    """Person 0 reaches city 2 by two body groundings and city 3 by one."""
    data = HeteroData()
    data["person"].num_nodes = 1
    data["city"].num_nodes = 4
    data["person", "born_in", "city"].edge_index = torch.tensor([[0, 0], [0, 1]])
    data["city", "near", "city"].edge_index = torch.tensor([[0, 1, 0], [2, 2, 3]])
    data["person", "lives_in", "city"].edge_index = torch.tensor([[0], [2]])
    return HeteroGraph(data)


def _grounding_rule() -> Rule:
    """lives_in(X, Y) :- born_in(X, Z0), near(Z0, Y)."""
    return Rule(
        head=Atom(
            relation="lives_in",
            subject=Term.variable("X", node_type="person"),
            object_=Term.variable("Y", node_type="city"),
        ),
        body=(
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
        ),
        rule_type=RuleType.CYCLIC,
    )


def test_noisy_or_ignores_grounding_count(multi_grounding_graph: HeteroGraph) -> None:
    """Under NOISY_OR a candidate is only reachable or not."""
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(
        multi_grounding_graph, results, scoring_strategy=ScoringStrategy.NOISY_OR
    )

    scores = predictor.score_tails(0)

    assert scores[2].item() == pytest.approx(scores[3].item())


def test_path_weighted_ranks_more_groundings_higher(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """Two body groundings must outrank one."""
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(
        multi_grounding_graph, results, scoring_strategy=ScoringStrategy.PATH_WEIGHTED
    )

    scores = predictor.score_tails(0)

    assert scores[2].item() > scores[3].item()


def test_path_weighted_matches_noisy_or_for_single_grounding(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """log2(1 + 1) == 1, so one grounding weighs exactly one firing."""
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    weighted = RulePredictor(
        multi_grounding_graph, results, scoring_strategy=ScoringStrategy.PATH_WEIGHTED
    ).score_tails(0)
    plain = RulePredictor(
        multi_grounding_graph, results, scoring_strategy=ScoringStrategy.NOISY_OR
    ).score_tails(0)

    assert weighted[3].item() == pytest.approx(plain[3].item())


def test_score_heads_is_path_weighted_too(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """The reverse direction must weight groundings the same way."""
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(
        multi_grounding_graph, results, scoring_strategy=ScoringStrategy.PATH_WEIGHTED
    )
    plain = RulePredictor(
        multi_grounding_graph, results, scoring_strategy=ScoringStrategy.NOISY_OR
    )

    assert predictor.score_heads(2)[0].item() > plain.score_heads(2)[0].item()


def _both_modes(
    graph: HeteroGraph, results: list
) -> tuple[RulePredictor, RulePredictor]:
    """Return materialised and on-demand predictors over the same rules."""
    return (
        RulePredictor(graph, results, grounding_mode=GroundingMode.MATERIALISED),
        RulePredictor(graph, results, grounding_mode=GroundingMode.ON_DEMAND),
    )


def test_on_demand_score_tails_matches_materialised(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """Grounding per query must not change the scores, only the memory."""
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    materialised, on_demand = _both_modes(multi_grounding_graph, results)

    assert torch.allclose(materialised.score_tails(0), on_demand.score_tails(0))


def test_on_demand_score_heads_matches_materialised(
    multi_grounding_graph: HeteroGraph,
) -> None:
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    materialised, on_demand = _both_modes(multi_grounding_graph, results)

    for tail_id in range(4):
        assert torch.allclose(
            materialised.score_heads(tail_id), on_demand.score_heads(tail_id)
        )


def test_on_demand_preserves_grounding_counts(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """Path multiplicity must survive vector propagation, not just reachability."""
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    _, on_demand = _both_modes(multi_grounding_graph, results)

    scores = on_demand.score_tails(0)

    assert scores[2].item() > scores[3].item()


def test_on_demand_matches_materialised_with_ac1_rules(
    evaluator_graph: HeteroGraph, cyclic_rule: Rule
) -> None:
    """AC1 groups read the same grounding through a different direction."""
    results = [
        (cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3)),
        (_make_ac1_subject_grounded(), RuleMetrics(1, 0.7, 0.33, 3)),
    ]
    materialised, on_demand = _both_modes(evaluator_graph, results)

    for person_id in range(4):
        assert torch.allclose(
            materialised.score_tails(person_id), on_demand.score_tails(person_id)
        )
    for city_id in range(3):
        assert torch.allclose(
            materialised.score_heads(city_id), on_demand.score_heads(city_id)
        )


def test_explain_names_the_firing_chain(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """A score with no provenance cannot be checked; explain must name the body."""
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    firings = predictor.explain(0, 2)

    assert len(firings) == 1
    assert firings[0].chain == (
        ("person", "born_in", "city"),
        ("city", "near", "city"),
    )
    assert firings[0].rule_type is RuleType.CYCLIC
    assert firings[0].confidence == pytest.approx(0.5)


def test_explain_reports_grounding_counts(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """City 2 is reached by two body groundings, city 3 by one."""
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    assert predictor.explain(0, 2)[0].groundings == 2
    assert predictor.explain(0, 3)[0].groundings == 1


def test_explain_returns_nothing_for_unreached_pairs(
    multi_grounding_graph: HeteroGraph,
) -> None:
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    assert predictor.explain(0, 0) == ()


def test_explanation_contributions_sum_to_the_score(
    evaluator_graph: HeteroGraph, cyclic_rule: Rule
) -> None:
    """Independent firings must compose by noisy-or into the reported score."""
    results = [(cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3))]
    predictor = RulePredictor(evaluator_graph, results)

    scores = predictor.score_tails(0)
    for city_id in range(evaluator_graph.node_count("city")):
        firings = predictor.explain(0, city_id)
        complement = 1.0
        for firing in firings:
            complement *= 1.0 - firing.contribution
        assert 1.0 - complement == pytest.approx(scores[city_id].item(), abs=1e-6)


def test_top_tails_is_sorted_and_carries_explanations(
    multi_grounding_graph: HeteroGraph,
) -> None:
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    predictions = predictor.top_tails(0, limit=5)

    assert [p.tail_id for p in predictions] == [2, 3]
    assert [p.score for p in predictions] == sorted(
        (p.score for p in predictions), reverse=True
    )
    assert all(p.explanations for p in predictions)


def test_top_tails_respects_the_limit(multi_grounding_graph: HeteroGraph) -> None:
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    assert len(predictor.top_tails(0, limit=1)) == 1


def test_top_tails_rejects_a_non_positive_limit(
    multi_grounding_graph: HeteroGraph,
) -> None:
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    with pytest.raises(ValueError, match="limit must be positive"):
        predictor.top_tails(0, limit=0)


def test_predict_leaves_explanations_empty(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """Attaching provenance to every pair would dwarf the scores."""
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    assert all(p.explanations == () for p in predictor.predict())


def test_rule_firing_describe_is_readable(
    multi_grounding_graph: HeteroGraph,
) -> None:
    results = [(_grounding_rule(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    described = predictor.explain(0, 2)[0].describe()

    assert "person_born_in_city -> city_near_city" in described
    assert "groundings=2" in described


def test_describe_disambiguates_shared_relation_names() -> None:
    """DBLP names every relation "to"; bare names would render chains alike."""
    firing = RuleFiring(
        chain=(("author", "to", "paper"), ("paper", "to", "author")),
        rule_type=RuleType.CYCLIC,
        confidence=0.5,
        groundings=3,
        contribution=0.5,
    )

    assert "author_to_paper -> paper_to_author" in firing.describe()
