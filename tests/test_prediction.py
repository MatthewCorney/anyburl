"""Tests for RulePredictor and Prediction."""

import math

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.core import QueryDirection
from anyburl.exceptions import ConfigurationError, GraphSchemaError, NotFittedError
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleMetrics
from anyburl.pipeline import AnyBURL, AnyBURLConfig
from anyburl.prediction import (
    ConfidenceAggregation,
    GroundingWeighting,
    Prediction,
    PredictionConfig,
    RuleEvidence,
    RuleFiring,
    RulePredictor,
)
from anyburl.prediction.grounding import body_chain_key, build_groundings
from anyburl.prediction.scoring import GROUNDING_WEIGHT_OFFSET, TIE_BREAK_WEIGHT
from anyburl.rule import Atom, Rule, RuleType, Term
from anyburl.rule.quality import corrected_confidence
from tests.rules import (
    ac2_born_in,
    cyclic_born_in_near,
    metrics_with,
    object_grounded,
    subject_grounded,
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
    results = [(cyclic_rule, metrics_with(confidence=0.25))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict()

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert pair_ids == {(0, 1), (1, 2), (2, 1), (3, 0)}


def test_predictions_sorted_by_score(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    results = [(cyclic_rule, metrics_with(confidence=0.25))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict()

    scores = [p.score for p in predictions]
    assert scores == sorted(scores, reverse=True)


def test_subject_grounded_predictions(evaluator_graph: HeteroGraph) -> None:
    """Subject-grounded AC1 rule for person:0 predicts via born_in @ near."""
    rule = subject_grounded(0)
    results = [(rule, metrics_with(confidence=0.5))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict()

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert (0, 1) in pair_ids


def test_object_grounded_predictions(evaluator_graph: HeteroGraph) -> None:
    """Object-grounded AC1 rule for city:0 predicts via born_in @ near."""
    rule = object_grounded(0)
    results = [(rule, metrics_with(confidence=0.5))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict()

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert (3, 0) in pair_ids


def test_two_rules_same_pair_noisy_or(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """Two rules both predicting (3,0) with conf 0.25 and 0.5."""
    obj_grounded_rule = object_grounded(0)

    results: list[tuple[Rule, RuleMetrics]] = [
        (cyclic_rule, metrics_with(confidence=0.25)),
        (obj_grounded_rule, metrics_with(confidence=0.5)),
    ]

    predictor = RulePredictor(evaluator_graph, results)
    predictions = predictor.predict()

    pair_30 = next(p for p in predictions if p.head_id == 3 and p.tail_id == 0)
    assert pair_30.score == pytest.approx(0.625)


def test_known_edge_removed(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> None:
    """(3,0) is a known lives_in edge -> removed when filter_known=True."""
    results = [(cyclic_rule, metrics_with(confidence=0.25))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict(filter_known=True)

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert (3, 0) not in pair_ids


def test_known_edge_kept_without_filter(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """(3,0) is kept when filter_known=False."""
    results = [(cyclic_rule, metrics_with(confidence=0.25))]
    predictor = RulePredictor(evaluator_graph, results)

    predictions = predictor.predict(filter_known=False)

    pair_ids = {(p.head_id, p.tail_id) for p in predictions}
    assert (3, 0) in pair_ids


def test_score_tails_cyclic(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> None:
    """score_tails for person 0 should score city 1."""
    results = [(cyclic_rule, metrics_with(confidence=0.5))]
    predictor = RulePredictor(evaluator_graph, results)

    scores = predictor.score_tails(0)

    assert scores.shape == (3,)
    assert scores[1].item() > 0
    assert scores[0].item() == pytest.approx(0.0)
    assert scores[2].item() == pytest.approx(0.0)


def test_score_tails_with_ac1(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> None:
    """Subject-grounded AC1 for person 0 also contributes to score_tails(0)."""
    results = [
        (cyclic_rule, metrics_with(confidence=0.25)),
        (subject_grounded(0), metrics_with(confidence=0.5)),
    ]
    predictor = RulePredictor(evaluator_graph, results)

    scores = predictor.score_tails(0)

    assert scores[1].item() == pytest.approx(0.625)


def test_score_heads_cyclic(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> None:
    """score_heads for city 0 should score person 3."""
    results = [(cyclic_rule, metrics_with(confidence=0.5))]
    predictor = RulePredictor(evaluator_graph, results)

    scores = predictor.score_heads(0)

    assert scores.shape == (4,)
    assert scores[3].item() > 0


def test_empty_results_raises(evaluator_graph: HeteroGraph) -> None:
    with pytest.raises(ConfigurationError, match="results must not be empty"):
        RulePredictor(evaluator_graph, [])


def test_ac2_rules_are_skipped(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
) -> None:
    """AC2 rules are included in results but produce no predictions."""
    ac2 = ac2_born_in()
    results = [
        (cyclic_rule, metrics_with(confidence=0.25)),
        (ac2, metrics_with(confidence=0.1)),
    ]
    predictor = RulePredictor(evaluator_graph, results)
    predictions = predictor.predict()

    cyclic_only = RulePredictor(
        evaluator_graph, [(cyclic_rule, metrics_with(confidence=0.25))]
    )
    expected = cyclic_only.predict()

    assert {(p.head_id, p.tail_id) for p in predictions} == {
        (p.head_id, p.tail_id) for p in expected
    }


def test_predict_before_fit_raises() -> None:
    pipeline = AnyBURL(AnyBURLConfig())
    with pytest.raises(NotFittedError, match="fit"):
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


def test_noisy_or_ignores_grounding_count(multi_grounding_graph: HeteroGraph) -> None:
    """Under ONCE weighting a candidate is only reachable or not."""
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(
        multi_grounding_graph,
        results,
        config=PredictionConfig(grounding_weighting=GroundingWeighting.ONCE),
    )

    scores = predictor.score_tails(0)

    assert scores[2].item() == pytest.approx(scores[3].item())


def test_path_weighted_ranks_more_groundings_higher(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """Two body groundings must outrank one."""
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(
        multi_grounding_graph,
        results,
        config=PredictionConfig(grounding_weighting=GroundingWeighting.LOG_COUNT),
    )

    scores = predictor.score_tails(0)

    assert scores[2].item() > scores[3].item()


def test_path_weighted_matches_noisy_or_for_single_grounding(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """log2(1 + 1) == 1, so one grounding weighs exactly one firing."""
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    weighted = RulePredictor(
        multi_grounding_graph,
        results,
        config=PredictionConfig(grounding_weighting=GroundingWeighting.LOG_COUNT),
    ).score_tails(0)
    plain = RulePredictor(
        multi_grounding_graph,
        results,
        config=PredictionConfig(grounding_weighting=GroundingWeighting.ONCE),
    ).score_tails(0)

    assert weighted[3].item() == pytest.approx(plain[3].item())


def test_score_heads_is_path_weighted_too(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """The reverse direction must weight groundings the same way."""
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(
        multi_grounding_graph,
        results,
        config=PredictionConfig(grounding_weighting=GroundingWeighting.LOG_COUNT),
    )
    plain = RulePredictor(
        multi_grounding_graph,
        results,
        config=PredictionConfig(grounding_weighting=GroundingWeighting.ONCE),
    )

    assert predictor.score_heads(2)[0].item() > plain.score_heads(2)[0].item()


def _product(graph: HeteroGraph, chain: tuple) -> torch.Tensor:
    """Return the dense product of a chain's adjacency matrices."""
    product = graph.get_csr_matrix(chain[0]).to_dense()
    for edge_type in chain[1:]:
        product = product @ graph.get_csr_matrix(edge_type).to_dense()
    return product


def _dense(entries: tuple[torch.Tensor, torch.Tensor], size: int) -> torch.Tensor:
    """Scatter ``(indices, counts)`` into a dense vector."""
    dense = torch.zeros(size)
    dense[entries[0]] = entries[1]
    return dense


@pytest.mark.parametrize(
    "rule", [cyclic_born_in_near(), subject_grounded(0)], ids=["cyclic", "ac1"]
)
def test_grounding_matches_the_chain_product(
    multi_grounding_graph: HeteroGraph, rule: Rule
) -> None:
    """Each row and column is the chain product's, path counts included."""
    chain = body_chain_key(rule)
    grounding = build_groundings(
        multi_grounding_graph, [(rule, RuleMetrics(2, 0.5, 0.5, 4))]
    )[chain]
    product = _product(multi_grounding_graph, chain)

    for head_id in range(product.size(0)):
        row = grounding.row(head_id)
        assert torch.equal(_dense(row, product.size(1)), product[head_id])
        assert torch.equal(row[0], torch.sort(row[0]).values)
    for tail_id in range(product.size(1)):
        column = grounding.column(tail_id)
        assert torch.equal(_dense(column, product.size(0)), product[:, tail_id])


def test_grounding_preserves_path_counts(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """Path multiplicity must survive grounding, not just reachability."""
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]

    scores = RulePredictor(multi_grounding_graph, results).score_tails(0)

    assert scores[2].item() > scores[3].item()


def test_grounding_rejects_an_edge_type_missing_from_the_graph(
    evaluator_graph: HeteroGraph,
) -> None:
    person = Term.variable("X", node_type="person")
    city = Term.variable("Y", node_type="city")
    rule = Rule(
        head=Atom(relation="lives_in", subject=person, object_=city),
        body=(Atom(relation="visited", subject=person, object_=city),),
        rule_type=RuleType.CYCLIC,
    )

    with pytest.raises(GraphSchemaError):
        RulePredictor(evaluator_graph, [(rule, RuleMetrics(1, 0.5, 0.5, 2))])


def test_explain_names_the_firing_chain(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """A score with no provenance cannot be checked; explain must name the body."""
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
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
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    assert predictor.explain(0, 2)[0].groundings == 2
    assert predictor.explain(0, 3)[0].groundings == 1


def test_explain_returns_nothing_for_unreached_pairs(
    multi_grounding_graph: HeteroGraph,
) -> None:
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
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
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    predictions = predictor.top_tails(0, limit=5)

    assert [p.tail_id for p in predictions] == [2, 3]
    assert [p.score for p in predictions] == sorted(
        (p.score for p in predictions), reverse=True
    )
    assert all(p.explanations for p in predictions)


def test_top_tails_respects_the_limit(multi_grounding_graph: HeteroGraph) -> None:
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    assert len(predictor.top_tails(0, limit=1)) == 1


def test_top_tails_rejects_a_non_positive_limit(
    multi_grounding_graph: HeteroGraph,
) -> None:
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    with pytest.raises(ConfigurationError, match="limit must be positive"):
        predictor.top_tails(0, limit=0)


def test_predict_leaves_explanations_empty(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """Attaching provenance to every pair would dwarf the scores."""
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    assert all(p.explanations == () for p in predictor.predict())


def test_rule_firing_describe_is_readable(
    multi_grounding_graph: HeteroGraph,
) -> None:
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    described = predictor.explain(0, 2)[0].describe()

    assert "person_born_in_city -> city_near_city" in described
    assert "groundings=2" in described


def test_describe_disambiguates_shared_relation_names() -> None:
    """Relation names shared across edge types stay distinguishable."""
    firing = RuleFiring(
        chain=(("author", "to", "paper"), ("paper", "to", "author")),
        rule_type=RuleType.CYCLIC,
        confidence=0.5,
        groundings=3,
        contribution=0.5,
    )

    assert "author_to_paper -> paper_to_author" in firing.describe()


@pytest.mark.parametrize("grounding_weighting", list(GroundingWeighting))
def test_chain_evidence_composes_into_the_score(
    evaluator_graph: HeteroGraph,
    cyclic_rule: Rule,
    grounding_weighting: GroundingWeighting,
) -> None:
    """Per-chain contributions must noisy-or back into the predictor's score."""
    results = [
        (cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3)),
        (subject_grounded(0), RuleMetrics(1, 0.7, 0.33, 3)),
        (object_grounded(1), RuleMetrics(1, 0.6, 0.33, 3)),
    ]
    predictor = RulePredictor(
        evaluator_graph,
        results,
        config=PredictionConfig(grounding_weighting=grounding_weighting),
    )

    for direction, query_ids, score in (
        (QueryDirection.TAIL, range(4), predictor.score_tails),
        (QueryDirection.HEAD, range(3), predictor.score_heads),
    ):
        for query_id in query_ids:
            evidence = predictor.evidence.chain_evidence(direction, query_id)
            composed = 1.0 - torch.prod(1.0 - evidence.contributions, dim=0)
            assert torch.allclose(composed, score(query_id), atol=1e-6)


def test_maximum_scores_strongest_chain_with_noisy_or_tie_break(
    evaluator_graph: HeteroGraph, cyclic_rule: Rule
) -> None:
    """MAXIMUM is the best chain's contribution plus a scaled noisy-or."""
    results = [
        (cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3)),
        (subject_grounded(0), RuleMetrics(1, 0.7, 0.33, 3)),
        (object_grounded(1), RuleMetrics(1, 0.6, 0.33, 3)),
    ]
    predictor = RulePredictor(
        evaluator_graph,
        results,
        config=PredictionConfig(aggregation=ConfidenceAggregation.MAXIMUM),
    )

    for direction, query_ids, score in (
        (QueryDirection.TAIL, range(4), predictor.score_tails),
        (QueryDirection.HEAD, range(3), predictor.score_heads),
    ):
        for query_id in query_ids:
            contributions = predictor.evidence.chain_evidence(
                direction, query_id
            ).contributions
            noisy_or = 1.0 - torch.prod(1.0 - contributions, dim=0)
            expected = contributions.max(dim=0).values + TIE_BREAK_WEIGHT * noisy_or
            assert torch.allclose(expected, score(query_id), atol=1e-6)


@pytest.fixture
def two_chain_graph() -> HeteroGraph:
    """Person 0 reaches city 1 via ``born_in, near`` and city 2 via ``works_in``."""
    data = HeteroData()
    data["person"].num_nodes = 1
    data["city"].num_nodes = 3
    data["person", "born_in", "city"].edge_index = torch.tensor([[0], [0]])
    data["city", "near", "city"].edge_index = torch.tensor([[0], [1]])
    data["person", "works_in", "city"].edge_index = torch.tensor([[0], [2]])
    data["person", "lives_in", "city"].edge_index = torch.tensor([[0], [0]])
    return HeteroGraph(data)


def test_maximum_keeps_one_strong_chain_above_many_weak_ones(
    two_chain_graph: HeteroGraph,
) -> None:
    """Weak chains that noisy-or past a strong one must not outrank it."""
    works_in = Rule(
        head=cyclic_born_in_near().head,
        body=(
            Atom(
                relation="works_in",
                subject=Term.variable("X", node_type="person"),
                object_=Term.variable("Y", node_type="city"),
            ),
        ),
        rule_type=RuleType.CYCLIC,
    )
    results = [
        (cyclic_born_in_near(), metrics_with(confidence=0.4)),
        (object_grounded(1), metrics_with(confidence=0.4)),
        (works_in, metrics_with(confidence=0.6)),
    ]
    noisy_or = RulePredictor(
        two_chain_graph,
        results,
        config=PredictionConfig(grounding_weighting=GroundingWeighting.ONCE),
    ).score_tails(0)
    maximum = RulePredictor(
        two_chain_graph,
        results,
        config=PredictionConfig(
            grounding_weighting=GroundingWeighting.ONCE,
            aggregation=ConfidenceAggregation.MAXIMUM,
        ),
    ).score_tails(0)

    assert noisy_or[1] > noisy_or[2]
    assert maximum[2] > maximum[1]


def test_chain_evidence_rows_follow_chains(
    evaluator_graph: HeteroGraph, cyclic_rule: Rule
) -> None:
    results = [
        (cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3)),
        (subject_grounded(0), RuleMetrics(1, 0.7, 0.33, 3)),
    ]
    predictor = RulePredictor(evaluator_graph, results)

    evidence = predictor.evidence.chain_evidence(QueryDirection.TAIL, 0)

    assert [rule_type for _, rule_type in predictor.evidence.chains] == [
        RuleType.CYCLIC,
        RuleType.AC1,
    ]
    assert evidence.contributions.shape == (2, evaluator_graph.node_count("city"))


def test_chain_evidence_reports_grounding_counts(
    multi_grounding_graph: HeteroGraph,
) -> None:
    """City 2 is reached by two body groundings, city 3 by one, city 0 by none."""
    results = [(cyclic_born_in_near(), RuleMetrics(2, 0.5, 0.5, 4))]
    predictor = RulePredictor(multi_grounding_graph, results)

    groundings = predictor.evidence.chain_evidence(QueryDirection.TAIL, 0).groundings[0]

    assert groundings[2].item() == 2
    assert groundings[3].item() == 1
    assert groundings[0].item() == 0


def test_chain_evidence_is_empty_without_scoring_rules(
    evaluator_graph: HeteroGraph,
) -> None:
    """AC2 rules are ignored, which can leave no chains at all."""
    predictor = RulePredictor(evaluator_graph, [(ac2_born_in(), metrics_with(0.5))])

    evidence = predictor.evidence.chain_evidence(QueryDirection.TAIL, 0)

    assert evidence.contributions.shape == (0, evaluator_graph.node_count("city"))


class TestCorrectedConfidence:
    def test_zero_unseen_negatives_keeps_raw_confidence(self) -> None:
        metrics = RuleMetrics(2, 1.0, 0.5, 2)

        assert corrected_confidence(metrics, 0) == pytest.approx(1.0)

    def test_unseen_negatives_shrink_rules_with_few_predictions(self) -> None:
        few = RuleMetrics(
            support=2, confidence=1.0, head_coverage=0.1, num_predictions=2
        )
        many = RuleMetrics(
            support=90, confidence=0.9, head_coverage=0.1, num_predictions=100
        )

        assert corrected_confidence(few, 5) == pytest.approx(2 / 7)
        assert corrected_confidence(many, 5) > corrected_confidence(few, 5)

    def test_negative_unseen_negatives_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="unseen_negatives"):
            PredictionConfig(unseen_negatives=-1)


def test_unseen_negatives_reorder_candidates(two_chain_graph: HeteroGraph) -> None:
    """A perfect rule with two predictions falls below a broad 0.9 rule."""
    works_in = Rule(
        head=cyclic_born_in_near().head,
        body=(
            Atom(
                relation="works_in",
                subject=Term.variable("X", node_type="person"),
                object_=Term.variable("Y", node_type="city"),
            ),
        ),
        rule_type=RuleType.CYCLIC,
    )
    results = [
        (cyclic_born_in_near(), RuleMetrics(2, 1.0, 0.1, 2)),
        (works_in, RuleMetrics(90, 0.9, 0.1, 100)),
    ]

    def tail_scores(unseen_negatives: int) -> torch.Tensor:
        config = PredictionConfig(
            grounding_weighting=GroundingWeighting.ONCE,
            aggregation=ConfidenceAggregation.MAXIMUM,
            unseen_negatives=unseen_negatives,
        )
        return RulePredictor(two_chain_graph, results, config=config).score_tails(0)

    raw, corrected = tail_scores(0), tail_scores(5)

    assert raw[1] > raw[2]
    assert corrected[2] > corrected[1]


_NUM_CITIES = 3
"""Query ids valid in both directions of ``evaluator_graph``."""


def _memo_results() -> list[tuple[Rule, RuleMetrics]]:
    """A cyclic rule and a subject-grounded AC1 rule over the same chain."""
    return [
        (cyclic_born_in_near(), metrics_with(confidence=0.25)),
        (subject_grounded(0), metrics_with(confidence=0.5)),
    ]


class TestRuleEvidenceMemo:
    @pytest.fixture
    def evidence(self, evaluator_graph: HeteroGraph) -> RuleEvidence:
        return RulePredictor(evaluator_graph, _memo_results()).evidence

    def test_reached_candidates_unites_chain_and_acyclic_evidence(
        self, evidence: RuleEvidence
    ) -> None:
        for direction in QueryDirection:
            for query_id in range(_NUM_CITIES):
                acyclic_ids, _ = evidence.acyclic_firings(direction, query_id)
                chains = evidence.chain_evidence(direction, query_id)
                chain_ids = torch.nonzero(chains.contributions.sum(dim=0) > 0)
                expected = torch.unique(torch.cat([chain_ids.flatten(), acyclic_ids]))

                reached = evidence.reached_candidates(direction, query_id)

                assert reached.tolist() == expected.tolist()

    def test_repeated_query_reuses_its_firings(self, evidence: RuleEvidence) -> None:
        first = evidence.acyclic_firings(QueryDirection.TAIL, 0)

        assert evidence.acyclic_firings(QueryDirection.TAIL, 0) is first

    def test_alternating_queries_match_fresh_evidence(
        self, evaluator_graph: HeteroGraph, evidence: RuleEvidence
    ) -> None:
        queries = [
            (direction, query_id)
            for query_id in range(_NUM_CITIES)
            for direction in QueryDirection
        ]

        for direction, query_id in queries * 2:
            fresh = RulePredictor(evaluator_graph, _memo_results()).evidence
            reached = evidence.reached_candidates(direction, query_id)
            ids, confidences = evidence.acyclic_firings(direction, query_id)
            fresh_ids, fresh_confidences = fresh.acyclic_firings(direction, query_id)

            assert (
                reached.tolist()
                == fresh.reached_candidates(direction, query_id).tolist()
            )
            assert ids.tolist() == fresh_ids.tolist()
            assert confidences.tolist() == fresh_confidences.tolist()


_NUM_QUERIES = {QueryDirection.TAIL: 4, QueryDirection.HEAD: _NUM_CITIES}


def _batch_results() -> list[tuple[Rule, RuleMetrics]]:
    return [
        (cyclic_born_in_near(), metrics_with(confidence=0.25)),
        (subject_grounded(0), metrics_with(confidence=0.5)),
        (object_grounded(1), metrics_with(confidence=0.4)),
    ]


def _per_chain_contributions(
    evidence: RuleEvidence, direction: QueryDirection, query_id: int
) -> list[tuple[int, float, float]]:
    expected: list[tuple[int, float, float]] = []
    for side in evidence.sides(direction):
        candidates, counts = side.fetch(query_id)
        query_confidence = side.confidence_for(query_id) or 0.0
        for candidate, count in zip(candidates.tolist(), counts.tolist(), strict=True):
            weight = math.log2(GROUNDING_WEIGHT_OFFSET + count)
            grounded = side.candidate_confidences.get(candidate, 0.0)
            complement = (1.0 - query_confidence) ** weight * (1.0 - grounded) ** weight
            expected.append((candidate, 1.0 - complement, float(count)))
    return expected


class TestChainContributions:
    @pytest.fixture
    def evidence(self, evaluator_graph: HeteroGraph) -> RuleEvidence:
        config = PredictionConfig(grounding_weighting=GroundingWeighting.LOG_COUNT)
        return RulePredictor(evaluator_graph, _batch_results(), config=config).evidence

    @pytest.mark.parametrize("direction", list(QueryDirection))
    def test_matches_each_chain_computed_alone(
        self, evidence: RuleEvidence, direction: QueryDirection
    ) -> None:
        for query_id in range(_NUM_QUERIES[direction]):
            found = evidence.chain_contributions(direction, query_id)
            expected = _per_chain_contributions(evidence, direction, query_id)

            assert found.candidates.tolist() == [c for c, _, _ in expected]
            assert found.contributions.tolist() == pytest.approx(
                [c for _, c, _ in expected], abs=1e-6
            )
            assert found.groundings.tolist() == [g for _, _, g in expected]
