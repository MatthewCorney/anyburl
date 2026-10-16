"""Tests for acyclic rules: sampling, generalization, evaluation, prediction."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.acyclic import (
    AcyclicStep,
    AcyclicStepSampler,
    WalkAnchor,
    generalize_acyclic,
)
from anyburl.core import Triple
from anyburl.exceptions import ConfigurationError
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleEvaluator, RuleMetrics
from anyburl.pipeline import AnyBURL, AnyBURLConfig
from anyburl.prediction import RulePredictor
from anyburl.rule import Rule, RuleConfig, RuleType, Term
from tests.rules import object_constant_rule, subject_constant_near_rule

LIVES_IN = ("person", "lives_in", "city")
BORN_IN = ("person", "born_in", "city")
NEAR = ("city", "near", "city")


class TestGeneralizeAcyclic:
    def test_head_anchor_grounds_the_tail(self) -> None:
        step = AcyclicStep(LIVES_IN, WalkAnchor.HEAD, 0, BORN_IN, 2)

        rules = generalize_acyclic(step)

        assert [str(rule) for rule in rules] == [
            str(object_constant_rule(0, 2)),
            str(object_constant_rule(0, None)),
        ]
        assert all(rule.rule_type is RuleType.ACYCLIC for rule in rules)

    def test_tail_anchor_grounds_the_head(self) -> None:
        step = AcyclicStep(LIVES_IN, WalkAnchor.TAIL, 1, NEAR, 2)

        rules = generalize_acyclic(step)

        assert rules[-1] == subject_constant_near_rule(1)
        assert rules[0].body[0].object_ == Term.constant(2, node_type="city")

    def test_drops_body_equal_to_head(self) -> None:
        step = AcyclicStep(LIVES_IN, WalkAnchor.HEAD, 0, LIVES_IN, 0)

        rules = generalize_acyclic(step)

        assert len(rules) == 1
        assert rules[0].body[0].object_.name == "A"


class TestAcyclicStepSampler:
    def test_steps_leave_their_anchor_along_real_edges(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        triple = Triple(3, 0, "person", "city", "lives_in")
        steps = AcyclicStepSampler(evaluator_graph, seed=0).sample(triple, 50)

        assert {step.anchor for step in steps} == set(WalkAnchor)
        for step in steps:
            source = 3 if step.anchor is WalkAnchor.HEAD else 0
            neighbors = evaluator_graph.get_neighbors(source, step.body_edge_type)
            assert step.neighbor_id in neighbors.tolist()

    def test_never_retraces_the_triple(self, evaluator_graph: HeteroGraph) -> None:
        triple = Triple(0, 0, "person", "city", "lives_in")
        steps = AcyclicStepSampler(evaluator_graph, seed=0).sample(triple, 200)

        assert all(
            not (step.body_edge_type == LIVES_IN and step.neighbor_id == 0)
            for step in steps
            if step.anchor is WalkAnchor.HEAD
        )

    def test_same_seed_same_steps(self, evaluator_graph: HeteroGraph) -> None:
        triple = Triple(3, 0, "person", "city", "lives_in")

        first = AcyclicStepSampler(evaluator_graph, seed=7).sample(triple, 10)
        second = AcyclicStepSampler(evaluator_graph, seed=7).sample(triple, 10)

        assert first == second


class TestAcyclicEvaluation:
    @pytest.mark.parametrize(
        ("rule", "expected"),
        [
            (object_constant_rule(0, 0), RuleMetrics(1, 0.5, 0.25, 2)),
            (object_constant_rule(0, None), RuleMetrics(2, 0.5, 0.5, 4)),
            (subject_constant_near_rule(1), RuleMetrics(1, 1 / 3, 0.25, 3)),
        ],
    )
    def test_counts_predictions_and_support(
        self, evaluator_graph: HeteroGraph, rule: Rule, expected: RuleMetrics
    ) -> None:
        metrics = RuleEvaluator(evaluator_graph, RuleConfig()).evaluate(rule)

        assert metrics.support == expected.support
        assert metrics.num_predictions == expected.num_predictions
        assert metrics.confidence == pytest.approx(expected.confidence)
        assert metrics.head_coverage == pytest.approx(expected.head_coverage)


class TestAcyclicPrediction:
    def test_rule_scores_its_constant_for_matching_heads(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        results = [(object_constant_rule(0, 0), RuleMetrics(1, 0.5, 0.25, 2))]
        predictor = RulePredictor(evaluator_graph, results)

        assert predictor.score_tails(2).tolist() == pytest.approx([0.5, 0.0, 0.0])
        assert predictor.score_tails(1).tolist() == pytest.approx([0.0, 0.0, 0.0])

    def test_rule_scores_matching_heads_for_its_constant(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        results = [(object_constant_rule(0, 0), RuleMetrics(1, 0.5, 0.25, 2))]
        predictor = RulePredictor(evaluator_graph, results)

        assert predictor.score_heads(0).tolist() == pytest.approx([0.5, 0, 0.5, 0])
        assert predictor.score_heads(1).tolist() == pytest.approx([0.0] * 4)

    def test_subject_constant_rule_fires_in_both_directions(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        results = [(subject_constant_near_rule(1), RuleMetrics(1, 0.4, 0.25, 3))]
        predictor = RulePredictor(evaluator_graph, results)

        assert predictor.score_tails(1).tolist() == pytest.approx([0.4] * 3)
        assert predictor.score_heads(2).tolist() == pytest.approx([0, 0.4, 0, 0])

    def test_firings_noisy_or_with_each_other(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        results = [
            (object_constant_rule(0, 0), RuleMetrics(1, 0.5, 0.25, 2)),
            (object_constant_rule(0, None), RuleMetrics(2, 0.5, 0.5, 4)),
        ]
        predictor = RulePredictor(evaluator_graph, results)

        assert predictor.score_tails(2)[0].item() == pytest.approx(0.75)


@pytest.fixture
def acyclic_hetero_data() -> HeteroData:
    """Every person born in city 0 lives in city 1, whatever else holds."""
    data = HeteroData()
    data["person"].num_nodes = 6
    data["city"].num_nodes = 3
    data["person", "born_in", "city"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 0, 0, 2, 2, 2]]
    )
    data["person", "lives_in", "city"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 1, 1, 2]]
    )
    return data


class TestPipelineAcyclic:
    def test_learns_acyclic_rules_when_enabled(
        self, acyclic_hetero_data: HeteroData
    ) -> None:
        config = AnyBURLConfig(
            sample_size=4,
            target_edge_type=LIVES_IN,
            acyclic_samples_per_end=10,
        )

        pipeline = AnyBURL(config).fit(acyclic_hetero_data)

        assert object_constant_rule(1, 0) in {rule for rule, _ in pipeline.results}

    def test_learns_no_acyclic_rules_by_default(
        self, acyclic_hetero_data: HeteroData
    ) -> None:
        config = AnyBURLConfig(sample_size=4, target_edge_type=LIVES_IN)

        pipeline = AnyBURL(config).fit(acyclic_hetero_data)

        assert all(rule.rule_type is not RuleType.ACYCLIC for rule in pipeline.rules)

    def test_negative_samples_raise(self) -> None:
        with pytest.raises(ConfigurationError, match="acyclic_samples_per_end"):
            AnyBURLConfig(acyclic_samples_per_end=-1)
