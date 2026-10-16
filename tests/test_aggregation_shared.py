"""Tests for acyclic evidence, relation-independent features and set stacking."""

import pytest
import torch

from anyburl.aggregation import (
    SharedFeature,
    SharedFeaturizer,
    TrainingSet,
    concat_training_sets,
)
from anyburl.core import QueryDirection
from anyburl.exceptions import ConfigurationError
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleMetrics
from anyburl.prediction import GroundingWeighting, PredictionConfig, RulePredictor
from anyburl.rule import Rule
from anyburl.stratification import relation_degrees
from tests.rules import object_constant_rule

LIVES_IN = ("person", "lives_in", "city")
PLAIN = PredictionConfig(grounding_weighting=GroundingWeighting.ONCE)


def column(features: torch.Tensor, feature: SharedFeature) -> torch.Tensor:
    """Return one named column of a shared feature matrix."""
    return features[:, list(SharedFeature).index(feature)]


class TestAcyclicEvidence:
    def test_summarises_rules_firing_per_candidate(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        results = [
            (object_constant_rule(0, 0), RuleMetrics(1, 0.5, 0.25, 2)),
            (object_constant_rule(0, None), RuleMetrics(2, 0.5, 0.5, 4)),
        ]
        predictor = RulePredictor(evaluator_graph, results)

        evidence = predictor.evidence.acyclic_evidence(QueryDirection.TAIL, 2)

        assert evidence.max_confidence.tolist() == pytest.approx([0.5, 0.0, 0.0])
        assert evidence.noisy_or.tolist() == pytest.approx([0.75, 0.0, 0.0])
        assert evidence.firings.tolist() == [2.0, 0.0, 0.0]

    def test_is_empty_without_acyclic_rules(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        predictor = RulePredictor(
            evaluator_graph, [(cyclic_rule, RuleMetrics(1, 0.25, 0.25, 4))]
        )

        evidence = predictor.evidence.acyclic_evidence(QueryDirection.HEAD, 0)

        assert evidence.firings.sum().item() == 0.0


class TestSharedFeaturizer:
    @pytest.fixture
    def features(self, evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> torch.Tensor:
        """Features of person 2's tails: cyclic reaches city 1, acyclic city 0."""
        results = [
            (cyclic_rule, RuleMetrics(1, 0.25, 0.25, 4)),
            (object_constant_rule(0, 0), RuleMetrics(1, 0.5, 0.25, 2)),
        ]
        predictor = RulePredictor(evaluator_graph, results, config=PLAIN)
        featurizer = SharedFeaturizer(
            predictor.evidence, relation_degrees(evaluator_graph, LIVES_IN)
        )
        return featurizer.featurize(QueryDirection.TAIL, 2)

    def test_has_one_column_per_feature(self, features: torch.Tensor) -> None:
        assert features.shape == (3, len(SharedFeature))

    def test_separates_evidence_by_rule_kind(self, features: torch.Tensor) -> None:
        assert column(features, SharedFeature.MAX_CYCLIC).tolist() == pytest.approx(
            [0.0, 0.25, 0.0]
        )
        assert column(features, SharedFeature.MAX_ACYCLIC).tolist() == pytest.approx(
            [0.5, 0.0, 0.0]
        )
        assert column(features, SharedFeature.MAX_AC1).sum().item() == 0.0

    def test_rule_score_combines_chains_and_acyclic_rules(
        self, features: torch.Tensor
    ) -> None:
        assert column(features, SharedFeature.RULE_SCORE).tolist() == pytest.approx(
            [0.5, 0.25, 0.0]
        )
        assert column(
            features, SharedFeature.MAX_CONTRIBUTION
        ).tolist() == pytest.approx([0.5, 0.25, 0.0])

    def test_second_contribution_is_zero_with_one_firing(
        self, features: torch.Tensor
    ) -> None:
        assert column(features, SharedFeature.SECOND_CONTRIBUTION).sum().item() == 0.0

    def test_score_column_names_the_rule_score(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        predictor = RulePredictor(
            evaluator_graph, [(cyclic_rule, RuleMetrics(1, 0.25, 0.25, 4))]
        )
        featurizer = SharedFeaturizer(
            predictor.evidence, relation_degrees(evaluator_graph, LIVES_IN)
        )

        name = featurizer.feature_names()[featurizer.score_column]

        assert name == SharedFeature.RULE_SCORE.value


class TestConcatTrainingSets:
    def test_pads_queries_to_the_widest_set(self) -> None:
        narrow = TrainingSet(torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool))
        wide = TrainingSet(torch.ones(2, 4, 3), torch.ones(2, 4, dtype=torch.bool))

        combined = concat_training_sets([narrow, wide])

        assert combined.features.shape == (3, 4, 3)
        assert combined.mask.sum(dim=1).tolist() == [2, 4, 4]

    def test_drops_padding_of_the_inputs(self) -> None:
        mask = torch.tensor([[True, False]])
        padded = TrainingSet(torch.ones(1, 2, 3), mask)

        combined = concat_training_sets([padded])

        assert combined.features.shape == (1, 1, 3)

    def test_mismatched_feature_counts_raise(self) -> None:
        first = TrainingSet(torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool))
        second = TrainingSet(torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool))

        with pytest.raises(ConfigurationError, match="feature counts"):
            concat_training_sets([first, second])
