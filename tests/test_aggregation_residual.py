"""Tests for offset fitting and per-relation corrections to a shared model."""

import pytest
import torch

from anyburl.aggregation import (
    AggregatedScorer,
    AggregatorConfig,
    CandidateFeaturizer,
    CandidateScope,
    JointFeaturizer,
    LinearAggregator,
    ResidualAggregator,
    SharedFeature,
    SharedFeaturizer,
    TrainingSet,
)
from anyburl.core import QueryDirection
from anyburl.graph import HeteroGraph
from anyburl.prediction import RulePredictor
from anyburl.rule import Rule, RuleMetrics
from anyburl.rule_evaluation import RuleSet
from anyburl.stratification import relation_degrees

LIVES_IN = ("person", "lives_in", "city")
NUM_QUERIES = 40
NUM_CANDIDATES = 5
NUM_BASE = 2
SIGNAL = 2.0


@pytest.fixture
def joint(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> JointFeaturizer:
    predictor = RulePredictor(
        evaluator_graph, RuleSet.of([(cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3))])
    )
    degrees = relation_degrees(evaluator_graph, LIVES_IN)
    return JointFeaturizer(
        SharedFeaturizer(predictor.evidence, degrees),
        CandidateFeaturizer(predictor.evidence, degrees),
    )


@pytest.fixture
def residual_set() -> TrainingSet:
    """Base columns are noise; the answer stands out in extra column 0."""
    generator = torch.Generator().manual_seed(0)
    features = torch.rand(
        NUM_QUERIES, NUM_CANDIDATES, NUM_BASE + 2, generator=generator
    )
    features[:, 0, NUM_BASE] += SIGNAL
    return TrainingSet(
        features=features,
        mask=torch.ones(NUM_QUERIES, NUM_CANDIDATES, dtype=torch.bool),
    )


def zero_model(num_features: int) -> LinearAggregator:
    """A model that scores every candidate zero."""
    return LinearAggregator(
        weights=torch.zeros(num_features),
        mean=torch.zeros(num_features),
        scale=torch.ones(num_features),
    )


class TestOffsetFit:
    def test_offsets_that_already_rank_need_smaller_weights(
        self, residual_set: TrainingSet
    ) -> None:
        offsets = torch.zeros(NUM_QUERIES, NUM_CANDIDATES)
        offsets[:, 0] = 10.0

        plain = LinearAggregator.fit(residual_set, AggregatorConfig())
        offset = LinearAggregator.fit(residual_set, AggregatorConfig(), offsets=offsets)

        assert offset.weights.norm() < plain.weights.norm()


class TestJointFeaturizer:
    def test_appends_extra_columns_after_base(self, joint: JointFeaturizer) -> None:
        names = joint.feature_names()
        features = joint.featurize(QueryDirection.TAIL, 0)

        assert features.size(1) == len(names)
        assert names[: joint.num_base_features] == tuple(
            feature.value for feature in SharedFeature
        )

    def test_restricted_rows_match_full_rows(self, joint: JointFeaturizer) -> None:
        candidates = joint.reached(QueryDirection.HEAD, 1)

        restricted = joint.featurize(QueryDirection.HEAD, 1, candidates=candidates)

        full = joint.featurize(QueryDirection.HEAD, 1)
        assert torch.equal(restricted, full[candidates])


class TestResidualAggregator:
    def test_correction_learns_what_the_base_misses(
        self, residual_set: TrainingSet
    ) -> None:
        model = ResidualAggregator.fit(
            residual_set, zero_model(NUM_BASE), AggregatorConfig()
        )

        scores = model.score(residual_set.features)

        assert bool((scores.argmax(dim=1) == 0).all())
        assert model.num_features == NUM_BASE + 2

    def test_score_is_base_plus_correction(self, residual_set: TrainingSet) -> None:
        base = LinearAggregator.fit(
            TrainingSet(residual_set.features[..., :NUM_BASE], residual_set.mask),
            AggregatorConfig(),
        )
        model = ResidualAggregator.fit(residual_set, base, AggregatorConfig())
        features = residual_set.features[0]

        expected = base.score(features[:, :NUM_BASE]) + model.residual.score(
            features[:, NUM_BASE:]
        )
        assert torch.allclose(model.score(features), expected)

    def test_scores_through_an_aggregated_scorer(self, joint: JointFeaturizer) -> None:
        num_base = joint.num_base_features
        num_extra = len(joint.feature_names()) - num_base
        model = ResidualAggregator(zero_model(num_base), zero_model(num_extra))

        scorer = AggregatedScorer(joint, model, scope=CandidateScope.REACHED)

        assert scorer.score_heads(1).shape == (4,)
