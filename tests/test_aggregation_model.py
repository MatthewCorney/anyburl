"""Tests for sampling, fitting and applying the learned aggregator."""

import pytest
import torch

from anyburl.aggregation import (
    AggregatedScorer,
    AggregatorConfig,
    CandidateFeature,
    CandidateFeaturizer,
    CandidateSampling,
    CandidateScope,
    KnownAnswers,
    LinearAggregator,
    RankingQuery,
    TrainingSet,
    build_training_set,
    column_of,
    ranking_queries,
)
from anyburl.core import QueryDirection, Triple
from anyburl.exceptions import ConfigurationError
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleMetrics
from anyburl.prediction import RulePredictor
from anyburl.rule import Rule
from anyburl.stratification import relation_degrees

LIVES_IN = ("person", "lives_in", "city")
NUM_QUERIES = 40
NUM_CANDIDATES = 6
SIGNAL_COLUMN = 0
NOISE_COLUMN = 1
REACHED_SAMPLING = CandidateSampling(
    max_rule_candidates=5, num_random_candidates=5, scope=CandidateScope.REACHED
)


@pytest.fixture
def featurizer(evaluator_graph: HeteroGraph, cyclic_rule: Rule) -> CandidateFeaturizer:
    predictor = RulePredictor(
        evaluator_graph, [(cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3))]
    )
    return CandidateFeaturizer(
        predictor.evidence, relation_degrees(evaluator_graph, LIVES_IN)
    )


@pytest.fixture
def separable_set() -> TrainingSet:
    """The answer always has the largest signal; the noise column is random."""
    generator = torch.Generator().manual_seed(0)
    features = torch.zeros(NUM_QUERIES, NUM_CANDIDATES, 3)
    features[:, :, SIGNAL_COLUMN] = torch.rand(
        NUM_QUERIES, NUM_CANDIDATES, generator=generator
    )
    features[:, 0, SIGNAL_COLUMN] += 1.0
    features[:, :, NOISE_COLUMN] = torch.rand(
        NUM_QUERIES, NUM_CANDIDATES, generator=generator
    )
    return TrainingSet(
        features=features,
        mask=torch.ones(NUM_QUERIES, NUM_CANDIDATES, dtype=torch.bool),
    )


class TestRankingQueries:
    def test_yields_tail_then_head_query_per_triple(self) -> None:
        triple = Triple(
            head_id=1, tail_id=2, head_type="person", tail_type="city", relation="r"
        )

        assert ranking_queries([triple]) == (
            RankingQuery(QueryDirection.TAIL, 1, 2),
            RankingQuery(QueryDirection.HEAD, 2, 1),
        )


class TestKnownAnswers:
    def test_indexes_both_directions(self) -> None:
        known = KnownAnswers.from_pairs([(0, 1), (0, 2), (3, 1)])

        assert known.of(RankingQuery(QueryDirection.TAIL, 0, 1)) == {1, 2}
        assert known.of(RankingQuery(QueryDirection.HEAD, 1, 0)) == {0, 3}
        assert known.of(RankingQuery(QueryDirection.TAIL, 9, 1)) == frozenset()


class TestBuildTrainingSet:
    def test_puts_target_first(self, featurizer: CandidateFeaturizer) -> None:
        """Person 0 reaches city 1 via born_in(0,0), near(0,1)."""
        query = RankingQuery(QueryDirection.TAIL, 0, 2)

        training_set = build_training_set(
            featurizer, [query], KnownAnswers.from_pairs([]), CandidateSampling()
        )

        full = featurizer.featurize(QueryDirection.TAIL, 0)
        assert torch.equal(training_set.features[0, 0], full[2])

    def test_never_samples_known_answers(self, featurizer: CandidateFeaturizer) -> None:
        """City 1 is the only rule-reached candidate, but it is a known answer."""
        query = RankingQuery(QueryDirection.TAIL, 0, 2)
        known = KnownAnswers.from_pairs([(0, 1)])

        training_set = build_training_set(
            featurizer, [query], known, CandidateSampling()
        )

        full = featurizer.featurize(QueryDirection.TAIL, 0)
        kept = training_set.features[0][training_set.mask[0]]
        assert kept.size(0) == 2
        assert not any(torch.equal(row, full[1]) for row in kept[1:])

    def test_prefers_rule_scored_negatives(
        self, featurizer: CandidateFeaturizer
    ) -> None:
        query = RankingQuery(QueryDirection.TAIL, 0, 2)
        sampling = CandidateSampling(max_rule_candidates=1, num_random_candidates=0)

        training_set = build_training_set(
            featurizer, [query], KnownAnswers.from_pairs([]), sampling
        )

        score = training_set.features[0, 1, column_of(CandidateFeature.RULE_SCORE)]
        assert training_set.features.size(1) == 2
        assert score.item() > 0.0

    def test_pads_shorter_queries(self, featurizer: CandidateFeaturizer) -> None:
        """Person 0 has one rule-reached negative; city 0 has three persons."""
        queries = [
            RankingQuery(QueryDirection.TAIL, 0, 2),
            RankingQuery(QueryDirection.HEAD, 0, 1),
        ]
        sampling = CandidateSampling(max_rule_candidates=5, num_random_candidates=5)

        training_set = build_training_set(
            featurizer, queries, KnownAnswers.from_pairs([]), sampling
        )

        assert training_set.mask.sum(dim=1).tolist() == [3, 4]

    def test_empty_queries_raise(self, featurizer: CandidateFeaturizer) -> None:
        with pytest.raises(ConfigurationError, match="must not be empty"):
            build_training_set(
                featurizer, [], KnownAnswers.from_pairs([]), CandidateSampling()
            )

    def test_reached_scope_drops_queries_the_rules_miss(
        self, featurizer: CandidateFeaturizer
    ) -> None:
        """Person 0's rules reach city 1 only."""
        queries = [
            RankingQuery(QueryDirection.TAIL, 0, 2),
            RankingQuery(QueryDirection.TAIL, 0, 1),
        ]

        training_set = build_training_set(
            featurizer, queries, KnownAnswers.from_pairs([]), REACHED_SAMPLING
        )

        full = featurizer.featurize(QueryDirection.TAIL, 0)
        assert training_set.features.size(0) == 1
        assert torch.equal(training_set.features[0, 0], full[1])

    def test_reached_scope_samples_only_reached_negatives(
        self, featurizer: CandidateFeaturizer
    ) -> None:
        """City 1's rules reach persons 0 and 2."""
        query = RankingQuery(QueryDirection.HEAD, 1, 0)

        training_set = build_training_set(
            featurizer, [query], KnownAnswers.from_pairs([]), REACHED_SAMPLING
        )

        scores = training_set.features[0, :, column_of(CandidateFeature.RULE_SCORE)]
        assert training_set.mask.sum().item() == 2
        assert bool((scores > 0).all())

    def test_reached_scope_may_keep_no_query(
        self, featurizer: CandidateFeaturizer
    ) -> None:
        query = RankingQuery(QueryDirection.TAIL, 0, 2)

        training_set = build_training_set(
            featurizer, [query], KnownAnswers.from_pairs([]), REACHED_SAMPLING
        )

        assert training_set.features.size(0) == 0
        assert training_set.features.size(2) == len(featurizer.feature_names())

    @pytest.mark.parametrize(("rule", "random"), [(0, 0), (-1, 5), (5, -1)])
    def test_invalid_sampling_raises(self, rule: int, random: int) -> None:
        with pytest.raises(ConfigurationError):
            CandidateSampling(max_rule_candidates=rule, num_random_candidates=random)


class TestLinearAggregator:
    def test_learns_to_rank_the_answer_first(self, separable_set: TrainingSet) -> None:
        model = LinearAggregator.fit(separable_set, AggregatorConfig())

        scores = model.score(separable_set.features)

        assert bool((scores.argmax(dim=1) == 0).all())

    def test_weights_the_signal_above_noise(self, separable_set: TrainingSet) -> None:
        model = LinearAggregator.fit(separable_set, AggregatorConfig())

        assert model.weights[SIGNAL_COLUMN] > 0
        assert model.weights[SIGNAL_COLUMN] > abs(model.weights[NOISE_COLUMN]) * 5

    def test_constant_feature_gets_no_weight(self, separable_set: TrainingSet) -> None:
        """Column 2 is all zeros; it must not produce NaN or a weight."""
        model = LinearAggregator.fit(separable_set, AggregatorConfig())

        assert not bool(torch.isnan(model.weights).any())
        assert model.weights[2].item() == pytest.approx(0.0, abs=1e-6)

    def test_ignores_padding(self, separable_set: TrainingSet) -> None:
        """A padded row with a huge signal must not distort the fit."""
        features = separable_set.features.clone()
        mask = separable_set.mask.clone()
        features[:, -1, SIGNAL_COLUMN] = 100.0
        mask[:, -1] = False

        model = LinearAggregator.fit(TrainingSet(features, mask), AggregatorConfig())

        assert bool((model.score(features[:, :-1]).argmax(dim=1) == 0).all())

    def test_fits_a_single_candidate_row(self) -> None:
        single = TrainingSet(torch.ones(1, 1, 3), torch.ones(1, 1, dtype=torch.bool))

        model = LinearAggregator.fit(single, AggregatorConfig())

        assert model.scale.tolist() == [1.0, 1.0, 1.0]

    def test_rejects_wrong_feature_count(self, separable_set: TrainingSet) -> None:
        model = LinearAggregator.fit(separable_set, AggregatorConfig())

        with pytest.raises(ConfigurationError, match="expected 3 features"):
            model.score(torch.zeros(4, 2))

    @pytest.mark.parametrize(("l2", "iterations"), [(-0.1, 10), (0.0, 0)])
    def test_invalid_config_raises(self, l2: float, iterations: int) -> None:
        with pytest.raises(ConfigurationError):
            AggregatorConfig(l2_penalty=l2, max_iterations=iterations)


class TestAggregatedScorer:
    def test_scores_every_candidate(self, featurizer: CandidateFeaturizer) -> None:
        queries = ranking_queries(
            [
                Triple(
                    head_id=0,
                    tail_id=1,
                    head_type="person",
                    tail_type="city",
                    relation="lives_in",
                )
            ]
        )
        training_set = build_training_set(
            featurizer, queries, KnownAnswers.from_pairs([]), CandidateSampling()
        )
        scorer = AggregatedScorer(
            featurizer, LinearAggregator.fit(training_set, AggregatorConfig())
        )

        assert scorer.score_tails(0).shape == (3,)
        assert scorer.score_heads(1).shape == (4,)

    def test_rejects_mismatched_model(self, featurizer: CandidateFeaturizer) -> None:
        model = LinearAggregator(
            weights=torch.zeros(2), mean=torch.zeros(2), scale=torch.ones(2)
        )

        with pytest.raises(ConfigurationError, match="fitted on 2"):
            AggregatedScorer(featurizer, model)

    def test_reached_scope_ranks_unreached_candidates_last(
        self, featurizer: CandidateFeaturizer
    ) -> None:
        """The model prefers low rule scores, but only persons 0, 2 are reached."""
        num_features = len(featurizer.feature_names())
        weights = torch.zeros(num_features)
        weights[column_of(CandidateFeature.RULE_SCORE)] = -1.0
        model = LinearAggregator(
            weights=weights,
            mean=torch.zeros(num_features),
            scale=torch.ones(num_features),
        )

        everywhere = AggregatedScorer(featurizer, model).score_heads(1)
        reached = AggregatedScorer(
            featurizer, model, scope=CandidateScope.REACHED
        ).score_heads(1)

        assert everywhere[1] > everywhere[0]
        assert reached[1] == reached[3]
        assert reached[1] < min(reached[0], reached[2])
        assert torch.equal(reached[[0, 2]], everywhere[[0, 2]])
