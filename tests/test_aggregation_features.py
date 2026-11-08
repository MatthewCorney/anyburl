"""Tests for per-candidate aggregation features."""

import math

import pytest
import torch

from anyburl.aggregation import CandidateFeature, CandidateFeaturizer, build_features
from anyburl.aggregation.features import LOG_SCORE_FLOOR
from anyburl.core import QueryDirection
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleMetrics
from anyburl.prediction import ChainEvidence, RulePredictor
from anyburl.rule import Rule
from anyburl.stratification import relation_degrees
from tests.rules import subject_grounded

LIVES_IN = ("person", "lives_in", "city")
NUM_SUMMARY = len(CandidateFeature)


def _column(feature: CandidateFeature) -> int:
    return list(CandidateFeature).index(feature)


class TestBuildFeatures:
    @pytest.fixture
    def features(self) -> torch.Tensor:
        """Two chains over three candidates; candidate 2 is unreached."""
        evidence = ChainEvidence(
            contributions=torch.tensor([[0.5, 0.2, 0.0], [0.5, 0.0, 0.0]]),
            groundings=torch.tensor([[3.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        return build_features(evidence, torch.tensor([0.0, 4.0, 9.0]))

    def test_has_summary_then_chain_columns(self, features: torch.Tensor) -> None:
        assert features.shape == (3, NUM_SUMMARY + 2)
        assert torch.equal(features[:, NUM_SUMMARY], torch.tensor([0.5, 0.2, 0.0]))

    def test_rule_score_is_noisy_or_of_chains(self, features: torch.Tensor) -> None:
        score = features[:, _column(CandidateFeature.RULE_SCORE)]

        assert torch.allclose(score, torch.tensor([0.75, 0.2, 0.0]))

    def test_unreached_candidate_has_finite_log_score(
        self, features: torch.Tensor
    ) -> None:
        log_score = features[2, _column(CandidateFeature.LOG_RULE_SCORE)].item()

        assert log_score == pytest.approx(math.log(LOG_SCORE_FLOOR))

    def test_evidence_summaries(self, features: torch.Tensor) -> None:
        assert features[:, _column(CandidateFeature.FIRING_CHAINS)].tolist() == [
            2.0,
            1.0,
            0.0,
        ]
        assert features[:, _column(CandidateFeature.MAX_CONTRIBUTION)].tolist() == [
            0.5,
            pytest.approx(0.2),
            0.0,
        ]
        assert features[0, _column(CandidateFeature.LOG_GROUNDINGS)].item() == (
            pytest.approx(math.log1p(4.0))
        )

    def test_degree_features(self, features: torch.Tensor) -> None:
        assert features[:, _column(CandidateFeature.IS_UNSEEN)].tolist() == [
            1.0,
            0.0,
            0.0,
        ]
        assert features[1, _column(CandidateFeature.LOG_DEGREE)].item() == (
            pytest.approx(math.log1p(4.0))
        )

    def test_no_chains_gives_summary_only(self) -> None:
        evidence = ChainEvidence(
            contributions=torch.zeros(0, 2), groundings=torch.zeros(0, 2)
        )

        features = build_features(evidence, torch.tensor([1.0, 0.0]))

        assert features.shape == (2, NUM_SUMMARY)
        assert features[:, _column(CandidateFeature.RULE_SCORE)].tolist() == [0, 0]


class TestCandidateFeaturizer:
    @pytest.fixture
    def predictor(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> RulePredictor:
        results = [
            (cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3)),
            (subject_grounded(0), RuleMetrics(1, 0.7, 0.33, 3)),
        ]
        return RulePredictor(evaluator_graph, results)

    @pytest.fixture
    def featurizer(
        self, evaluator_graph: HeteroGraph, predictor: RulePredictor
    ) -> CandidateFeaturizer:
        return CandidateFeaturizer(
            predictor.evidence, relation_degrees(evaluator_graph, LIVES_IN)
        )

    def test_rule_score_column_matches_predictor(
        self, predictor: RulePredictor, featurizer: CandidateFeaturizer
    ) -> None:
        column = _column(CandidateFeature.RULE_SCORE)

        for person_id in range(4):
            features = featurizer.featurize(QueryDirection.TAIL, person_id)
            assert torch.allclose(features[:, column], predictor.score_tails(person_id))
        for city_id in range(3):
            features = featurizer.featurize(QueryDirection.HEAD, city_id)
            assert torch.allclose(features[:, column], predictor.score_heads(city_id))

    def test_degrees_follow_the_ranked_side(
        self, featurizer: CandidateFeaturizer
    ) -> None:
        """Tail queries rank cities, so they carry city in-degrees."""
        column = _column(CandidateFeature.LOG_DEGREE)

        cities = featurizer.featurize(QueryDirection.TAIL, 0)[:, column]
        persons = featurizer.featurize(QueryDirection.HEAD, 0)[:, column]

        assert cities.tolist() == pytest.approx([math.log1p(d) for d in (2, 1, 1)])
        assert persons.tolist() == pytest.approx([math.log1p(1.0)] * 4)

    def test_names_match_columns(self, featurizer: CandidateFeaturizer) -> None:
        names = featurizer.feature_names()

        assert len(names) == featurizer.featurize(QueryDirection.TAIL, 0).size(1)
        assert names[0] == CandidateFeature.RULE_SCORE.value
        assert names[NUM_SUMMARY] == (
            "chain[cyclic]: person_born_in_city -> city_near_city"
        )

    def test_restricted_features_match_full_rows(
        self, featurizer: CandidateFeaturizer
    ) -> None:
        candidates = torch.tensor([2, 0])

        restricted = featurizer.featurize(QueryDirection.HEAD, 1, candidates=candidates)

        full = featurizer.featurize(QueryDirection.HEAD, 1)
        assert torch.equal(restricted, full[candidates])

    def test_reached_are_the_positively_scored_candidates(
        self, predictor: RulePredictor, featurizer: CandidateFeaturizer
    ) -> None:
        for city_id in range(3):
            expected = torch.nonzero(predictor.score_heads(city_id) > 0).flatten()
            reached = featurizer.reached(QueryDirection.HEAD, city_id)
            assert reached.tolist() == expected.tolist()
