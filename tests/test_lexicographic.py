"""Tests for lexicographic ranking of rule contributions."""

import pytest
import torch

from anyburl.graph import HeteroGraph
from anyburl.prediction import ConfidenceAggregation, PredictionConfig, RulePredictor
from anyburl.prediction.lexicographic import lexicographic_scores
from tests.rules import cyclic_born_in_near, metrics_with, subject_grounded

NUM_CANDIDATES = 5
_CLOSE = 1e-6


def _scores(entries: list[tuple[int, float]], *, depth: int = 10) -> list[float]:
    candidates = torch.tensor([candidate for candidate, _ in entries], dtype=torch.long)
    values = torch.tensor([value for _, value in entries])
    return lexicographic_scores(
        candidates, values, NUM_CANDIDATES, depth=depth
    ).tolist()


class TestLexicographicScores:
    def test_strongest_contribution_decides(self) -> None:
        scores = _scores([(0, 0.9), (1, 0.5), (1, 0.5), (1, 0.5)])

        assert scores[0] > scores[1]

    def test_next_strongest_breaks_ties(self) -> None:
        scores = _scores([(0, 0.8), (0, 0.2), (1, 0.8), (1, 0.6)])

        assert scores[1] > scores[0]

    def test_more_contributions_break_ties(self) -> None:
        scores = _scores([(0, 0.8), (1, 0.8), (1, 0.1)])

        assert scores[1] > scores[0]

    def test_equal_lists_tie(self) -> None:
        scores = _scores([(0, 0.7), (0, 0.3), (1, 0.3), (1, 0.7)])

        assert scores[0] == scores[1]

    def test_contributions_past_the_depth_are_ignored(self) -> None:
        scores = _scores([(0, 0.8), (0, 0.1), (1, 0.8), (1, 0.5)], depth=1)

        assert scores[0] == scores[1]

    def test_unreached_candidates_score_zero(self) -> None:
        scores = _scores([(2, 0.4), (3, 0.0)])

        assert [scores[0], scores[1], scores[3], scores[4]] == [0.0] * 4
        assert scores[2] > 0.0

    def test_scores_stay_close_to_the_strongest_contribution(self) -> None:
        scores = _scores([(0, 0.6), (0, 0.5), (1, 0.6), (2, 0.6), (2, 0.4)])

        assert scores == pytest.approx([0.6, 0.6, 0.6, 0.0, 0.0], abs=_CLOSE)

    def test_offsets_never_overtake_a_stronger_value(self) -> None:
        stronger = float(torch.tensor(0.5).nextafter(torch.tensor(1.0)))
        scores = _scores([(0, 0.5), (0, 0.4), (0, 0.3), (1, stronger)])

        assert scores[1] > scores[0]

    def test_no_contributions_score_zero(self) -> None:
        assert _scores([]) == [0.0] * NUM_CANDIDATES


class TestLexicographicAggregation:
    def test_scores_match_the_strongest_rule(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        results = [
            (cyclic_born_in_near(), metrics_with(confidence=0.25)),
            (subject_grounded(0), metrics_with(confidence=0.5)),
        ]
        lexicographic = RulePredictor(
            evaluator_graph,
            results,
            config=PredictionConfig(aggregation=ConfidenceAggregation.LEXICOGRAPHIC),
        )
        maximum = RulePredictor(
            evaluator_graph,
            results,
            config=PredictionConfig(aggregation=ConfidenceAggregation.MAXIMUM),
        )

        for head_id in range(evaluator_graph.node_count("person")):
            assert lexicographic.score_tails(head_id).tolist() == pytest.approx(
                maximum.score_tails(head_id).tolist(), abs=1e-3
            )
