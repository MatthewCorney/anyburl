"""Tests for blocking candidates already linked to the query entity."""

import pytest
import torch
from torch import Tensor

from anyburl.core import Triple
from anyburl.exceptions import ConfigurationError
from anyburl.graph import HeteroGraph
from anyburl.prediction import BLOCKED_SCORE, LinkedPairBlocker, linked_pair_share

LIVES_IN = ("person", "lives_in", "city")
NEAR = ("city", "near", "city")
_NUM_PERSONS = 4
_NUM_CITIES = 3


class _ConstantScorer:
    """Scores every candidate 1.0."""

    def __init__(self, num_tails: int, num_heads: int) -> None:
        self.num_tails = num_tails
        self.num_heads = num_heads

    def score_tails(self, head_id: int) -> Tensor:
        return torch.ones(self.num_tails)

    def score_heads(self, tail_id: int) -> Tensor:
        return torch.ones(self.num_heads)


def _blocker(graph: HeteroGraph, edge_type: tuple[str, str, str]) -> LinkedPairBlocker:
    num_heads = graph.node_count(edge_type[0])
    num_tails = graph.node_count(edge_type[2])
    return LinkedPairBlocker(_ConstantScorer(num_tails, num_heads), graph, edge_type)


class TestLinkedPairBlocker:
    def test_blocks_tails_linked_by_any_relation(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        scores = _blocker(evaluator_graph, LIVES_IN).score_tails(2)

        assert scores.tolist() == [BLOCKED_SCORE, 1.0, BLOCKED_SCORE]

    def test_blocks_heads_linked_by_any_relation(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        scores = _blocker(evaluator_graph, LIVES_IN).score_heads(0)

        assert scores.tolist() == [BLOCKED_SCORE, 1.0, BLOCKED_SCORE, BLOCKED_SCORE]

    def test_blocks_links_in_both_directions_within_a_node_type(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        scores = _blocker(evaluator_graph, NEAR).score_tails(1)

        assert scores.tolist() == [BLOCKED_SCORE, 1.0, BLOCKED_SCORE]

    def test_leaves_the_wrapped_scores_unchanged(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        original = torch.ones(_NUM_CITIES)

        class _Fixed(_ConstantScorer):
            def score_tails(self, head_id: int) -> Tensor:
                return original

        LinkedPairBlocker(
            _Fixed(_NUM_CITIES, _NUM_PERSONS), evaluator_graph, LIVES_IN
        ).score_tails(0)

        assert original.tolist() == [1.0] * _NUM_CITIES


class TestLinkedPairShare:
    def test_counts_pairs_linked_by_any_relation(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        triples = [
            Triple(1, 1, "person", "city", "lives_in"),
            Triple(1, 0, "person", "city", "lives_in"),
            Triple(2, 0, "person", "city", "lives_in"),
            Triple(3, 1, "person", "city", "lives_in"),
        ]

        assert linked_pair_share(evaluator_graph, triples) == pytest.approx(0.5)

    def test_rejects_no_triples(self, evaluator_graph: HeteroGraph) -> None:
        with pytest.raises(ConfigurationError, match="at least one"):
            linked_pair_share(evaluator_graph, [])
