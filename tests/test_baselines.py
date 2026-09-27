"""Tests for the reference scorers used to calibrate a learned MRR."""

import pytest
import torch

from anyburl.baselines import (
    MetaPathScorer,
    PopularityScorer,
    RandomScorer,
    reverse_chain,
)
from anyburl.graph import HeteroGraph

TARGET = ("person", "lives_in", "city")
BORN_IN = ("person", "born_in", "city")
NEAR = ("city", "near", "city")


def test_random_scorer_shapes(evaluator_graph: HeteroGraph) -> None:
    scorer = RandomScorer(evaluator_graph, TARGET)

    assert scorer.score_tails(0).shape == (evaluator_graph.node_count("city"),)
    assert scorer.score_heads(0).shape == (evaluator_graph.node_count("person"),)


def test_random_scorer_is_reproducible(evaluator_graph: HeteroGraph) -> None:
    first = RandomScorer(evaluator_graph, TARGET, seed=7).score_tails(0)
    second = RandomScorer(evaluator_graph, TARGET, seed=7).score_tails(0)

    assert torch.equal(first, second)


def test_popularity_scorer_counts_degree(evaluator_graph: HeteroGraph) -> None:
    scorer = PopularityScorer(evaluator_graph, TARGET)
    edge_index = evaluator_graph.edge_index(TARGET)

    scores = scorer.score_tails(0)

    for city_id in range(evaluator_graph.node_count("city")):
        assert scores[city_id].item() == int((edge_index[1] == city_id).sum())


def test_popularity_scorer_ignores_the_query(evaluator_graph: HeteroGraph) -> None:
    """A query-independent score is the point: it measures degree alone."""
    scorer = PopularityScorer(evaluator_graph, TARGET)

    assert torch.equal(scorer.score_tails(0), scorer.score_tails(1))


def test_popularity_scorer_returns_a_fresh_tensor(
    evaluator_graph: HeteroGraph,
) -> None:
    """Callers filter scores in place, so a shared tensor would corrupt later calls."""
    scorer = PopularityScorer(evaluator_graph, TARGET)

    scorer.score_tails(0)[0] = -999.0

    assert scorer.score_tails(0)[0].item() != -999.0


def test_meta_path_scorer_counts_paths(evaluator_graph: HeteroGraph) -> None:
    """born_in then near should reach exactly the cities two steps away."""
    scorer = MetaPathScorer(evaluator_graph, (BORN_IN, NEAR))

    scores = scorer.score_tails(0)

    born = evaluator_graph.get_neighbors(0, BORN_IN)
    expected = torch.zeros(evaluator_graph.node_count("city"))
    for city in born.tolist():
        for reachable in evaluator_graph.get_neighbors(city, NEAR).tolist():
            expected[reachable] += 1
    assert torch.equal(scores, expected)


def test_meta_path_scorer_directions_agree(evaluator_graph: HeteroGraph) -> None:
    """Path counts must be symmetric: the same pair counted from either end."""
    scorer = MetaPathScorer(evaluator_graph, (BORN_IN, NEAR))

    for person_id in range(evaluator_graph.node_count("person")):
        tails = scorer.score_tails(person_id)
        for city_id in range(evaluator_graph.node_count("city")):
            heads = scorer.score_heads(city_id)
            assert tails[city_id].item() == heads[person_id].item()


def test_meta_path_scorer_rejects_empty_chain(evaluator_graph: HeteroGraph) -> None:
    with pytest.raises(ValueError, match="at least one edge type"):
        MetaPathScorer(evaluator_graph, ())


def test_meta_path_scorer_rejects_disjoint_chain(evaluator_graph: HeteroGraph) -> None:
    """A chain whose endpoints do not meet cannot be multiplied."""
    with pytest.raises(ValueError, match="does not join"):
        MetaPathScorer(evaluator_graph, (NEAR, BORN_IN))


def test_reverse_chain_flips_order_and_direction() -> None:
    assert reverse_chain((BORN_IN, NEAR)) == (
        ("city", "near", "city"),
        ("city", "born_in", "person"),
    )
