"""Tests for WalkEngine."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.anyburl import build_walk_engine
from anyburl.graph import HeteroGraph
from anyburl.sampler import Triple
from anyburl.walk import WalkConfig, WalkStrategy


def _make_linear_graph() -> HeteroGraph:
    """Create a deterministic linear graph: A0 -> B0 -> A1.

    This graph guarantees a successful walk from A0 to A1 in exactly 2 steps
    when using the "r1" and "r2" edges.
    """
    data = HeteroData()
    data["A"].num_nodes = 2
    data["B"].num_nodes = 1

    data["A", "r1", "B"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data["B", "r2", "A"].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return HeteroGraph(data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_length": 0}, "max_length must be positive"),
        ({"min_length": 0}, "min_length must be positive"),
        ({"max_attempts": 0}, "max_attempts must be positive"),
        ({"min_length": 3, "max_length": 2}, "min_length"),
    ],
)
def test_walk_config_invalid_params(
    kwargs: dict[str, int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        WalkConfig(**kwargs)  # type: ignore[arg-type]


def test_walk_finds_path_on_linear_graph() -> None:
    graph = _make_linear_graph()
    config = WalkConfig(min_length=2, max_length=2, max_attempts=50, seed=42)
    engine = build_walk_engine(graph, config)
    triple = Triple(
        head_id=0, tail_id=1, head_type="A", tail_type="A", relation="target"
    )

    paths = engine.walk_from_triple(triple)
    assert len(paths) >= 1

    path = paths[0]
    assert len(path) == 3
    assert path[0] == (0, "A", "r1")
    assert path[1] == (0, "B", "r2")
    assert path[2] == (1, "A", "")


def test_walk_respects_min_length() -> None:
    graph = _make_linear_graph()
    config = WalkConfig(min_length=3, max_length=5, max_attempts=50, seed=42)
    engine = build_walk_engine(graph, config)
    triple = Triple(
        head_id=0, tail_id=1, head_type="A", tail_type="A", relation="target"
    )

    paths = engine.walk_from_triple(triple)
    assert paths == []


def test_walk_deduplicates_paths() -> None:
    graph = _make_linear_graph()
    config = WalkConfig(min_length=2, max_length=2, max_attempts=100, seed=42)
    engine = build_walk_engine(graph, config)
    triple = Triple(
        head_id=0, tail_id=1, head_type="A", tail_type="A", relation="target"
    )

    paths = engine.walk_from_triple(triple)
    assert len(paths) == 1


def test_walk_returns_empty_on_dead_end() -> None:
    data = HeteroData()
    data["X"].num_nodes = 2
    data["X", "rel", "X"].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    graph = HeteroGraph(data)
    config = WalkConfig(min_length=2, max_length=3, max_attempts=10, seed=42)
    engine = build_walk_engine(graph, config)
    triple = Triple(head_id=0, tail_id=0, head_type="X", tail_type="X", relation="self")

    paths = engine.walk_from_triple(triple)
    assert paths == []


def test_pathstep_format_compatible_with_generalizer() -> None:
    graph = _make_linear_graph()
    config = WalkConfig(min_length=2, max_length=2, max_attempts=50, seed=42)
    engine = build_walk_engine(graph, config)
    triple = Triple(
        head_id=0, tail_id=1, head_type="A", tail_type="A", relation="target"
    )

    paths = engine.walk_from_triple(triple)
    path = paths[0]

    for step in path:
        assert len(step) == 3
        assert isinstance(step[0], int)
        assert isinstance(step[1], str)
        assert isinstance(step[2], str)
    assert path[-1][2] == ""
    assert path[-1][0] == triple.tail_id
    assert path[-1][1] == triple.tail_type


def test_walk_on_heterogeneous_graph(simple_graph: HeteroGraph) -> None:
    config = WalkConfig(min_length=2, max_length=5, max_attempts=200, seed=42)
    engine = build_walk_engine(simple_graph, config)

    triple = Triple(
        head_id=0,
        tail_id=2,
        head_type="person",
        tail_type="person",
        relation="some_relation",
    )

    paths = engine.walk_from_triple(triple)
    for path in paths:
        assert path[0][0] == 0
        assert path[0][1] == "person"
        assert path[-1][0] == 2
        assert path[-1][1] == "person"
        assert path[-1][2] == ""


def test_build_walk_engine_relation_weighted(simple_graph: HeteroGraph) -> None:
    config = WalkConfig(
        min_length=2,
        max_length=3,
        max_attempts=50,
        strategy=WalkStrategy.RELATION_WEIGHTED,
        seed=42,
    )
    engine = build_walk_engine(simple_graph, config)
    triple = Triple(
        head_id=0,
        tail_id=2,
        head_type="person",
        tail_type="person",
        relation="knows",
    )
    paths = engine.walk_from_triple(triple)
    assert isinstance(paths, list)
