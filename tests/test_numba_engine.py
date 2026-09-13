"""Tests for the Numba-backed walk engine and its integer graph view."""

from itertools import pairwise

import torch
from torch_geometric.data import HeteroData

from anyburl.anyburl import build_walk_engine
from anyburl.graph import HeteroGraph
from anyburl.sampler import Triple
from anyburl.walk import (
    NumbaWalkEngine,
    UniformEdgeSelector,
    WalkConfig,
    WalkEngine,
    WalkStrategy,
)
from anyburl.walk._numba_graph import build_numba_graph_view


def _reference_engine(graph: HeteroGraph, config: WalkConfig) -> WalkEngine:
    """Build the torch reference engine with a uniform selector."""
    generator = torch.Generator().manual_seed(config.seed)
    return WalkEngine(graph, config, UniformEdgeSelector(generator))


def test_build_walk_engine_uniform_returns_numba(simple_graph: HeteroGraph) -> None:
    config = WalkConfig(strategy=WalkStrategy.UNIFORM, seed=42)
    engine = build_walk_engine(simple_graph, config)
    assert isinstance(engine, NumbaWalkEngine)


def _assert_path_valid(
    graph: HeteroGraph,
    path: list[tuple[int, str, str]],
    triple: Triple,
) -> None:
    """Assert every step is a real edge and the path ends at the tail."""
    assert path[0][0] == triple.head_id
    assert path[0][1] == triple.head_type
    for (src_id, src_type, relation), (dst_id, dst_type, _) in pairwise(path):
        neighbors = graph.get_neighbors(src_id, (src_type, relation, dst_type))
        assert dst_id in neighbors.tolist()
    assert path[-1][0] == triple.tail_id
    assert path[-1][1] == triple.tail_type
    assert path[-1][2] == ""


def test_numba_paths_are_structurally_valid(simple_graph: HeteroGraph) -> None:
    config = WalkConfig(min_length=2, max_length=4, max_attempts=300, seed=42)
    triple = Triple(0, 1, "person", "city", "lives_in")

    paths = NumbaWalkEngine(simple_graph, config).walk_from_triple(triple)

    assert len(paths) > 0
    for path in paths:
        _assert_path_valid(simple_graph, path, triple)


def test_numba_matches_reference_when_saturated(simple_graph: HeteroGraph) -> None:
    # Short walks on a tiny graph: many attempts saturate every possible
    # path, so both engines must discover the identical set despite
    # using different random number generators.
    config = WalkConfig(min_length=2, max_length=3, max_attempts=5000, seed=42)
    triple = Triple(0, 1, "person", "city", "lives_in")

    reference_paths = {
        tuple(p)
        for p in _reference_engine(simple_graph, config).walk_from_triple(triple)
    }
    numba_paths = {
        tuple(p) for p in NumbaWalkEngine(simple_graph, config).walk_from_triple(triple)
    }

    assert numba_paths == reference_paths
    assert len(numba_paths) > 0


def test_numba_reproducible_with_seed(simple_graph: HeteroGraph) -> None:
    config = WalkConfig(min_length=2, max_length=4, max_attempts=200, seed=7)
    triples = [
        Triple(0, 1, "person", "city", "lives_in"),
        Triple(1, 2, "person", "city", "lives_in"),
    ]

    first = NumbaWalkEngine(simple_graph, config)
    second = NumbaWalkEngine(simple_graph, config)

    for triple in triples:
        assert first.walk_from_triple(triple) == second.walk_from_triple(triple)


def test_numba_paths_reach_tail(simple_graph: HeteroGraph) -> None:
    config = WalkConfig(min_length=2, max_length=5, max_attempts=200, seed=42)
    triple = Triple(0, 2, "person", "person", "knows")

    for path in NumbaWalkEngine(simple_graph, config).walk_from_triple(triple):
        assert path[0][0] == triple.head_id
        assert path[0][1] == triple.head_type
        assert path[-1][0] == triple.tail_id
        assert path[-1][1] == triple.tail_type
        assert path[-1][2] == ""


def test_numba_returns_empty_on_dead_end() -> None:
    # city:1 has no outgoing edges back to a person, so a person->person
    # target can never be completed through it.
    data = HeteroData()
    data["person"].num_nodes = 2
    data["city"].num_nodes = 2
    data["person", "lives_in", "city"].edge_index = torch.tensor(
        [[0], [1]], dtype=torch.long
    )
    graph = HeteroGraph(data)
    config = WalkConfig(min_length=2, max_length=4, max_attempts=100, seed=42)
    triple = Triple(0, 1, "person", "person", "knows")

    assert NumbaWalkEngine(graph, config).walk_from_triple(triple) == []


def test_graph_view_encodes_relations_and_types(simple_graph: HeteroGraph) -> None:
    view = build_numba_graph_view(simple_graph)

    assert set(view.node_type_names) == {"person", "city"}
    assert set(view.edge_relation_names) == {"lives_in", "knows", "near"}
    assert view.node_type_to_id["person"] in range(len(view.node_type_names))
    assert view.crow_offsets[-1] == view.crow_all.shape[0]
    assert view.col_offsets[-1] == view.col_all.shape[0]
