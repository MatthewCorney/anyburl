"""Tests for type-reachability pruning of walk candidates."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.graph import HeteroGraph
from anyburl.sampler import Triple
from anyburl.walk import (
    EdgeWeighting,
    NumbaWalkEngine,
    WalkConfig,
    WalkStrategy,
)
from anyburl.walk._numba_graph import build_numba_graph_view
from anyburl.walk._reachability import MAX_TRACKED_STEPS, build_step_tables

KNOWS = ("person", "knows", "person")
LIVES_IN = ("person", "lives_in", "city")
HAS_JOB = ("person", "has_job", "job")
REQUIRES = ("job", "requires", "job")


@pytest.fixture
def sink_graph() -> HeteroGraph:
    """A graph where ``job`` is an absorbing sink that never reaches ``city``."""
    data = HeteroData()
    data["person"].num_nodes = 4
    data["city"].num_nodes = 3
    data["job"].num_nodes = 3
    data[KNOWS].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    data[LIVES_IN].edge_index = torch.tensor([[1, 2, 3], [0, 1, 2]])
    data[HAS_JOB].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data[REQUIRES].edge_index = torch.tensor([[0, 1], [1, 2]])
    return HeteroGraph(data)


def _block(tables, view, target: str, node_type: str, steps_left: int) -> list[int]:
    """Return the candidate edge ids of one table block."""
    stride = MAX_TRACKED_STEPS + 1
    index = (
        view.node_type_to_id[target] * tables.num_node_types
        + view.node_type_to_id[node_type]
    ) * stride + steps_left
    start = int(tables.offsets[index])
    stop = int(tables.offsets[index + 1])
    return [int(e) for e in tables.edges[start:stop]]


def test_unpruned_table_keeps_every_outgoing_type(sink_graph: HeteroGraph) -> None:
    """The uniform strategy must see exactly the candidates it always saw."""
    view = build_numba_graph_view(sink_graph)
    tables = build_step_tables(view, prune=False)

    outgoing = len(sink_graph.outgoing_edge_types("person"))
    for steps_left in range(MAX_TRACKED_STEPS + 1):
        assert len(_block(tables, view, "city", "person", steps_left)) == outgoing


def test_pruning_drops_edges_into_a_sink(sink_graph: HeteroGraph) -> None:
    """`has_job` can never lead to a city, so it must never be offered."""
    view = build_numba_graph_view(sink_graph)
    tables = build_step_tables(view, prune=True)

    for steps_left in range(1, MAX_TRACKED_STEPS + 1):
        candidates = _block(tables, view, "city", "person", steps_left)
        relations = {view.edge_relation_names[e] for e in candidates}
        assert "has_job" not in relations


def test_a_sink_type_offers_nothing(sink_graph: HeteroGraph) -> None:
    """From inside the sink the walk is already lost and should abort."""
    view = build_numba_graph_view(sink_graph)
    tables = build_step_tables(view, prune=True)

    for steps_left in range(MAX_TRACKED_STEPS + 1):
        assert _block(tables, view, "city", "job", steps_left) == []


def test_pruning_widens_as_more_steps_remain(sink_graph: HeteroGraph) -> None:
    """One step from a person only `lives_in` lands on a city; two admit `knows`."""
    view = build_numba_graph_view(sink_graph)
    tables = build_step_tables(view, prune=True)

    one_step = {
        view.edge_relation_names[e] for e in _block(tables, view, "city", "person", 1)
    }
    two_steps = {
        view.edge_relation_names[e] for e in _block(tables, view, "city", "person", 2)
    }

    assert one_step == {"lives_in"}
    assert two_steps == {"lives_in", "knows"}


def test_no_steps_left_offers_nothing(sink_graph: HeteroGraph) -> None:
    view = build_numba_graph_view(sink_graph)
    tables = build_step_tables(view, prune=True)

    assert _block(tables, view, "city", "person", 0) == []


def test_pruned_walks_never_enter_the_sink(sink_graph: HeteroGraph) -> None:
    """The end-to-end guarantee: no returned path passes through `job`."""
    config = WalkConfig(
        max_length=4,
        min_length=2,
        max_attempts=200,
        strategy=WalkStrategy.REACHABILITY_PRUNED,
    )
    engine = NumbaWalkEngine(sink_graph, config)
    triple = Triple(
        head_id=0, tail_id=1, head_type="person", tail_type="city", relation="lives_in"
    )

    for path in engine.walk_from_triple(triple):
        assert all(node_type != "job" for _, node_type, _ in path)


def test_pruned_walks_still_reach_the_tail(sink_graph: HeteroGraph) -> None:
    """Pruning must remove only hopeless continuations, not valid ones."""
    config = WalkConfig(
        max_length=4,
        min_length=2,
        max_attempts=500,
        strategy=WalkStrategy.REACHABILITY_PRUNED,
    )
    engine = NumbaWalkEngine(sink_graph, config)
    triple = Triple(
        head_id=0, tail_id=1, head_type="person", tail_type="city", relation="lives_in"
    )

    paths = engine.walk_from_triple(triple)

    assert paths
    for path in paths:
        assert path[-1][0] == 1
        assert path[-1][1] == "city"


def test_pruned_finds_at_least_as_many_paths(sink_graph: HeteroGraph) -> None:
    """Pruning exists to raise yield; on a graph with a sink it must."""
    triple = Triple(
        head_id=0, tail_id=1, head_type="person", tail_type="city", relation="lives_in"
    )

    def count(strategy: WalkStrategy) -> int:
        config = WalkConfig(
            max_length=3, min_length=2, max_attempts=300, strategy=strategy, seed=1
        )
        return len(NumbaWalkEngine(sink_graph, config).walk_from_triple(triple))

    assert count(WalkStrategy.REACHABILITY_PRUNED) >= count(WalkStrategy.UNIFORM)


def test_inverse_frequency_favours_rare_relations(sink_graph: HeteroGraph) -> None:
    """`knows` (3 edges) should be taken more often than a 100x commoner type."""
    data = HeteroData()
    data["person"].num_nodes = 60
    data["city"].num_nodes = 2
    common = torch.stack(
        [torch.arange(60).repeat_interleave(1), torch.zeros(60, dtype=torch.long)]
    )
    data[("person", "common", "city")].edge_index = common
    data[("person", "rare", "city")].edge_index = torch.tensor([[0], [1]])

    graph = HeteroGraph(data)
    triple = Triple(
        head_id=0, tail_id=1, head_type="person", tail_type="city", relation="common"
    )

    def rare_share(weighting: EdgeWeighting) -> int:
        config = WalkConfig(
            max_length=1,
            min_length=1,
            max_attempts=400,
            edge_weighting=weighting,
            seed=5,
        )
        paths = NumbaWalkEngine(graph, config).walk_from_triple(triple)
        return len(paths)

    assert rare_share(EdgeWeighting.INVERSE_FREQUENCY) >= rare_share(
        EdgeWeighting.UNIFORM
    )


def test_edge_weighting_defaults_to_uniform() -> None:
    assert WalkConfig().edge_weighting is EdgeWeighting.UNIFORM
