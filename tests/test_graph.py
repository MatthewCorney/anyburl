"""Tests for HeteroGraph wrapper."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.exceptions import GraphSchemaError
from anyburl.graph import HeteroGraph, mirrored_edge_type


def test_raises_on_no_edge_types() -> None:
    data = HeteroData()
    data["person"].num_nodes = 5
    with pytest.raises(GraphSchemaError, match="no edge types"):
        HeteroGraph(data)


def test_raises_on_all_empty_edges() -> None:
    data = HeteroData()
    data["person"].num_nodes = 5
    data["city"].num_nodes = 3
    data["person", "lives_in", "city"].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )
    with pytest.raises(GraphSchemaError, match="empty edge indices"):
        HeteroGraph(data)


@pytest.mark.parametrize(("node_type", "expected"), [("person", 5), ("city", 3)])
def test_node_count(simple_graph: HeteroGraph, node_type: str, expected: int) -> None:
    assert simple_graph.node_count(node_type) == expected


@pytest.mark.parametrize("method", ["node_count", "outgoing_edge_types"])
def test_unknown_node_type_raises(simple_graph: HeteroGraph, method: str) -> None:
    with pytest.raises(GraphSchemaError, match="Unknown node type"):
        getattr(simple_graph, method)("animal")


@pytest.mark.parametrize(
    ("edge_type", "expected"),
    [
        (("person", "lives_in", "city"), 6),
        (("person", "knows", "person"), 5),
        (("city", "near", "city"), 3),
    ],
)
def test_edge_count(
    simple_graph: HeteroGraph,
    edge_type: tuple[str, str, str],
    expected: int,
) -> None:
    assert simple_graph.edge_count(edge_type) == expected


@pytest.mark.parametrize("method", ["edge_count", "csr_index", "edge_index"])
def test_unknown_edge_type_raises(simple_graph: HeteroGraph, method: str) -> None:
    with pytest.raises(GraphSchemaError, match="Unknown edge type"):
        getattr(simple_graph, method)(("person", "flies_to", "city"))


def test_total_edge_count(simple_graph: HeteroGraph) -> None:
    assert simple_graph.total_edge_count() == 14


def test_node_types(simple_graph: HeteroGraph) -> None:
    assert set(simple_graph.node_types) == {"person", "city"}


def test_edge_types(simple_graph: HeteroGraph) -> None:
    edge_types = set(simple_graph.edge_types)
    assert ("person", "lives_in", "city") in edge_types
    assert ("person", "knows", "person") in edge_types
    assert ("city", "near", "city") in edge_types


def test_outgoing_edge_types(simple_graph: HeteroGraph) -> None:
    person_ets = simple_graph.outgoing_edge_types("person")
    assert ("person", "lives_in", "city") in person_ets
    assert ("person", "knows", "person") in person_ets


def test_edge_index_shape(simple_graph: HeteroGraph) -> None:
    ei = simple_graph.edge_index(("person", "knows", "person"))
    assert ei.shape == (2, 5)


def test_mirrored_edge_type_swaps_endpoints() -> None:
    assert mirrored_edge_type(("person", "lives_in", "city")) == (
        "city",
        "lives_in",
        "person",
    )


class TestCached:
    def test_builds_once_and_shares_result(self, simple_graph: HeteroGraph) -> None:
        calls: list[HeteroGraph] = []

        def build(graph: HeteroGraph) -> list[int]:
            calls.append(graph)
            return [graph.total_edge_count()]

        first = simple_graph.cached(build)
        second = simple_graph.cached(build)

        assert first is second
        assert calls == [simple_graph]

    def test_distinct_builders_are_cached_separately(
        self, simple_graph: HeteroGraph
    ) -> None:
        assert simple_graph.cached(lambda g: "a") != simple_graph.cached(lambda g: "b")
