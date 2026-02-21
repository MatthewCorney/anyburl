"""Tests for HeteroGraph wrapper."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.graph import HeteroGraph


def test_builds_from_valid_data(simple_graph: HeteroGraph) -> None:
    assert simple_graph is not None


def test_raises_on_no_edge_types() -> None:
    data = HeteroData()
    data["person"].num_nodes = 5
    with pytest.raises(ValueError, match="no edge types"):
        HeteroGraph(data)


def test_raises_on_all_empty_edges() -> None:
    data = HeteroData()
    data["person"].num_nodes = 5
    data["city"].num_nodes = 3
    data["person", "lives_in", "city"].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )
    with pytest.raises(ValueError, match="empty edge indices"):
        HeteroGraph(data)


@pytest.mark.parametrize(("node_type", "expected"), [("person", 5), ("city", 3)])
def test_node_count(simple_graph: HeteroGraph, node_type: str, expected: int) -> None:
    assert simple_graph.node_count(node_type) == expected


def test_node_count_unknown_type(simple_graph: HeteroGraph) -> None:
    with pytest.raises(ValueError, match="Unknown node type"):
        simple_graph.node_count("animal")


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


def test_edge_count_unknown_type(simple_graph: HeteroGraph) -> None:
    with pytest.raises(ValueError, match="Unknown edge type"):
        simple_graph.edge_count(("person", "flies_to", "city"))


def test_total_edge_count(simple_graph: HeteroGraph) -> None:
    assert simple_graph.total_edge_count() == 14


def test_get_neighbors_lives_in(simple_graph: HeteroGraph) -> None:
    neighbors = simple_graph.get_neighbors(0, ("person", "lives_in", "city"))
    assert sorted(neighbors.tolist()) == [0, 1]


def test_get_neighbors_knows(simple_graph: HeteroGraph) -> None:
    neighbors = simple_graph.get_neighbors(0, ("person", "knows", "person"))
    assert neighbors.tolist() == [1]


def test_get_neighbors_no_outgoing(simple_graph: HeteroGraph) -> None:
    neighbors = simple_graph.get_neighbors(1, ("person", "lives_in", "city"))
    assert neighbors.tolist() == [1]


def test_get_neighbors_unknown_edge_type(simple_graph: HeteroGraph) -> None:
    with pytest.raises(ValueError, match="Unknown edge type"):
        simple_graph.get_neighbors(0, ("person", "flies_to", "city"))


def test_csr_matrix_shape(simple_graph: HeteroGraph) -> None:
    matrix = simple_graph.get_csr_matrix(("person", "lives_in", "city"))
    assert matrix.shape == (5, 3)


def test_csr_matrix_values(simple_graph: HeteroGraph) -> None:
    matrix = simple_graph.get_csr_matrix(("person", "lives_in", "city"))
    dense = matrix.to_dense()
    assert dense[0, 0].item() == 1.0
    assert dense[0, 1].item() == 1.0
    assert dense[0, 2].item() == 0.0


def test_csr_matrix_cached(simple_graph: HeteroGraph) -> None:
    m1 = simple_graph.get_csr_matrix(("person", "lives_in", "city"))
    m2 = simple_graph.get_csr_matrix(("person", "lives_in", "city"))
    assert m1 is m2


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


def test_outgoing_edge_types_unknown(simple_graph: HeteroGraph) -> None:
    with pytest.raises(ValueError, match="Unknown node type"):
        simple_graph.outgoing_edge_types("animal")


def test_edge_index_shape(simple_graph: HeteroGraph) -> None:
    ei = simple_graph.edge_index(("person", "knows", "person"))
    assert ei.shape == (2, 5)


def test_edge_index_unknown(simple_graph: HeteroGraph) -> None:
    with pytest.raises(ValueError, match="Unknown edge type"):
        simple_graph.edge_index(("person", "flies_to", "city"))
