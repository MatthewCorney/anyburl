"""Shared fixtures for the AnyBURL benchmark suite.

Synthetic graph topology
------------------------
3 node types: A, B, C — each with ``nodes_per_type`` nodes.
4 edge types:

    (A, r_ab, B)   body atom 1 of the CYCLIC / AC1 rules
    (B, r_bc, C)   body atom 2 of the CYCLIC / AC1 rules
    (B, r_ba, A)   enables back-walks
    (A, r_ac, C)   target relation (rule head)

All edges are random (uniform src/dst sampling) with fixed seed for
reproducibility.
"""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.graph import HeteroGraph

GRAPH_PROFILES = {
    "small": {"nodes_per_type": 500, "avg_degree": 10, "seed": 0},
    "medium": {"nodes_per_type": 5_000, "avg_degree": 10, "seed": 0},
    "large": {"nodes_per_type": 50_000, "avg_degree": 10, "seed": 0},
}

EDGE_TYPES = [
    ("A", "r_ab", "B"),
    ("B", "r_bc", "C"),
    ("B", "r_ba", "A"),
    ("A", "r_ac", "C"),
]


def make_synthetic_hetero_data(
    nodes_per_type: int,
    avg_degree: int,
    seed: int,
) -> HeteroData:
    """Create a random HeteroData with 3 node types and 4 edge types.

    Parameters
    ----------
    nodes_per_type : int
        Number of nodes for each of the three node types (A, B, C).
    avg_degree : int
        Average out-degree per node per edge type.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    HeteroData
        A PyG HeteroData object ready for HeteroGraph wrapping.
    """
    gen = torch.Generator().manual_seed(seed)
    n = nodes_per_type
    data = HeteroData()
    for node_type in ("A", "B", "C"):
        data[node_type].num_nodes = n
    for src, rel, dst in EDGE_TYPES:
        n_edges = n * avg_degree
        src_idx = torch.randint(0, n, (n_edges,), generator=gen)
        dst_idx = torch.randint(0, n, (n_edges,), generator=gen)
        data[src, rel, dst].edge_index = torch.stack([src_idx, dst_idx])
    return data


def make_synthetic_graph(
    nodes_per_type: int,
    avg_degree: int,
    seed: int,
) -> HeteroGraph:
    """Create a HeteroGraph from a synthetic random HeteroData.

    Parameters
    ----------
    nodes_per_type : int
        Number of nodes for each of the three node types.
    avg_degree : int
        Average out-degree per node per edge type.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    HeteroGraph
        A wrapped graph with prebuilt CSR indices.
    """
    return HeteroGraph(make_synthetic_hetero_data(nodes_per_type, avg_degree, seed))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", params=["small", "medium", "large"])
def graph(request) -> HeteroGraph:
    """Session-scoped HeteroGraph parametrised by size (small/medium/large)."""
    return make_synthetic_graph(**GRAPH_PROFILES[request.param])


@pytest.fixture(scope="session", params=["small", "medium", "large"])
def raw_hetero_data(request) -> HeteroData:
    """Session-scoped HeteroData parametrised by size (small/medium/large)."""
    return make_synthetic_hetero_data(**GRAPH_PROFILES[request.param])


@pytest.fixture(scope="session", params=["small", "medium", "large"])
def eval_graph(request) -> HeteroGraph:
    """Session-scoped HeteroGraph parametrised by size (small/medium/large)."""
    return make_synthetic_graph(**GRAPH_PROFILES[request.param])


@pytest.fixture(scope="session")
def small_graph() -> HeteroGraph:
    """Session-scoped small HeteroGraph (500 nodes/type)."""
    return make_synthetic_graph(**GRAPH_PROFILES["small"])
