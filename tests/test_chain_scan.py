"""Tests for counting chain groundings without materialising the product."""

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl._chain_scan import ChainScanner
from anyburl._csr_tables import build_csr_tables
from anyburl.graph import EdgeTypeTuple, HeteroGraph

BORN_IN = ("person", "born_in", "city")
NEAR = ("city", "near", "city")
LIVES_IN = ("person", "lives_in", "city")


def _reference_counts(
    graph: HeteroGraph,
    signature: tuple[EdgeTypeTuple, ...],
    head_edge_type: EdgeTypeTuple,
) -> tuple[int, int]:
    """Oracle: chain matmul, then compare coordinate sets.

    Written independently of the production helpers so that agreement is
    evidence about the kernel rather than two copies of one mistake.
    """
    product = graph.get_csr_matrix(signature[0])
    for edge_type in signature[1:]:
        product = product @ graph.get_csr_matrix(edge_type)

    predicted = _coordinates(product)
    if not predicted:
        return 0, 0
    known = _coordinates(graph.get_csr_matrix(head_edge_type))
    return len(predicted), len(predicted & known)


def _coordinates(matrix: torch.Tensor) -> set[tuple[int, int]]:
    """Return a CSR tensor's stored (row, column) pairs."""
    crow = matrix.crow_indices()
    col = matrix.col_indices()
    rows = torch.repeat_interleave(torch.arange(crow.numel() - 1), crow[1:] - crow[:-1])
    return set(zip(rows.tolist(), col.tolist(), strict=True))


def _assert_matches_oracle(
    graph: HeteroGraph,
    signature: tuple[EdgeTypeTuple, ...],
    head_edge_type: EdgeTypeTuple,
) -> None:
    expected = _reference_counts(graph, signature, head_edge_type)
    actual = ChainScanner(graph).scan_all_rows(signature, head_edge_type)
    assert actual == expected, f"{signature} vs {head_edge_type}"


def test_matches_oracle_on_the_shared_cyclic_rule(
    evaluator_graph: HeteroGraph,
) -> None:
    _assert_matches_oracle(evaluator_graph, (BORN_IN, NEAR), LIVES_IN)


def test_single_atom_chain(evaluator_graph: HeteroGraph) -> None:
    _assert_matches_oracle(evaluator_graph, (LIVES_IN,), LIVES_IN)


def test_per_row_counts_sum_to_the_total(evaluator_graph: HeteroGraph) -> None:
    scanner = ChainScanner(evaluator_graph)
    total = scanner.scan_all_rows((BORN_IN, NEAR), LIVES_IN)

    rows = np.arange(evaluator_graph.node_count("person"), dtype=np.int64)
    predictions, support = scanner.scan_rows((BORN_IN, NEAR), LIVES_IN, rows)

    assert (int(predictions.sum()), int(support.sum())) == total


def test_scanning_one_row_at_a_time_agrees(evaluator_graph: HeteroGraph) -> None:
    """Stamps must stay valid across calls that reuse the scanner."""
    scanner = ChainScanner(evaluator_graph)
    rows = np.arange(evaluator_graph.node_count("person"), dtype=np.int64)
    batched, batched_support = scanner.scan_rows((BORN_IN, NEAR), LIVES_IN, rows)

    for row in rows.tolist():
        one = np.array([row], dtype=np.int64)
        predictions, support = scanner.scan_rows((BORN_IN, NEAR), LIVES_IN, one)
        assert int(predictions[0]) == int(batched[row])
        assert int(support[0]) == int(batched_support[row])


def test_repeated_results_are_deterministic(evaluator_graph: HeteroGraph) -> None:
    scanner = ChainScanner(evaluator_graph)
    first = scanner.scan_all_rows((BORN_IN, NEAR), LIVES_IN)
    second = scanner.scan_all_rows((BORN_IN, NEAR), LIVES_IN)

    assert first == second


# --- the failure mode a per-row stamp would cause ---------------------------

SELF = ("a", "self", "a")
CROSS = ("a", "cross", "b")
HEAD_AB = ("a", "head", "b")


@pytest.fixture
def repeated_type_graph() -> HeteroGraph:
    """Node type ``a`` recurs at several chain levels, as BIOKG's worst chain does."""
    data = HeteroData()
    data["a"].num_nodes = 6
    data["b"].num_nodes = 4
    data[SELF].edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 4]])
    data[CROSS].edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 1]])
    data[HEAD_AB].edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    return HeteroGraph(data)


def test_node_type_repeated_across_levels(repeated_type_graph: HeteroGraph) -> None:
    """A node reached at level 1 must still be admissible at level 2.

    Stamping per row rather than per level would suppress it and undercount.
    """
    _assert_matches_oracle(repeated_type_graph, (SELF, SELF, CROSS), HEAD_AB)


def test_repeated_edge_type_within_one_chain(
    repeated_type_graph: HeteroGraph,
) -> None:
    _assert_matches_oracle(repeated_type_graph, (SELF, SELF, SELF, CROSS), HEAD_AB)


def test_self_loop_edge_type() -> None:
    data = HeteroData()
    data["a"].num_nodes = 4
    data["b"].num_nodes = 3
    data[SELF].edge_index = torch.tensor([[0, 1, 1, 2], [0, 1, 2, 2]])
    data[CROSS].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data[HEAD_AB].edge_index = torch.tensor([[0, 1], [1, 2]])
    graph = HeteroGraph(data)

    _assert_matches_oracle(graph, (SELF, CROSS), HEAD_AB)


def test_duplicate_edges_are_counted_once() -> None:
    """CSR coalesces duplicates, so distinct pairs is the shared semantics."""
    data = HeteroData()
    data["a"].num_nodes = 3
    data["b"].num_nodes = 3
    data[CROSS].edge_index = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 2]])
    data[HEAD_AB].edge_index = torch.tensor([[0, 1], [1, 2]])
    graph = HeteroGraph(data)

    _assert_matches_oracle(graph, (CROSS,), HEAD_AB)


def test_isolated_rows_reach_nothing() -> None:
    data = HeteroData()
    data["a"].num_nodes = 5
    data["b"].num_nodes = 3
    data[CROSS].edge_index = torch.tensor([[0], [0]])
    data[HEAD_AB].edge_index = torch.tensor([[0], [0]])
    graph = HeteroGraph(data)

    rows = np.arange(5, dtype=np.int64)
    predictions, support = ChainScanner(graph).scan_rows((CROSS,), HEAD_AB, rows)

    assert predictions.tolist() == [1, 0, 0, 0, 0]
    assert support.tolist() == [1, 0, 0, 0, 0]


def test_empty_row_selection_returns_empty(evaluator_graph: HeteroGraph) -> None:
    empty = np.zeros(0, dtype=np.int64)
    predictions, support = ChainScanner(evaluator_graph).scan_rows(
        (BORN_IN, NEAR), LIVES_IN, empty
    )

    assert predictions.size == 0
    assert support.size == 0


# --- randomised parity sweep ------------------------------------------------


def _random_graph(rng: np.random.Generator) -> tuple[HeteroGraph, list[EdgeTypeTuple]]:
    """Build a small random heterogeneous graph and list its edge types."""
    node_types = ["t0", "t1", "t2"]
    sizes = {name: int(rng.integers(5, 40)) for name in node_types}

    data = HeteroData()
    for name, size in sizes.items():
        data[name].num_nodes = size

    edge_types: list[EdgeTypeTuple] = []
    for index in range(int(rng.integers(4, 7))):
        src = node_types[int(rng.integers(0, len(node_types)))]
        dst = node_types[int(rng.integers(0, len(node_types)))]
        edge_type = (src, f"r{index}", dst)
        count = int(rng.integers(1, sizes[src] * 3))
        data[edge_type].edge_index = torch.stack(
            [
                torch.from_numpy(rng.integers(0, sizes[src], count)),
                torch.from_numpy(rng.integers(0, sizes[dst], count)),
            ]
        ).long()
        edge_types.append(edge_type)
    return HeteroGraph(data), edge_types


def _random_chain(
    rng: np.random.Generator,
    edge_types: list[EdgeTypeTuple],
    length: int,
) -> tuple[EdgeTypeTuple, ...] | None:
    """Build a joining chain of the requested length, or ``None`` if stuck."""
    start = edge_types[int(rng.integers(0, len(edge_types)))]
    chain = [start]
    for _ in range(length - 1):
        options = [et for et in edge_types if et[0] == chain[-1][2]]
        if not options:
            return None
        chain.append(options[int(rng.integers(0, len(options)))])
    return tuple(chain)


def test_randomised_parity_against_the_oracle() -> None:
    """Exhaustive-ish agreement across shapes, densities and chain lengths."""
    rng = np.random.default_rng(20260927)
    compared = 0

    for _ in range(40):
        graph, edge_types = _random_graph(rng)
        for length in (1, 2, 3, 4):
            chain = _random_chain(rng, edge_types, length)
            if chain is None:
                continue
            heads = [
                et
                for et in edge_types
                if et[0] == chain[0][0] and et[2] == chain[-1][2]
            ]
            if not heads:
                continue
            _assert_matches_oracle(graph, chain, heads[0])
            compared += 1

    assert compared > 50, f"sweep only compared {compared} cases"


# --- table layout and validation --------------------------------------------


def test_edge_ids_follow_table_order(evaluator_graph: HeteroGraph) -> None:
    tables = build_csr_tables(evaluator_graph)

    ids = tables.edge_ids((BORN_IN, NEAR))

    assert [
        tables.edge_type_to_id[BORN_IN],
        tables.edge_type_to_id[NEAR],
    ] == ids.tolist()


def test_block_offsets_cover_the_concatenation(evaluator_graph: HeteroGraph) -> None:
    tables = build_csr_tables(evaluator_graph)

    assert int(tables.crow_offsets[-1]) == tables.crow_all.shape[0]
    assert int(tables.col_offsets[-1]) == tables.col_all.shape[0]


def test_rejects_an_empty_chain(evaluator_graph: HeteroGraph) -> None:
    with pytest.raises(ValueError, match="at least one edge type"):
        ChainScanner(evaluator_graph).scan_all_rows((), LIVES_IN)


def test_rejects_a_chain_that_does_not_join(evaluator_graph: HeteroGraph) -> None:
    with pytest.raises(ValueError, match="does not join"):
        ChainScanner(evaluator_graph).scan_all_rows((NEAR, BORN_IN), LIVES_IN)


def test_rejects_a_chain_whose_start_differs_from_the_head(
    evaluator_graph: HeteroGraph,
) -> None:
    with pytest.raises(ValueError, match="chain starts at"):
        ChainScanner(evaluator_graph).scan_all_rows((NEAR,), LIVES_IN)


def test_rejects_a_chain_whose_end_differs_from_the_head(
    repeated_type_graph: HeteroGraph,
) -> None:
    with pytest.raises(ValueError, match="chain ends at"):
        ChainScanner(repeated_type_graph).scan_all_rows((SELF,), HEAD_AB)
