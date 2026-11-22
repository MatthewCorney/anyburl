"""Tests for counting chain groundings without materialising the product."""

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.chain import ChainScanner, PathCounter, build_csr_tables
from anyburl.core import EdgeTypeTuple
from anyburl.exceptions import GraphSchemaError, InvalidRuleError
from anyburl.graph import HeteroGraph
from tests import adjacency

BORN_IN = ("person", "born_in", "city")
NEAR = ("city", "near", "city")
LIVES_IN = ("person", "lives_in", "city")
KNOWS = ("person", "knows", "person")


def _reference_counts(
    graph: HeteroGraph,
    signature: tuple[EdgeTypeTuple, ...],
    head_edge_type: EdgeTypeTuple,
) -> tuple[int, int]:
    """Oracle: chain matmul, then compare coordinate sets.

    Written independently of the production helpers so that agreement is
    evidence about the kernel rather than two copies of one mistake.
    """
    product = adjacency.adjacency_matrix(graph, signature[0])
    for edge_type in signature[1:]:
        product = product @ adjacency.adjacency_matrix(graph, edge_type)

    predicted = _coordinates(product)
    if not predicted:
        return 0, 0
    known = _coordinates(adjacency.adjacency_matrix(graph, head_edge_type))
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


@pytest.mark.parametrize(
    "signature",
    [(BORN_IN, NEAR), (LIVES_IN,)],
    ids=["shared_cyclic_rule", "single_atom_chain"],
)
def test_matches_oracle_on_the_evaluator_graph(
    evaluator_graph: HeteroGraph, signature: tuple[EdgeTypeTuple, ...]
) -> None:
    _assert_matches_oracle(evaluator_graph, signature, LIVES_IN)


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


SELF = ("a", "self", "a")
CROSS = ("a", "cross", "b")
HEAD_AB = ("a", "head", "b")


@pytest.fixture
def repeated_type_graph() -> HeteroGraph:
    """Node type ``a`` recurs at several levels of one chain."""
    data = HeteroData()
    data["a"].num_nodes = 6
    data["b"].num_nodes = 4
    data[SELF].edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 4]])
    data[CROSS].edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 1]])
    data[HEAD_AB].edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    return HeteroGraph(data)


@pytest.mark.parametrize(
    "signature",
    [(SELF, SELF, CROSS), (SELF, SELF, SELF, CROSS)],
    ids=["node_type_repeated_across_levels", "repeated_edge_type_within_one_chain"],
)
def test_matches_oracle_with_repeated_types(
    repeated_type_graph: HeteroGraph, signature: tuple[EdgeTypeTuple, ...]
) -> None:
    """A node reached at level 1 must still be admissible at level 2.

    Stamping per row rather than per level would suppress it and undercount.
    """
    _assert_matches_oracle(repeated_type_graph, signature, HEAD_AB)


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


@pytest.mark.parametrize(
    ("signature", "match"),
    [
        ((), "at least one edge type"),
        ((NEAR, BORN_IN), "does not join"),
        ((NEAR,), "chain starts at"),
    ],
    ids=["empty", "does_not_join", "start_differs_from_head"],
)
def test_rejects_an_invalid_chain(
    evaluator_graph: HeteroGraph, signature: tuple[EdgeTypeTuple, ...], match: str
) -> None:
    with pytest.raises(InvalidRuleError, match=match):
        ChainScanner(evaluator_graph).scan_all_rows(signature, LIVES_IN)


def test_rejects_a_chain_whose_end_differs_from_the_head(
    repeated_type_graph: HeteroGraph,
) -> None:
    with pytest.raises(InvalidRuleError, match="chain ends at"):
        ChainScanner(repeated_type_graph).scan_all_rows((SELF,), HEAD_AB)


class TestPathCounter:
    @pytest.mark.parametrize(
        "chain",
        [
            (KNOWS, KNOWS, LIVES_IN),
            (NEAR, NEAR),
            (LIVES_IN, NEAR, NEAR),
        ],
        ids=["knows-knows-lives_in", "near-near", "lives_in-near-near"],
    )
    def test_matches_the_dense_chain_product(
        self, simple_graph: HeteroGraph, chain: tuple[EdgeTypeTuple, ...]
    ) -> None:
        product = adjacency.adjacency_matrix(simple_graph, chain[0]).to_dense()
        for edge_type in chain[1:]:
            product = (
                product @ adjacency.adjacency_matrix(simple_graph, edge_type).to_dense()
            )
        counter = PathCounter(simple_graph)

        for source in range(product.size(0)):
            nodes, counts = counter.from_source(chain, source)
            row = torch.zeros(product.size(1))
            row[nodes] = counts
            assert torch.equal(row, product[source])
        for target in range(product.size(1)):
            nodes, counts = counter.into_target(chain, target)
            column = torch.zeros(product.size(0))
            column[nodes] = counts
            assert torch.equal(column, product[:, target])

    def test_rejects_an_unknown_edge_type(self, simple_graph: HeteroGraph) -> None:
        with pytest.raises(GraphSchemaError):
            PathCounter(simple_graph).check((("person", "visited", "city"),))
