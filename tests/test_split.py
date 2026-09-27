"""Tests for holding out target-relation edges."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.split import (
    InverseEdgeHandling,
    SplitConfig,
    mirrored_edge_type,
    split_target_edges,
)

TARGET = ("author", "to", "paper")
INVERSE = ("paper", "to", "author")
NUM_EDGES = 10


@pytest.fixture
def mirrored_data() -> HeteroData:
    """Ten author-paper edges, stored in both directions."""
    data = HeteroData()
    data["author"].num_nodes = NUM_EDGES
    data["paper"].num_nodes = NUM_EDGES
    heads = torch.arange(NUM_EDGES)
    tails = torch.arange(NUM_EDGES)
    data[TARGET].edge_index = torch.stack([heads, tails])
    data[INVERSE].edge_index = torch.stack([tails, heads])
    return data


def test_mirrored_edge_type_swaps_endpoints() -> None:
    assert mirrored_edge_type(TARGET) == INVERSE


def test_split_removes_test_edges_from_training_graph(
    mirrored_data: HeteroData,
) -> None:
    split = split_target_edges(
        mirrored_data, SplitConfig(target_edge_type=TARGET, test_fraction=0.3)
    )

    assert len(split.test_triples) == 3
    assert split.train_data[TARGET].edge_index.size(1) == NUM_EDGES - 3


def test_split_leaves_input_graph_untouched(mirrored_data: HeteroData) -> None:
    split_target_edges(
        mirrored_data, SplitConfig(target_edge_type=TARGET, test_fraction=0.3)
    )

    assert mirrored_data[TARGET].edge_index.size(1) == NUM_EDGES


def test_split_removes_mirrored_inverse_edges(mirrored_data: HeteroData) -> None:
    """A held-out edge left in the reverse direction is reachable in one hop."""
    split = split_target_edges(
        mirrored_data, SplitConfig(target_edge_type=TARGET, test_fraction=0.3)
    )

    reverse = split.train_data[INVERSE].edge_index
    surviving = set(zip(reverse[1].tolist(), reverse[0].tolist(), strict=True))
    held_out = {(t.head_id, t.tail_id) for t in split.test_triples}

    assert surviving.isdisjoint(held_out)


def test_keep_inverse_leaves_reverse_edges_intact(mirrored_data: HeteroData) -> None:
    split = split_target_edges(
        mirrored_data,
        SplitConfig(
            target_edge_type=TARGET,
            test_fraction=0.3,
            inverse_handling=InverseEdgeHandling.KEEP,
        ),
    )

    assert split.train_data[INVERSE].edge_index.size(1) == NUM_EDGES


def test_test_triples_carry_target_types(mirrored_data: HeteroData) -> None:
    split = split_target_edges(
        mirrored_data, SplitConfig(target_edge_type=TARGET, test_fraction=0.3)
    )

    triple = split.test_triples[0]
    assert (triple.head_type, triple.relation, triple.tail_type) == TARGET


def test_split_is_deterministic_for_a_seed(mirrored_data: HeteroData) -> None:
    config = SplitConfig(target_edge_type=TARGET, test_fraction=0.3, seed=7)

    first = split_target_edges(mirrored_data, config).test_triples
    second = split_target_edges(mirrored_data, config).test_triples

    assert first == second


def test_unknown_target_edge_type_raises(mirrored_data: HeteroData) -> None:
    config = SplitConfig(target_edge_type=("author", "cites", "paper"))

    with pytest.raises(ValueError, match="not in graph"):
        split_target_edges(mirrored_data, config)


def test_fraction_holding_out_nothing_raises(mirrored_data: HeteroData) -> None:
    config = SplitConfig(target_edge_type=TARGET, test_fraction=0.01)

    with pytest.raises(ValueError, match="holds out 0 of"):
        split_target_edges(mirrored_data, config)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_fraction_raises(fraction: float) -> None:
    with pytest.raises(ValueError, match="test_fraction must be in"):
        SplitConfig(target_edge_type=TARGET, test_fraction=fraction)


def test_split_without_inverse_edge_type_is_a_noop(mirrored_data: HeteroData) -> None:
    """Graphs storing only one direction need no mirror removal."""
    del mirrored_data[INVERSE]

    split = split_target_edges(
        mirrored_data, SplitConfig(target_edge_type=TARGET, test_fraction=0.3)
    )

    assert len(split.test_triples) == 3


def test_edge_parallel_attribute_on_target_raises(mirrored_data: HeteroData) -> None:
    """Filtering only edge_index would leave edge_attr misaligned."""
    mirrored_data[TARGET].edge_attr = torch.ones(NUM_EDGES, 1)

    with pytest.raises(ValueError, match="edge-parallel attributes"):
        split_target_edges(
            mirrored_data, SplitConfig(target_edge_type=TARGET, test_fraction=0.3)
        )


def test_edge_parallel_attribute_on_inverse_raises(mirrored_data: HeteroData) -> None:
    """The mirrored inverse is filtered too, so it must be checked as well."""
    mirrored_data[INVERSE].edge_attr = torch.ones(NUM_EDGES, 1)

    with pytest.raises(ValueError, match="edge-parallel attributes"):
        split_target_edges(
            mirrored_data, SplitConfig(target_edge_type=TARGET, test_fraction=0.3)
        )


def test_inverse_attribute_ignored_when_inverse_is_kept(
    mirrored_data: HeteroData,
) -> None:
    """KEEP never touches the inverse, so its attributes cannot desynchronise."""
    mirrored_data[INVERSE].edge_attr = torch.ones(NUM_EDGES, 1)

    split = split_target_edges(
        mirrored_data,
        SplitConfig(
            target_edge_type=TARGET,
            test_fraction=0.3,
            inverse_handling=InverseEdgeHandling.KEEP,
        ),
    )

    assert len(split.test_triples) == 3


RENAMED_INVERSE = ("paper", "authored_by", "author")


@pytest.fixture
def renamed_inverse_data() -> HeteroData:
    """Inverse edges stored under a DIFFERENT relation name, as BIOKG does."""
    data = HeteroData()
    data["author"].num_nodes = NUM_EDGES
    data["paper"].num_nodes = NUM_EDGES
    ids = torch.arange(NUM_EDGES)
    data[TARGET].edge_index = torch.stack([ids, ids])
    data[RENAMED_INVERSE].edge_index = torch.stack([ids, ids])
    return data


def test_differently_named_inverse_is_not_found_by_default(
    renamed_inverse_data: HeteroData,
) -> None:
    """The name-mirrored default cannot see an inverse under another name."""
    split = split_target_edges(
        renamed_inverse_data,
        SplitConfig(target_edge_type=TARGET, test_fraction=0.3),
    )

    assert split.train_data[RENAMED_INVERSE].edge_index.size(1) == NUM_EDGES


def test_explicit_inverse_edge_type_is_removed(
    renamed_inverse_data: HeteroData,
) -> None:
    """Naming the inverse closes the one-hop leak."""
    split = split_target_edges(
        renamed_inverse_data,
        SplitConfig(
            target_edge_type=TARGET,
            test_fraction=0.3,
            inverse_edge_type=RENAMED_INVERSE,
        ),
    )

    reverse = split.train_data[RENAMED_INVERSE].edge_index
    surviving = set(zip(reverse[1].tolist(), reverse[0].tolist(), strict=True))
    held_out = {(t.head_id, t.tail_id) for t in split.test_triples}

    assert surviving.isdisjoint(held_out)
    assert reverse.size(1) == NUM_EDGES - 3


def test_explicit_inverse_absent_from_graph_raises(
    mirrored_data: HeteroData,
) -> None:
    """A typo would silently leak the whole test set, so it must fail."""
    config = SplitConfig(
        target_edge_type=TARGET,
        test_fraction=0.3,
        inverse_edge_type=("paper", "typo", "author"),
    )

    with pytest.raises(ValueError, match=r"inverse_edge_type .* not in graph"):
        split_target_edges(mirrored_data, config)


def test_explicit_inverse_with_keep_raises() -> None:
    """Naming an inverse to remove while asking to keep it is contradictory."""
    with pytest.raises(ValueError, match="contradict"):
        SplitConfig(
            target_edge_type=TARGET,
            inverse_edge_type=RENAMED_INVERSE,
            inverse_handling=InverseEdgeHandling.KEEP,
        )
