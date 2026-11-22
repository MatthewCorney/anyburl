"""Tests for holding out target-relation test and validation edges."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.exceptions import ConfigurationError, GraphSchemaError
from anyburl.split import (
    InverseEdgeHandling,
    SplitConfig,
    split_target_edges,
    split_with_validation,
)

TARGET = ("author", "to", "paper")
INVERSE = ("paper", "to", "author")
NUM_EDGES = 10
VALIDATION_GRAPH_EDGES = 20
HELD_OUT_TEST_FRACTION = 0.25
VALIDATION_FRACTION = 0.2
NUM_HELD_OUT_TEST = 5
NUM_VALIDATION = 3


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

    with pytest.raises(GraphSchemaError, match="not in graph"):
        split_target_edges(mirrored_data, config)


def test_fraction_holding_out_nothing_raises(mirrored_data: HeteroData) -> None:
    config = SplitConfig(target_edge_type=TARGET, test_fraction=0.01)

    with pytest.raises(ConfigurationError, match="holds out 0 of"):
        split_target_edges(mirrored_data, config)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_fraction_raises(fraction: float) -> None:
    with pytest.raises(ConfigurationError, match="test_fraction must be in"):
        SplitConfig(target_edge_type=TARGET, test_fraction=fraction)


def test_split_without_inverse_edge_type_is_a_noop(mirrored_data: HeteroData) -> None:
    """Graphs storing only one direction need no mirror removal."""
    del mirrored_data[INVERSE]

    split = split_target_edges(
        mirrored_data, SplitConfig(target_edge_type=TARGET, test_fraction=0.3)
    )

    assert len(split.test_triples) == 3


@pytest.mark.parametrize("edge_type", [TARGET, INVERSE], ids=["target", "inverse"])
def test_edge_parallel_attribute_raises(
    mirrored_data: HeteroData, edge_type: tuple[str, str, str]
) -> None:
    """Filtering only edge_index would leave edge_attr misaligned.

    The mirrored inverse is filtered too, so it must be checked as well.
    """
    mirrored_data[edge_type].edge_attr = torch.ones(NUM_EDGES, 1)

    with pytest.raises(GraphSchemaError, match="edge-parallel attributes"):
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
    """Inverse edges stored under a different relation name."""
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

    with pytest.raises(GraphSchemaError, match=r"inverse_edge_type .* not in graph"):
        split_target_edges(mirrored_data, config)


def test_explicit_inverse_with_keep_raises() -> None:
    """Naming an inverse to remove while asking to keep it is contradictory."""
    with pytest.raises(ConfigurationError, match="contradict"):
        SplitConfig(
            target_edge_type=TARGET,
            inverse_edge_type=RENAMED_INVERSE,
            inverse_handling=InverseEdgeHandling.KEEP,
        )


@pytest.fixture
def twenty_mirrored_edges() -> HeteroData:
    """Twenty author-paper edges, stored in both directions."""
    data = HeteroData()
    data["author"].num_nodes = VALIDATION_GRAPH_EDGES
    data["paper"].num_nodes = VALIDATION_GRAPH_EDGES
    ids = torch.arange(VALIDATION_GRAPH_EDGES)
    data[TARGET].edge_index = torch.stack([ids, ids])
    data[INVERSE].edge_index = torch.stack([ids, ids])
    return data


@pytest.fixture
def two_way_config() -> SplitConfig:
    return SplitConfig(target_edge_type=TARGET, test_fraction=HELD_OUT_TEST_FRACTION)


def _pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))


class TestSplitWithValidation:
    def test_test_set_matches_two_way_split(
        self, twenty_mirrored_edges: HeteroData, two_way_config: SplitConfig
    ) -> None:
        """Existing test-set results stay comparable."""
        two_way = split_target_edges(twenty_mirrored_edges, two_way_config)

        split = split_with_validation(
            twenty_mirrored_edges,
            two_way_config,
            validation_fraction=VALIDATION_FRACTION,
        )

        assert split.test_triples == two_way.test_triples
        assert torch.equal(
            split.train_data[TARGET].edge_index, two_way.train_data[TARGET].edge_index
        )

    def test_holds_out_fraction_of_training_edges(
        self, twenty_mirrored_edges: HeteroData, two_way_config: SplitConfig
    ) -> None:
        split = split_with_validation(
            twenty_mirrored_edges,
            two_way_config,
            validation_fraction=VALIDATION_FRACTION,
        )

        assert len(split.test_triples) == NUM_HELD_OUT_TEST
        assert len(split.validation_triples) == NUM_VALIDATION
        assert split.fit_data[TARGET].edge_index.size(1) == (
            VALIDATION_GRAPH_EDGES - NUM_HELD_OUT_TEST - NUM_VALIDATION
        )

    def test_partitions_target_edges(
        self, twenty_mirrored_edges: HeteroData, two_way_config: SplitConfig
    ) -> None:
        split = split_with_validation(
            twenty_mirrored_edges,
            two_way_config,
            validation_fraction=VALIDATION_FRACTION,
        )

        fit = _pairs(split.fit_data[TARGET].edge_index)
        validation = {(t.head_id, t.tail_id) for t in split.validation_triples}
        test = {(t.head_id, t.tail_id) for t in split.test_triples}

        assert fit.isdisjoint(validation)
        assert fit.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert fit | validation | test == _pairs(
            twenty_mirrored_edges[TARGET].edge_index
        )

    def test_fit_data_drops_mirrors_of_held_out_edges(
        self, twenty_mirrored_edges: HeteroData, two_way_config: SplitConfig
    ) -> None:
        """Either held-out set left in reverse would be reachable in one hop."""
        split = split_with_validation(
            twenty_mirrored_edges,
            two_way_config,
            validation_fraction=VALIDATION_FRACTION,
        )

        reverse = split.fit_data[INVERSE].edge_index
        surviving = set(zip(reverse[1].tolist(), reverse[0].tolist(), strict=True))
        held_out = {
            (t.head_id, t.tail_id)
            for t in (*split.validation_triples, *split.test_triples)
        }

        assert surviving.isdisjoint(held_out)

    def test_train_data_keeps_validation_edges(
        self, twenty_mirrored_edges: HeteroData, two_way_config: SplitConfig
    ) -> None:
        """Validation edges must count as known facts when ranking test triples."""
        split = split_with_validation(
            twenty_mirrored_edges,
            two_way_config,
            validation_fraction=VALIDATION_FRACTION,
        )

        validation = {(t.head_id, t.tail_id) for t in split.validation_triples}

        assert validation <= _pairs(split.train_data[TARGET].edge_index)

    def test_is_deterministic_for_a_seed(
        self, twenty_mirrored_edges: HeteroData, two_way_config: SplitConfig
    ) -> None:
        first = split_with_validation(twenty_mirrored_edges, two_way_config)
        second = split_with_validation(twenty_mirrored_edges, two_way_config)

        assert first.validation_triples == second.validation_triples

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_validation_fraction_raises(
        self,
        twenty_mirrored_edges: HeteroData,
        two_way_config: SplitConfig,
        fraction: float,
    ) -> None:
        with pytest.raises(ConfigurationError, match="validation_fraction must be in"):
            split_with_validation(
                twenty_mirrored_edges, two_way_config, validation_fraction=fraction
            )

    def test_fraction_holding_out_nothing_names_validation(
        self, twenty_mirrored_edges: HeteroData, two_way_config: SplitConfig
    ) -> None:
        with pytest.raises(ConfigurationError, match=r"validation_fraction 0\.01"):
            split_with_validation(
                twenty_mirrored_edges, two_way_config, validation_fraction=0.01
            )
