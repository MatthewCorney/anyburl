"""Tests for holding out validation edges alongside test edges."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.exceptions import ConfigurationError
from anyburl.split import SplitConfig, split_target_edges
from anyburl.validation_split import split_with_validation

TARGET = ("author", "to", "paper")
INVERSE = ("paper", "to", "author")
NUM_EDGES = 20
TEST_FRACTION = 0.25
VALIDATION_FRACTION = 0.2
NUM_TEST = 5
NUM_VALIDATION = 3


@pytest.fixture
def mirrored_data() -> HeteroData:
    """Twenty author-paper edges, stored in both directions."""
    data = HeteroData()
    data["author"].num_nodes = NUM_EDGES
    data["paper"].num_nodes = NUM_EDGES
    ids = torch.arange(NUM_EDGES)
    data[TARGET].edge_index = torch.stack([ids, ids])
    data[INVERSE].edge_index = torch.stack([ids, ids])
    return data


@pytest.fixture
def config() -> SplitConfig:
    return SplitConfig(target_edge_type=TARGET, test_fraction=TEST_FRACTION)


def _pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))


class TestSplitWithValidation:
    def test_test_set_matches_two_way_split(
        self, mirrored_data: HeteroData, config: SplitConfig
    ) -> None:
        """Existing test-set results stay comparable."""
        two_way = split_target_edges(mirrored_data, config)

        split = split_with_validation(
            mirrored_data, config, validation_fraction=VALIDATION_FRACTION
        )

        assert split.test_triples == two_way.test_triples
        assert torch.equal(
            split.train_data[TARGET].edge_index, two_way.train_data[TARGET].edge_index
        )

    def test_holds_out_fraction_of_training_edges(
        self, mirrored_data: HeteroData, config: SplitConfig
    ) -> None:
        split = split_with_validation(
            mirrored_data, config, validation_fraction=VALIDATION_FRACTION
        )

        assert len(split.test_triples) == NUM_TEST
        assert len(split.validation_triples) == NUM_VALIDATION
        assert split.fit_data[TARGET].edge_index.size(1) == (
            NUM_EDGES - NUM_TEST - NUM_VALIDATION
        )

    def test_partitions_target_edges(
        self, mirrored_data: HeteroData, config: SplitConfig
    ) -> None:
        split = split_with_validation(
            mirrored_data, config, validation_fraction=VALIDATION_FRACTION
        )

        fit = _pairs(split.fit_data[TARGET].edge_index)
        validation = {(t.head_id, t.tail_id) for t in split.validation_triples}
        test = {(t.head_id, t.tail_id) for t in split.test_triples}

        assert fit.isdisjoint(validation)
        assert fit.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert fit | validation | test == _pairs(mirrored_data[TARGET].edge_index)

    def test_fit_data_drops_mirrors_of_held_out_edges(
        self, mirrored_data: HeteroData, config: SplitConfig
    ) -> None:
        """Either held-out set left in reverse would be reachable in one hop."""
        split = split_with_validation(
            mirrored_data, config, validation_fraction=VALIDATION_FRACTION
        )

        reverse = split.fit_data[INVERSE].edge_index
        surviving = set(zip(reverse[1].tolist(), reverse[0].tolist(), strict=True))
        held_out = {
            (t.head_id, t.tail_id)
            for t in (*split.validation_triples, *split.test_triples)
        }

        assert surviving.isdisjoint(held_out)

    def test_train_data_keeps_validation_edges(
        self, mirrored_data: HeteroData, config: SplitConfig
    ) -> None:
        """Validation edges must count as known facts when ranking test triples."""
        split = split_with_validation(
            mirrored_data, config, validation_fraction=VALIDATION_FRACTION
        )

        validation = {(t.head_id, t.tail_id) for t in split.validation_triples}

        assert validation <= _pairs(split.train_data[TARGET].edge_index)

    def test_is_deterministic_for_a_seed(
        self, mirrored_data: HeteroData, config: SplitConfig
    ) -> None:
        first = split_with_validation(mirrored_data, config)
        second = split_with_validation(mirrored_data, config)

        assert first.validation_triples == second.validation_triples

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_validation_fraction_raises(
        self, mirrored_data: HeteroData, config: SplitConfig, fraction: float
    ) -> None:
        with pytest.raises(ConfigurationError, match="validation_fraction must be in"):
            split_with_validation(mirrored_data, config, validation_fraction=fraction)

    def test_fraction_holding_out_nothing_names_validation(
        self, mirrored_data: HeteroData, config: SplitConfig
    ) -> None:
        with pytest.raises(ConfigurationError, match=r"validation_fraction 0\.01"):
            split_with_validation(mirrored_data, config, validation_fraction=0.01)
