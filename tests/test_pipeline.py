"""Tests for the end-to-end AnyBURL pipeline."""

import pytest
from torch_geometric.data import HeteroData

from anyburl import AnyBURL, AnyBURLConfig, AnytimeConfig
from anyburl.exceptions import ConfigurationError
from anyburl.graph import HeteroGraph

LIVES_IN = ("person", "lives_in", "city")
CONFIG = AnyBURLConfig(
    sample_size=20,
    min_walk_length=1,
    max_walk_length=3,
    max_walk_attempts=200,
    target_edge_type=LIVES_IN,
)


class TestFit:
    def test_accepts_a_prebuilt_graph(self, simple_hetero_data: HeteroData) -> None:
        graph = HeteroGraph(simple_hetero_data)

        pipeline = AnyBURL(CONFIG).fit(graph)

        assert pipeline.graph is graph

    def test_prebuilt_graph_learns_the_same_rules(
        self, simple_hetero_data: HeteroData
    ) -> None:
        from_data = AnyBURL(CONFIG).fit(simple_hetero_data)
        from_graph = AnyBURL(CONFIG).fit(HeteroGraph(simple_hetero_data))

        assert from_data.results
        assert from_graph.results == from_data.results
        assert from_graph.rules == from_data.rules


class TestFitAnytime:
    def test_rejects_acyclic_sampling(self, simple_hetero_data: HeteroData) -> None:
        config = AnyBURLConfig(target_edge_type=LIVES_IN, acyclic_samples_per_end=2)

        with pytest.raises(ConfigurationError, match="acyclic"):
            AnyBURL(config).fit_anytime(simple_hetero_data)

    def test_rules_are_every_candidate_like_fit(
        self, simple_hetero_data: HeteroData
    ) -> None:
        pipeline = AnyBURL(CONFIG).fit_anytime(
            simple_hetero_data, AnytimeConfig(min_length=1, max_length=3)
        )

        passing = [rule for rule, _ in pipeline.results]
        assert passing
        assert set(passing) <= set(pipeline.rules)
        assert len(pipeline.rules) == len(set(pipeline.rules))
        assert len(pipeline.rules) > len(passing)
