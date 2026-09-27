"""Tests for triple samplers."""

from collections import Counter

import pytest
import torch

from anyburl.anyburl import build_triple_sampler
from anyburl.graph import HeteroGraph
from anyburl.sampler import (
    EntityBalancedTripleSampler,
    SamplerConfig,
    SamplingStrategy,
    Triple,
    UniformTripleSampler,
    WeightedTripleSampler,
)


def test_triple_is_frozen() -> None:
    t = Triple(head_id=0, tail_id=1, head_type="A", tail_type="B", relation="r")
    with pytest.raises(AttributeError):
        t.head_id = 2  # type: ignore[misc]


@pytest.mark.parametrize("invalid_size", [0, -1])
def test_sampler_config_invalid_sample_size(invalid_size: int) -> None:
    with pytest.raises(ValueError, match="sample_size must be positive"):
        SamplerConfig(sample_size=invalid_size)


def test_sampler_config_invalid_target_edge_type() -> None:
    with pytest.raises(ValueError, match="target_edge_type must be a 3-tuple"):
        SamplerConfig(target_edge_type=("only_two_elements",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "strategy",
    [
        SamplingStrategy.UNIFORM,
        SamplingStrategy.RELATION_PROPORTIONAL,
        SamplingStrategy.RELATION_INVERSE,
    ],
)
def test_build_triple_sampler_all_strategies(
    simple_graph: HeteroGraph,
    strategy: SamplingStrategy,
) -> None:
    config = SamplerConfig(sample_size=5, strategy=strategy)
    sampler = build_triple_sampler(simple_graph, config)
    triples = sampler.sample()
    assert len(triples) == 5
    for t in triples:
        assert isinstance(t, Triple)


def test_uniform_sample_all_valid_triples(simple_graph: HeteroGraph) -> None:
    config = SamplerConfig(sample_size=10, strategy=SamplingStrategy.UNIFORM)
    sampler = build_triple_sampler(simple_graph, config)
    triples = sampler.sample()
    for t in triples:
        assert isinstance(t, Triple)
        assert t.head_type in simple_graph.node_types
        assert t.tail_type in simple_graph.node_types


def test_uniform_sample_reproducible_with_seed(simple_graph: HeteroGraph) -> None:
    config = SamplerConfig(sample_size=5, strategy=SamplingStrategy.UNIFORM, seed=123)
    s1 = build_triple_sampler(simple_graph, config)
    s2 = build_triple_sampler(simple_graph, config)
    assert s1.sample() == s2.sample()


def test_uniform_sample_capped_at_total(simple_graph: HeteroGraph) -> None:
    config = SamplerConfig(sample_size=100, strategy=SamplingStrategy.UNIFORM)
    sampler = build_triple_sampler(simple_graph, config)
    triples = sampler.sample()
    assert len(triples) == simple_graph.total_edge_count()


def test_inverse_sample_favors_rare_relations(simple_graph: HeteroGraph) -> None:
    config = SamplerConfig(
        sample_size=1000, strategy=SamplingStrategy.RELATION_INVERSE, seed=42
    )
    sampler = build_triple_sampler(simple_graph, config)
    triples = sampler.sample()
    relation_counts: dict[str, int] = {}
    for t in triples:
        relation_counts[t.relation] = relation_counts.get(t.relation, 0) + 1

    # Inverse weighting: each rare edge gets higher per-edge sampling rate.
    # simple_graph has near=3 edges, lives_in=6 edges.
    near_count = relation_counts.get("near", 0)
    lives_in_count = relation_counts.get("lives_in", 0)
    assert near_count / 3 > lives_in_count / 6


def test_target_edge_type_filters_sampling(simple_graph: HeteroGraph) -> None:
    target = ("person", "knows", "person")
    config = SamplerConfig(
        sample_size=50,
        strategy=SamplingStrategy.UNIFORM,
        target_edge_type=target,
    )
    sampler = build_triple_sampler(simple_graph, config)
    triples = sampler.sample()
    for t in triples:
        assert t.relation == "knows"
        assert t.head_type == "person"
        assert t.tail_type == "person"


def test_target_edge_type_unknown_raises(simple_graph: HeteroGraph) -> None:
    target = ("person", "works_at", "company")
    config = SamplerConfig(
        sample_size=10,
        target_edge_type=target,
    )
    with pytest.raises(ValueError, match="not found in graph"):
        build_triple_sampler(simple_graph, config)


def test_weighted_sampler_wrong_weight_count(simple_graph: HeteroGraph) -> None:
    config = SamplerConfig(
        sample_size=5, strategy=SamplingStrategy.RELATION_PROPORTIONAL
    )
    wrong_weights = torch.ones(99)
    with pytest.raises(ValueError, match="Weights must match"):
        WeightedTripleSampler(simple_graph, config, wrong_weights)


def test_entity_balanced_sampler_returns_requested_count(
    simple_graph: HeteroGraph,
) -> None:
    config = SamplerConfig(
        sample_size=10, strategy=SamplingStrategy.ENTITY_BALANCED, seed=1
    )

    triples = EntityBalancedTripleSampler(simple_graph, config).sample()

    assert len(triples) == 10


def test_entity_balanced_sampler_caps_at_the_edge_count(
    simple_graph: HeteroGraph,
) -> None:
    """Like the other samplers, it never returns more triples than edges."""
    config = SamplerConfig(
        sample_size=10_000, strategy=SamplingStrategy.ENTITY_BALANCED, seed=1
    )

    triples = EntityBalancedTripleSampler(simple_graph, config).sample()

    assert len(triples) == simple_graph.total_edge_count()


def test_entity_balanced_sampler_is_deterministic(
    simple_graph: HeteroGraph,
) -> None:
    config = SamplerConfig(
        sample_size=20, strategy=SamplingStrategy.ENTITY_BALANCED, seed=7
    )

    first = EntityBalancedTripleSampler(simple_graph, config).sample()
    second = EntityBalancedTripleSampler(simple_graph, config).sample()

    assert first == second


def test_entity_balanced_sampler_yields_real_edges(
    simple_graph: HeteroGraph,
) -> None:
    """Every sampled triple must be an edge that exists."""
    target = ("person", "lives_in", "city")
    config = SamplerConfig(
        sample_size=30,
        strategy=SamplingStrategy.ENTITY_BALANCED,
        target_edge_type=target,
        seed=3,
    )

    triples = EntityBalancedTripleSampler(simple_graph, config).sample()

    edge_index = simple_graph.edge_index(target)
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    assert all((t.head_id, t.tail_id) in edges for t in triples)


def test_entity_balanced_sampler_flattens_degree_skew(
    simple_graph: HeteroGraph,
) -> None:
    """A hub must not dominate the sample the way uniform edge draws allow."""
    target = ("person", "lives_in", "city")
    kwargs = {"sample_size": 400, "target_edge_type": target, "seed": 11}

    balanced = EntityBalancedTripleSampler(
        simple_graph,
        SamplerConfig(strategy=SamplingStrategy.ENTITY_BALANCED, **kwargs),
    ).sample()
    uniform = UniformTripleSampler(
        simple_graph, SamplerConfig(strategy=SamplingStrategy.UNIFORM, **kwargs)
    ).sample()

    def top_share(triples: list[Triple]) -> float:
        counts = Counter(t.head_id for t in triples)
        return counts.most_common(1)[0][1] / len(triples)

    assert top_share(balanced) <= top_share(uniform)
