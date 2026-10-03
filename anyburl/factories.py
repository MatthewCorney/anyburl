"""Build samplers and walk engines from their configurations."""

from typing import assert_never

import torch

from .graph import HeteroGraph
from .sampler import (
    BaseTripleSampler,
    EntityBalancedTripleSampler,
    SamplerConfig,
    SamplingStrategy,
    UniformTripleSampler,
    WeightedTripleSampler,
)
from .walk import (
    NumbaWalkEngine,
    RelationWeightedEdgeSelector,
    WalkConfig,
    WalkEngine,
    WalkStrategy,
)

__all__ = ["build_triple_sampler", "build_walk_engine"]


def build_triple_sampler(
    graph: HeteroGraph,
    config: SamplerConfig,
) -> BaseTripleSampler:
    """Build a triple sampler for the configured strategy.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    config : SamplerConfig
        Sampler configuration.

    Returns
    -------
    BaseTripleSampler
        A sampler instance.
    """
    match config.strategy:
        case SamplingStrategy.UNIFORM:
            return UniformTripleSampler(graph, config)
        case SamplingStrategy.ENTITY_BALANCED:
            return EntityBalancedTripleSampler(graph, config)
        case SamplingStrategy.RELATION_PROPORTIONAL:
            weights = torch.ones(len(_non_empty_edge_counts(graph)))
            return WeightedTripleSampler(graph, config, weights)
        case SamplingStrategy.RELATION_INVERSE:
            inverse = 1.0 / _non_empty_edge_counts(graph)
            return WeightedTripleSampler(graph, config, inverse / inverse.sum())
        case _ as unreachable:
            assert_never(unreachable)


def _non_empty_edge_counts(graph: HeteroGraph) -> torch.Tensor:
    """Return the edge count of every edge type that has edges, in graph order."""
    return torch.tensor(
        [graph.edge_count(et) for et in graph.edge_types if graph.edge_count(et) > 0],
        dtype=torch.float32,
    )


def build_walk_engine(
    graph: HeteroGraph,
    config: WalkConfig,
) -> WalkEngine | NumbaWalkEngine:
    """Build a walk engine for the configured strategy.

    Uniform and reachability-pruned walks use :class:`NumbaWalkEngine`;
    relation-weighted walks use :class:`WalkEngine`.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    config : WalkConfig
        Walk configuration.

    Returns
    -------
    WalkEngine | NumbaWalkEngine
        A walk engine instance.
    """
    match config.strategy:
        case WalkStrategy.UNIFORM | WalkStrategy.REACHABILITY_PRUNED:
            return NumbaWalkEngine(graph, config)
        case WalkStrategy.RELATION_WEIGHTED:
            generator = torch.Generator().manual_seed(config.seed)
            selector = RelationWeightedEdgeSelector(graph, generator)
            return WalkEngine(graph, config, selector)
        case _ as unreachable:
            assert_never(unreachable)
