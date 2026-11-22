"""Sampling target triples: configuration, the samplers and their factory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import torch
from torch import Tensor

from ._logging import get_logger
from .core import DEFAULT_RANDOM_SEED, EdgeTypeTuple, Triple
from .exceptions import ConfigurationError, GraphSchemaError
from .graph import HeteroGraph

__all__ = [
    "BaseTripleSampler",
    "EntityBalancedTripleSampler",
    "SamplerConfig",
    "SamplingStrategy",
    "UniformTripleSampler",
    "WeightedTripleSampler",
    "build_triple_sampler",
]

logger = get_logger(__name__)

DEFAULT_SAMPLE_SIZE: int = 1000
EDGE_TYPE_TUPLE_LENGTH: int = 3


class SamplingStrategy(StrEnum):
    """Strategy for sampling target triples from the knowledge graph.

    Attributes
    ----------
    UNIFORM : str
        Each *edge* has equal probability of being sampled. More common
        relations naturally yield more samples.
    RELATION_PROPORTIONAL : str
        Each *relation type* has equal selection probability, then
        sample uniformly within the chosen type. Ensures balanced
        representation regardless of edge count.
    RELATION_INVERSE : str
        Sample inversely proportional to relation frequency. Rarer
        relations get more samples, improving coverage.
    ENTITY_BALANCED : str
        Draw a *head entity* uniformly, then one of its edges, so
        high-degree entities are not over-represented.
    """

    UNIFORM = "uniform"
    RELATION_PROPORTIONAL = "relation_proportional"
    RELATION_INVERSE = "relation_inverse"
    ENTITY_BALANCED = "entity_balanced"


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """Configuration for sampling target triples from the knowledge graph.

    Parameters
    ----------
    sample_size : int
        Number of triples to sample per iteration.
    strategy : SamplingStrategy
        How to weight triple selection.
    seed : int
        Random seed for reproducibility.
    target_edge_type : EdgeTypeTuple | None
        When set, only sample triples from this edge type.
        When ``None`` (default), all edge types are eligible.

    Raises
    ------
    ConfigurationError
        If ``sample_size`` is not positive or ``target_edge_type``
        is not a 3-tuple of strings.
    """

    sample_size: int = DEFAULT_SAMPLE_SIZE
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    seed: int = DEFAULT_RANDOM_SEED
    target_edge_type: EdgeTypeTuple | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.sample_size < 1:
            raise ConfigurationError(
                f"sample_size must be positive, got {self.sample_size}"
            )
        if self.target_edge_type is not None and (
            not isinstance(self.target_edge_type, tuple)
            or len(self.target_edge_type) != EDGE_TYPE_TUPLE_LENGTH
            or not all(isinstance(s, str) for s in self.target_edge_type)
        ):
            raise ConfigurationError(
                f"target_edge_type must be a 3-tuple of strings, "
                f"got {self.target_edge_type!r}"
            )


class BaseTripleSampler(ABC):
    """Base class holding the eligible edge types and their edge counts.

    Parameters
    ----------
    graph : HeteroGraph
        The graph to sample from.
    config : SamplerConfig
        Sample size, seed and optional target edge type.

    Raises
    ------
    GraphSchemaError
        If the target edge type is missing or empty, or no edge type has
        any edges.
    """

    def __init__(self, graph: HeteroGraph, config: SamplerConfig) -> None:
        self._graph = graph
        self._config = config
        self._generator = torch.Generator().manual_seed(config.seed)

        self._edge_types = [et for et in graph.edge_types if graph.edge_count(et) > 0]

        if config.target_edge_type is not None:
            target = config.target_edge_type
            if target not in self._edge_types:
                raise GraphSchemaError(
                    f"Target edge type {target!r} not found in graph or has zero edges"
                )
            self._edge_types = [target]

        self._edge_counts = torch.tensor(
            [graph.edge_count(et) for et in self._edge_types],
            dtype=torch.long,
        )

        if len(self._edge_counts) == 0:
            raise GraphSchemaError("No eligible edge types available for sampling.")

        self._cumulative_counts = torch.cumsum(self._edge_counts, dim=0)
        self._total_edges = int(self._cumulative_counts[-1].item())

    def _extract_triple(
        self,
        edge_type: EdgeTypeTuple,
        local_idx: int,
    ) -> Triple:
        """Return the triple stored at ``local_idx`` within ``edge_type``."""
        ei: Tensor = self._graph.edge_index(edge_type)
        src_type, relation, dst_type = edge_type
        return Triple(
            head_id=int(ei[0, local_idx].item()),
            tail_id=int(ei[1, local_idx].item()),
            head_type=src_type,
            tail_type=dst_type,
            relation=relation,
        )

    def _sample_size(self) -> int:
        """Return the number of triples to draw, capped by the edge count."""
        return min(self._config.sample_size, self._total_edges)

    @abstractmethod
    def sample(self) -> list[Triple]:
        """Sample target triples from the graph.

        Returns
        -------
        list[Triple]
            At most ``config.sample_size`` triples.
        """


class UniformTripleSampler(BaseTripleSampler):
    """Uniform sampling across all edges."""

    def sample(self) -> list[Triple]:
        """Sample triples uniformly at random across all edges."""
        n = self._sample_size()

        flat_indices = torch.randint(
            0,
            self._total_edges,
            (n,),
            generator=self._generator,
        )

        type_indices = torch.searchsorted(
            self._cumulative_counts,
            flat_indices,
            right=True,
        )

        triples: list[Triple] = []

        for type_idx, flat_idx in zip(
            type_indices.tolist(),
            flat_indices.tolist(),
            strict=True,
        ):
            offset = (
                int(self._cumulative_counts[type_idx - 1].item()) if type_idx > 0 else 0
            )
            local_idx = flat_idx - offset
            triples.append(self._extract_triple(self._edge_types[type_idx], local_idx))

        logger.debug("Sampled %d triples via uniform strategy", len(triples))
        return triples


class EntityBalancedTripleSampler(BaseTripleSampler):
    """Samples a head entity uniformly, then one of its edges.

    Uniform edge sampling picks a head with probability proportional to its
    degree; drawing the head first spreads the sample evenly across
    entities. With several eligible edge types the budget is split evenly
    between them and balanced within each.
    """

    def sample(self) -> list[Triple]:
        """Sample triples with every head entity equally likely.

        Returns
        -------
        list[Triple]
            Sampled triples, at most ``sample_size`` of them.
        """
        wanted = self._sample_size()
        per_type = self._per_type_quota(wanted)

        triples: list[Triple] = []
        for edge_type, quota in zip(self._edge_types, per_type, strict=True):
            if quota > 0:
                triples.extend(self._sample_within(edge_type, quota))

        logger.debug("Sampled %d triples via entity-balanced strategy", len(triples))
        return triples[:wanted]

    def _per_type_quota(self, wanted: int) -> list[int]:
        """Split the sample budget evenly across eligible edge types."""
        types = len(self._edge_types)
        base, remainder = divmod(wanted, types)
        return [base + (1 if i < remainder else 0) for i in range(types)]

    def _sample_within(self, edge_type: EdgeTypeTuple, quota: int) -> list[Triple]:
        """Draw ``quota`` triples of one edge type, balanced across heads.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The edge type to sample from.
        quota : int
            How many triples to draw.

        Returns
        -------
        list[Triple]
            Sampled triples.
        """
        edge_index = self._graph.edge_index(edge_type)
        heads = edge_index[0]
        present, inverse = torch.unique(heads, return_inverse=True)

        chosen_heads = torch.randint(
            0, present.numel(), (quota,), generator=self._generator
        )

        order = torch.argsort(inverse, stable=True)
        starts = torch.searchsorted(inverse[order], torch.arange(present.numel()))
        degrees = torch.bincount(inverse, minlength=present.numel())

        offsets = (
            torch.rand(quota, generator=self._generator) * degrees[chosen_heads]
        ).long()
        edge_positions = order[starts[chosen_heads] + offsets]

        return [
            self._extract_triple(edge_type, int(position))
            for position in edge_positions.tolist()
        ]


class WeightedTripleSampler(BaseTripleSampler):
    """Weighted sampling over edge types, uniform within each type.

    Parameters
    ----------
    graph : HeteroGraph
        The graph to sample from.
    config : SamplerConfig
        Sample size, seed and optional target edge type.
    weights : Tensor
        Selection weight per eligible edge type, in graph order.

    Raises
    ------
    ConfigurationError
        If ``weights`` does not have one entry per eligible edge type.
    """

    def __init__(
        self,
        graph: HeteroGraph,
        config: SamplerConfig,
        weights: Tensor,
    ) -> None:
        super().__init__(graph, config)

        if weights.shape[0] != len(self._edge_types):
            raise ConfigurationError("Weights must match number of edge types.")

        self._weights = weights.float()

    def sample(self) -> list[Triple]:
        """Sample triples according to the configured edge-type weights."""
        n = self._sample_size()
        num_types = len(self._edge_types)

        type_selections = torch.multinomial(
            self._weights,
            n,
            replacement=True,
            generator=self._generator,
        )

        type_counts = torch.bincount(
            type_selections,
            minlength=num_types,
        )

        local_indices_by_type: list[Tensor] = []

        for type_idx in range(num_types):
            count = int(type_counts[type_idx].item())
            if count == 0:
                local_indices_by_type.append(torch.empty(0, dtype=torch.long))
                continue

            edge_count = int(self._edge_counts[type_idx].item())

            local_indices_by_type.append(
                torch.randint(
                    0,
                    edge_count,
                    (count,),
                    generator=self._generator,
                )
            )

        consumed = [0] * num_types
        triples: list[Triple] = []

        for type_idx in type_selections.tolist():
            local_idx = int(local_indices_by_type[type_idx][consumed[type_idx]].item())
            consumed[type_idx] += 1

            triples.append(
                self._extract_triple(
                    self._edge_types[type_idx],
                    local_idx,
                )
            )

        logger.debug("Sampled %d triples via weighted strategy", len(triples))
        return triples


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
