from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor

from ..graph import EdgeTypeTuple, HeteroGraph

DEFAULT_SAMPLE_SIZE: int = 1000
DEFAULT_RANDOM_SEED: int = 42
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
    """

    UNIFORM = "uniform"
    RELATION_PROPORTIONAL = "relation_proportional"
    RELATION_INVERSE = "relation_inverse"


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
    ValueError
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
            raise ValueError(f"sample_size must be positive, got {self.sample_size}")
        if self.target_edge_type is not None and (
            not isinstance(self.target_edge_type, tuple)
            or len(self.target_edge_type) != EDGE_TYPE_TUPLE_LENGTH
            or not all(isinstance(s, str) for s in self.target_edge_type)
        ):
            raise ValueError(
                f"target_edge_type must be a 3-tuple of strings, "
                f"got {self.target_edge_type!r}"
            )


@dataclass(frozen=True, slots=True)
class Triple:
    """A single target triple sampled from the knowledge graph.

    Parameters
    ----------
    head_id : int
        Source entity index.
    tail_id : int
        Destination entity index.
    head_type : str
        Node type of the source entity.
    tail_type : str
        Node type of the destination entity.
    relation : str
        The relation connecting head to tail.
    """

    head_id: int
    tail_id: int
    head_type: str
    tail_type: str
    relation: str


class BaseTripleSampler:
    """Base class containing shared graph preparation and utilities."""

    def __init__(self, graph: HeteroGraph, config: SamplerConfig) -> None:
        self._graph = graph
        self._config = config
        self._generator = torch.Generator().manual_seed(config.seed)

        self._edge_types = [et for et in graph.edge_types if graph.edge_count(et) > 0]

        if config.target_edge_type is not None:
            target = config.target_edge_type
            if target not in self._edge_types:
                raise ValueError(
                    f"Target edge type {target!r} not found in graph or has zero edges"
                )
            self._edge_types = [target]

        self._edge_counts = torch.tensor(
            [graph.edge_count(et) for et in self._edge_types],
            dtype=torch.long,
        )

        if len(self._edge_counts) == 0:
            raise ValueError("No eligible edge types available for sampling.")

        self._cumulative_counts = torch.cumsum(self._edge_counts, dim=0)
        self._total_edges = int(self._cumulative_counts[-1].item())

    # ---- shared helpers ----

    def _extract_triple(
        self,
        edge_type: EdgeTypeTuple,
        local_idx: int,
    ) -> Triple:
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
        return min(self._config.sample_size, self._total_edges)

    def sample(self) -> list[Triple]:
        """Sample triples from the graph. Implemented by subclasses."""
        raise NotImplementedError
