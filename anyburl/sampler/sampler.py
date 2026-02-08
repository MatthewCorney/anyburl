"""Triple sampling from a heterogeneous knowledge graph."""

from dataclasses import dataclass
from typing import assert_never

import torch
from torch import Tensor

from .._logging import get_logger
from ..graph import EdgeTypeTuple, HeteroGraph
from .config import SamplerConfig, SamplingStrategy

logger = get_logger(__name__)


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


class TripleSampler:
    """Samples target triples from a heterogeneous knowledge graph.

    Uses bulk tensor operations for efficient sampling across edge types.

    Parameters
    ----------
    graph : HeteroGraph
        The graph to sample from.
    config : SamplerConfig
        Sampling configuration (size, strategy, seed).
    """

    def __init__(self, graph: HeteroGraph, config: SamplerConfig) -> None:
        self._graph = graph
        self._config = config
        self._generator = torch.Generator().manual_seed(config.seed)

        self._edge_types = [et for et in graph.edge_types if graph.edge_count(et) > 0]
        self._edge_counts = torch.tensor(
            [graph.edge_count(et) for et in self._edge_types], dtype=torch.long
        )
        self._cumulative_counts = torch.cumsum(self._edge_counts, dim=0)
        self._total_edges = int(self._cumulative_counts[-1].item())

        self._equal_weights = torch.ones(len(self._edge_types), dtype=torch.float32)
        self._inverse_weights = self._compute_inverse_weights()

    def sample(self) -> list[Triple]:
        """Sample triples according to the configured strategy.

        Returns
        -------
        list[Triple]
            Sampled triples. Length equals ``config.sample_size``,
            capped at the total number of edges.
        """
        match self._config.strategy:
            case SamplingStrategy.UNIFORM:
                triples = self._sample_uniform()
            case SamplingStrategy.RELATION_PROPORTIONAL:
                triples = self._sample_weighted(self._equal_weights)
            case SamplingStrategy.RELATION_INVERSE:
                triples = self._sample_weighted(self._inverse_weights)
            case _ as unreachable:
                assert_never(unreachable)
        logger.debug(
            "Sampled %d triples via %s strategy",
            len(triples),
            self._config.strategy.value,
        )
        return triples

    def _sample_uniform(self) -> list[Triple]:
        """Sample uniformly across all edges via flat random indexing.

        Each edge has equal probability of being selected, so more
        common relations naturally dominate the sample.

        Returns
        -------
        list[Triple]
            Sampled triples.
        """
        n = min(self._config.sample_size, self._total_edges)
        flat_indices = torch.randint(
            0, self._total_edges, (n,), generator=self._generator
        )
        type_indices = torch.searchsorted(
            self._cumulative_counts, flat_indices, right=True
        )

        triples: list[Triple] = []
        for type_idx_val, flat_idx_val in zip(
            type_indices.tolist(), flat_indices.tolist(), strict=True
        ):
            et = self._edge_types[type_idx_val]
            offset = (
                int(self._cumulative_counts[type_idx_val - 1].item())
                if type_idx_val > 0
                else 0
            )
            local_idx = flat_idx_val - offset
            triples.append(self._extract_triple(et, local_idx))
        return triples

    def _sample_weighted(self, weights: Tensor) -> list[Triple]:
        """Sample by selecting edge types with given weights, then uniformly within.

        Groups selections by edge type and generates all random
        indices per type in a single ``torch.randint`` call to
        avoid per-sample Python loop overhead.

        Parameters
        ----------
        weights : Tensor
            Selection weights per edge type (need not sum to 1;
            ``torch.multinomial`` normalizes internally).

        Returns
        -------
        list[Triple]
            Sampled triples.
        """
        n = min(self._config.sample_size, self._total_edges)
        num_types = len(self._edge_types)

        type_selections = torch.multinomial(
            weights,
            n,
            replacement=True,
            generator=self._generator,
        )

        type_counts = torch.bincount(type_selections, minlength=num_types)

        # Batch-generate random edge indices per type
        local_indices_by_type: list[Tensor] = []
        for type_idx in range(num_types):
            count = int(type_counts[type_idx].item())
            if count == 0:
                local_indices_by_type.append(torch.empty(0, dtype=torch.long))
                continue
            edge_count = int(self._edge_counts[type_idx].item())
            local_indices_by_type.append(
                torch.randint(
                    0, edge_count, (count,), generator=self._generator
                )
            )

        # Scatter back into original order
        consumed = [0] * num_types
        triples: list[Triple] = []
        for type_idx_val in type_selections.tolist():
            et = self._edge_types[type_idx_val]
            local_idx = int(
                local_indices_by_type[type_idx_val][consumed[type_idx_val]].item()
            )
            consumed[type_idx_val] += 1
            triples.append(self._extract_triple(et, local_idx))
        return triples

    def _extract_triple(self, edge_type: EdgeTypeTuple, local_idx: int) -> Triple:
        """Build a Triple from an edge type and local edge index.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The ``(src_type, relation, dst_type)`` edge type.
        local_idx : int
            Index into this edge type's edge_index tensor.

        Returns
        -------
        Triple
            The extracted triple.
        """
        ei: Tensor = self._graph.edge_index(edge_type)
        src_type, relation, dst_type = edge_type
        return Triple(
            head_id=int(ei[0, local_idx].item()),
            tail_id=int(ei[1, local_idx].item()),
            head_type=src_type,
            tail_type=dst_type,
            relation=relation,
        )

    def _compute_inverse_weights(self) -> Tensor:
        """Compute inverse-frequency weights for edge type selection.

        Returns
        -------
        Tensor
            Normalized weights (sums to 1).
        """
        inverse = 1.0 / self._edge_counts.float()
        return inverse / inverse.sum()
