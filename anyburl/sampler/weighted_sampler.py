import torch
from torch import Tensor

from .._logging import get_logger
from ..graph import HeteroGraph
from .base import BaseTripleSampler, SamplerConfig, Triple

logger = get_logger(__name__)


class WeightedTripleSampler(BaseTripleSampler):
    """Weighted sampling over edge types, uniform within each type."""

    def __init__(
        self,
        graph: HeteroGraph,
        config: SamplerConfig,
        weights: Tensor,
    ) -> None:
        super().__init__(graph, config)

        if weights.shape[0] != len(self._edge_types):
            raise ValueError("Weights must match number of edge types.")

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
