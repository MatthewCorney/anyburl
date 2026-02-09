"""Triple sampling from a heterogeneous knowledge graph."""
import torch
from .base import BaseTripleSampler, Triple
from .._logging import get_logger
logger = get_logger(__name__)


class UniformTripleSampler(BaseTripleSampler):
    """Uniform sampling across all edges."""

    def sample(self) -> list[Triple]:
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
                int(self._cumulative_counts[type_idx - 1].item())
                if type_idx > 0
                else 0
            )
            local_idx = flat_idx - offset
            triples.append(
                self._extract_triple(self._edge_types[type_idx], local_idx)
            )

        logger.debug("Sampled %d triples via uniform strategy", len(triples))
        return triples
