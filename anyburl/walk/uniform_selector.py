import torch

from ..graph import EdgeTypeTuple


class UniformEdgeSelector:
    """Select outgoing edge types uniformly."""

    def __init__(self, generator: torch.Generator) -> None:
        self._generator = generator

    def select(
        self,
        node_type: str,
        candidates: tuple[EdgeTypeTuple, ...],
    ) -> EdgeTypeTuple:
        idx = int(
            torch.randint(
                0,
                len(candidates),
                (1,),
                generator=self._generator,
            ).item()
        )
        return candidates[idx]

