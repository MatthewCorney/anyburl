import torch

from ..graph import EdgeTypeTuple, HeteroGraph


class RelationWeightedEdgeSelector:
    """Select edge types using inverse-frequency weighting."""

    def __init__(
            self,
            graph: HeteroGraph,
            generator: torch.Generator,
    ) -> None:
        self._graph = graph
        self._generator = generator
        self._weights = self._compute_inverse_weights()

    def select(
            self,
            node_type: str,
            candidates: tuple[EdgeTypeTuple, ...],
    ) -> EdgeTypeTuple:
        weights = self._weights.get(node_type)

        if weights is None:
            # Fallback to uniform
            idx = int(
                torch.randint(
                    0,
                    len(candidates),
                    (1,),
                    generator=self._generator,
                ).item()
            )
            return candidates[idx]

        idx = int(
            torch.multinomial(
                weights,
                1,
                generator=self._generator,
            ).item()
        )
        return candidates[idx]

    def _compute_inverse_weights(self) -> dict[str, torch.Tensor]:
        """Precompute inverse-frequency weight tensors per node type."""
        result: dict[str, torch.Tensor] = {}

        for node_type in self._graph.node_types:
            edge_types = self._graph.outgoing_edge_types(node_type)
            if not edge_types:
                continue

            raw = torch.tensor(
                [
                    1.0 / self._graph.edge_count(et)
                    if self._graph.edge_count(et) > 0
                    else 1.0
                    for et in edge_types
                ],
                dtype=torch.float32,
            )

            result[node_type] = raw

        return result
