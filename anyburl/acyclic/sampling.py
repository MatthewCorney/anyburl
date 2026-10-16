"""One-step walks from either end of a triple, for acyclic rules."""

import numpy as np
from numpy.typing import NDArray

from ..core import EdgeTypeTuple, Triple
from ..graph import HeteroGraph, mirrored_edge_type
from .generalization import AcyclicStep, WalkAnchor

__all__ = ["AcyclicStepSampler"]


class AcyclicStepSampler:
    """Samples uniformly random edges leaving the endpoints of triples.

    Every edge leaving an endpoint is equally likely, whatever its type.
    The triple's own edge, and its name-mirrored inverse, are never taken.

    Parameters
    ----------
    graph : HeteroGraph
        The graph to walk.
    seed : int
        Random seed.
    """

    def __init__(self, graph: HeteroGraph, seed: int) -> None:
        self._graph = graph
        self._rng = np.random.default_rng(seed)
        self._row_ptrs: dict[EdgeTypeTuple, NDArray[np.int64]] = {}
        self._targets: dict[EdgeTypeTuple, NDArray[np.int64]] = {}

    def sample(self, triple: Triple, samples_per_end: int) -> list[AcyclicStep]:
        """Return distinct steps sampled from both endpoints of ``triple``.

        Parameters
        ----------
        triple : Triple
            The target triple.
        samples_per_end : int
            Draws from each endpoint, before deduplication.

        Returns
        -------
        list[AcyclicStep]
            Distinct steps, in draw order.
        """
        target = (triple.head_type, triple.relation, triple.tail_type)
        steps: dict[AcyclicStep, None] = {}
        for anchor in WalkAnchor:
            for edge_type, neighbor in self._draw(triple, anchor, samples_per_end):
                if _is_own_edge(triple, anchor, edge_type, neighbor):
                    continue
                step = AcyclicStep(
                    target_edge_type=target,
                    anchor=anchor,
                    constant_id=triple.tail_id
                    if anchor is WalkAnchor.HEAD
                    else triple.head_id,
                    body_edge_type=edge_type,
                    neighbor_id=neighbor,
                )
                steps.setdefault(step, None)
        return list(steps)

    def _draw(
        self, triple: Triple, anchor: WalkAnchor, count: int
    ) -> list[tuple[EdgeTypeTuple, int]]:
        """Draw ``count`` uniform edges leaving the anchored endpoint."""
        if anchor is WalkAnchor.HEAD:
            entity, node_type = triple.head_id, triple.head_type
        else:
            entity, node_type = triple.tail_id, triple.tail_type
        edge_types = self._graph.outgoing_edge_types(node_type)
        starts = np.array([self._row_ptr(et)[entity] for et in edge_types])
        degrees = (
            np.array([self._row_ptr(et)[entity + 1] for et in edge_types]) - starts
        )
        total = int(degrees.sum())
        if total == 0:
            return []

        offsets = np.concatenate(([0], np.cumsum(degrees)))
        draws = self._rng.integers(0, total, size=count)
        type_positions = np.searchsorted(offsets, draws, side="right") - 1
        return [
            (
                edge_types[position],
                int(
                    self._targets_of(edge_types[position])[
                        starts[position] + draw - offsets[position]
                    ]
                ),
            )
            for draw, position in zip(
                draws.tolist(), type_positions.tolist(), strict=True
            )
        ]

    def _row_ptr(self, edge_type: EdgeTypeTuple) -> NDArray[np.int64]:
        """Return the cached CSR row pointer of ``edge_type``."""
        if edge_type not in self._row_ptrs:
            index = self._graph.csr_index(edge_type)
            self._row_ptrs[edge_type] = index.crow_indices.numpy().astype(np.int64)
            self._targets[edge_type] = index.col_indices.numpy().astype(np.int64)
        return self._row_ptrs[edge_type]

    def _targets_of(self, edge_type: EdgeTypeTuple) -> NDArray[np.int64]:
        """Return the cached CSR column indices of ``edge_type``."""
        self._row_ptr(edge_type)
        return self._targets[edge_type]


def _is_own_edge(
    triple: Triple, anchor: WalkAnchor, edge_type: EdgeTypeTuple, neighbor: int
) -> bool:
    """Return whether a step retraces the triple's own edge."""
    target = (triple.head_type, triple.relation, triple.tail_type)
    if anchor is WalkAnchor.HEAD:
        return edge_type == target and neighbor == triple.tail_id
    return edge_type == mirrored_edge_type(target) and neighbor == triple.head_id
