"""A plain-Python random walker, the reference the Numba walk engine is checked against.

Not a fixture module: tests import :class:`ReferenceWalker` directly.
"""

import torch

from anyburl.core import EdgeTypeTuple, Triple
from anyburl.graph import HeteroGraph
from anyburl.rule import PathStep
from anyburl.walk import WalkConfig
from anyburl.walk.base import EMPTY_RELATION
from tests import adjacency


class ReferenceWalker:
    """Walks one step at a time, choosing edge types and neighbours uniformly.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to walk over.
    config : WalkConfig
        Walk lengths, attempts and seed.
    """

    def __init__(self, graph: HeteroGraph, config: WalkConfig) -> None:
        self._graph = graph
        self._config = config
        self._generator = torch.Generator().manual_seed(config.seed)
        self._edge_generator = torch.Generator().manual_seed(config.seed)

    def walk_from_triple(self, triple: Triple) -> list[list[PathStep]]:
        """Return the unique successful walks from the triple's head to its tail."""
        seen: set[tuple[PathStep, ...]] = set()
        paths: list[list[PathStep]] = []
        for _ in range(self._config.max_attempts):
            path = self._single_walk(triple)
            if path is not None and tuple(path) not in seen:
                seen.add(tuple(path))
                paths.append(path)
        return paths

    def _single_walk(self, triple: Triple) -> list[PathStep] | None:
        """Return one walk reaching the tail within the length bounds, if any."""
        current_id, current_type = triple.head_id, triple.head_type
        steps: list[PathStep] = []
        for step_num in range(self._config.max_length):
            edge_type = self._select_edge_type(current_type)
            if edge_type is None:
                return None
            neighbors = adjacency.neighbors(self._graph, current_id, edge_type)
            if neighbors.numel() == 0:
                return None
            index = torch.randint(0, neighbors.numel(), (1,), generator=self._generator)
            _, relation, next_type = edge_type
            steps.append((current_id, current_type, relation))
            current_id = int(neighbors[int(index.item())].item())
            current_type = next_type
            if (
                current_id == triple.tail_id
                and current_type == triple.tail_type
                and step_num + 1 >= self._config.min_length
            ):
                steps.append((current_id, current_type, EMPTY_RELATION))
                return steps
        return None

    def _select_edge_type(self, node_type: str) -> EdgeTypeTuple | None:
        """Return a uniformly chosen outgoing edge type of ``node_type``."""
        candidates = self._graph.outgoing_edge_types(node_type)
        if not candidates:
            return None
        index = torch.randint(0, len(candidates), (1,), generator=self._edge_generator)
        return candidates[int(index.item())]
