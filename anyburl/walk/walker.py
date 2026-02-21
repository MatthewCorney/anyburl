"""Random walk engine with pluggable edge-selection strategies."""

import torch

from .._logging import get_logger
from ..graph import EdgeTypeTuple, HeteroGraph
from ..rule import PathStep
from ..sampler import Triple
from .base import WalkConfig
from .relation_weighted_selector import RelationWeightedEdgeSelector
from .uniform_selector import UniformEdgeSelector

logger = get_logger(__name__)

DEFAULT_MAX_WALK_LENGTH: int = 5
DEFAULT_MIN_WALK_LENGTH: int = 2
DEFAULT_MAX_WALK_ATTEMPTS: int = 100
DEFAULT_RANDOM_SEED: int = 42

EMPTY_RELATION: str = ""
"""Sentinel relation for the final step in a walk path."""


class WalkEngine:
    """Performs random walks using a pluggable edge-selection strategy."""

    def __init__(
        self,
        graph: HeteroGraph,
        config: WalkConfig,
        selector: UniformEdgeSelector | RelationWeightedEdgeSelector,
    ) -> None:
        self._graph = graph
        self._config = config
        self._selector = selector
        self._generator = torch.Generator().manual_seed(config.seed)

    def walk_from_triple(self, triple: Triple) -> list[list[PathStep]]:
        """Run random walks from a target triple's head toward its tail."""
        seen: set[tuple[PathStep, ...]] = set()
        paths: list[list[PathStep]] = []

        for _ in range(self._config.max_attempts):
            path = self._single_walk(triple)
            if path is None:
                continue
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)

        logger.debug(
            "Found %d unique paths for triple (%s:%d -> %s:%d)",
            len(paths),
            triple.head_type,
            triple.head_id,
            triple.tail_type,
            triple.tail_id,
        )

        return paths

    def _single_walk(self, triple: Triple) -> list[PathStep] | None:
        current_id = triple.head_id
        current_type = triple.head_type
        steps: list[PathStep] = []

        for step_num in range(self._config.max_length):
            edge_type = self._select_edge_type(current_type)
            if edge_type is None:
                return None

            neighbors = self._graph.get_neighbors(current_id, edge_type)
            if neighbors.numel() == 0:
                return None

            rand_val = torch.randint(
                0,
                neighbors.numel(),
                (1,),
                generator=self._generator,
            )
            rand_idx = int(rand_val.item())
            next_id = int(neighbors[rand_idx].item())
            _, relation, next_type = edge_type

            steps.append((current_id, current_type, relation))
            current_id = next_id
            current_type = next_type

            num_steps = step_num + 1
            if (
                current_id == triple.tail_id
                and current_type == triple.tail_type
                and num_steps >= self._config.min_length
            ):
                steps.append((current_id, current_type, EMPTY_RELATION))
                return steps

        return None

    def _select_edge_type(self, node_type: str) -> EdgeTypeTuple | None:
        candidates = self._graph.outgoing_edge_types(node_type)
        if not candidates:
            return None
        return self._selector.select(node_type, candidates)
