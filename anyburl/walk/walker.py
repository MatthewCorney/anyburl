"""Random walk engine for discovering rule paths in a knowledge graph."""

from typing import assert_never

import torch

from .._logging import get_logger
from ..graph import EdgeTypeTuple, HeteroGraph
from ..rule import PathStep
from ..sampler.sampler import Triple
from .config import WalkConfig, WalkStrategy

logger = get_logger(__name__)

EMPTY_RELATION: str = ""
"""Sentinel relation for the final step in a walk path."""


class WalkEngine:
    """Performs random walks on a heterogeneous graph to discover rule paths.

    Given a target triple ``(head, relation, tail)``, the engine walks
    from head attempting to reach tail within the configured step bounds.
    Successful paths are returned as ``PathStep`` sequences compatible
    with :class:`~anyburl.rule.RuleGeneralizer`.

    Parameters
    ----------
    graph : HeteroGraph
        The graph to walk on.
    config : WalkConfig
        Walk configuration (length bounds, attempts, strategy, seed).
    """

    def __init__(self, graph: HeteroGraph, config: WalkConfig) -> None:
        self._graph = graph
        self._config = config
        self._generator = torch.Generator().manual_seed(config.seed)

        self._inverse_weights: dict[str, torch.Tensor] = {}
        if config.strategy is WalkStrategy.RELATION_WEIGHTED:
            self._inverse_weights = self._compute_inverse_weights()

    def walk_from_triple(self, triple: Triple) -> list[list[PathStep]]:
        """Run random walks from a target triple's head toward its tail.

        Parameters
        ----------
        triple : Triple
            The target triple defining start (head) and goal (tail).

        Returns
        -------
        list[list[PathStep]]
            Unique successful paths. Each path is a list of
            ``(entity_id, node_type, relation)`` steps where the
            last step has an empty relation string.
        """
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
        """Attempt a single random walk from head to tail.

        Parameters
        ----------
        triple : Triple
            The target triple.

        Returns
        -------
        list[PathStep] | None
            A valid path if tail is reached within bounds, else ``None``.
        """
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
                0, neighbors.numel(), (1,), generator=self._generator
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
        """Select an outgoing edge type from the given node type.

        Parameters
        ----------
        node_type : str
            Current node type.

        Returns
        -------
        EdgeTypeTuple | None
            Selected edge type, or ``None`` if no outgoing edges exist.
        """
        candidates = self._graph.outgoing_edge_types(node_type)
        if not candidates:
            return None

        match self._config.strategy:
            case WalkStrategy.UNIFORM:
                idx = int(
                    torch.randint(
                        0, len(candidates), (1,), generator=self._generator
                    ).item()
                )
                return candidates[idx]
            case WalkStrategy.RELATION_WEIGHTED:
                return self._select_weighted(node_type, candidates)
            case _ as unreachable:
                assert_never(unreachable)

    def _select_weighted(
        self,
        node_type: str,
        candidates: tuple[EdgeTypeTuple, ...],
    ) -> EdgeTypeTuple:
        """Select an edge type using inverse-frequency weights.

        Parameters
        ----------
        node_type : str
            Current node type.
        candidates : tuple[EdgeTypeTuple, ...]
            Available outgoing edge types.

        Returns
        -------
        EdgeTypeTuple
            The selected edge type.
        """
        weights = self._inverse_weights.get(node_type)
        if weights is None:
            idx = int(
                torch.randint(
                    0, len(candidates), (1,), generator=self._generator
                ).item()
            )
            return candidates[idx]
        idx = int(
            torch.multinomial(weights, 1, generator=self._generator).item()
        )
        return candidates[idx]

    def _compute_inverse_weights(self) -> dict[str, torch.Tensor]:
        """Precompute inverse-frequency weight tensors per node type.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from node_type to weight tensor (one element per
            outgoing edge type, suitable for ``torch.multinomial``).
        """
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
