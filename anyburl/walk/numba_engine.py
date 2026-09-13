"""Numba-backed random walk engine.

Drop-in replacement for :class:`~anyburl.walk.walker.WalkEngine` that runs
the JIT-compiled kernel in :mod:`anyburl.walk._numba_kernel`. The public
API (:meth:`walk_from_triple`) is identical, so the pipeline, generalizer,
and tests are unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._logging import get_logger
from ..rule import PathStep
from ._numba_graph import NO_RELATION_ID, build_numba_graph_view
from ._numba_kernel import STEP_FIELDS, WALK_FAILED, run_walks
from .base import WalkConfig
from .walker import EMPTY_RELATION

if TYPE_CHECKING:
    from ..graph import HeteroGraph
    from ..sampler import Triple

logger = get_logger(__name__)


class NumbaWalkEngine:
    """Runs random walks via a JIT-compiled kernel over integer arrays.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to walk over.
    config : WalkConfig
        Walk configuration (lengths, attempts, seed).
    """

    def __init__(self, graph: HeteroGraph, config: WalkConfig) -> None:
        self._config = config
        self._view = build_numba_graph_view(graph)
        self._call_index = 0

        buffer_shape = (config.max_attempts, config.max_length + 1, STEP_FIELDS)
        self._out_buffer = np.empty(buffer_shape, dtype=np.int64)
        self._out_lengths = np.empty(config.max_attempts, dtype=np.int64)

    def walk_from_triple(self, triple: Triple) -> list[list[PathStep]]:
        """Run random walks from a target triple's head toward its tail.

        Parameters
        ----------
        triple : Triple
            The target triple providing walk start and goal.

        Returns
        -------
        list[list[PathStep]]
            Unique successful walk paths.
        """
        view = self._view
        start_type = view.node_type_to_id[triple.head_type]
        tail_type = view.node_type_to_id[triple.tail_type]

        seed = self._next_seed()
        run_walks(
            view.crow_all,
            view.col_all,
            view.crow_offsets,
            view.col_offsets,
            view.edge_dst_type,
            view.node_out_edges,
            view.node_out_offsets,
            triple.head_id,
            start_type,
            triple.tail_id,
            tail_type,
            self._config.min_length,
            self._config.max_length,
            self._config.max_attempts,
            seed,
            self._out_buffer,
            self._out_lengths,
        )

        return self._decode_unique_paths()

    def _next_seed(self) -> int:
        """Return a deterministic per-call seed and advance the counter."""
        seed = self._config.seed + self._call_index
        self._call_index += 1
        return seed

    def _decode_unique_paths(self) -> list[list[PathStep]]:
        """Decode successful attempts in the output buffer, deduplicated."""
        seen: set[tuple[PathStep, ...]] = set()
        paths: list[list[PathStep]] = []

        for attempt in range(self._config.max_attempts):
            length = int(self._out_lengths[attempt])
            if length == WALK_FAILED:
                continue
            path = self._decode_path(attempt, length)
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)

        return paths

    def _decode_path(self, attempt: int, length: int) -> list[PathStep]:
        """Decode one attempt row of ``length`` steps into ``PathStep`` tuples."""
        view = self._view
        row = self._out_buffer[attempt]
        steps: list[PathStep] = []
        for i in range(length):
            node_id = int(row[i, 0])
            node_type = view.node_type_names[int(row[i, 1])]
            edge_id = int(row[i, 2])
            relation = (
                EMPTY_RELATION
                if edge_id == NO_RELATION_ID
                else view.edge_relation_names[edge_id]
            )
            steps.append((node_id, node_type, relation))
        return steps
