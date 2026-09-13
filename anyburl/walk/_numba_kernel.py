"""Numba ``@njit`` random-walk kernel over flat integer graph arrays.

The kernel mirrors the semantics of
:meth:`~anyburl.walk.walker.WalkEngine._single_walk`: from a start node it
takes up to ``max_len`` steps, at each step choosing an outgoing edge type
uniformly and then a neighbor uniformly, succeeding when it reaches the
target tail no earlier than ``min_len`` steps.

Results are written into caller-provided integer buffers to avoid any
Python object churn inside the hot loop. See
:mod:`anyburl.walk._numba_graph` for the array layout.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ._numba_graph import NO_RELATION_ID

STEP_FIELDS: int = 3
"""Columns per recorded step: (node_id, node_type_id, edge_type_id)."""

WALK_FAILED: int = -1
"""``out_lengths`` sentinel for an attempt that never reached the tail."""


@njit(cache=True)
def run_walks(
    crow_all: np.ndarray,
    col_all: np.ndarray,
    crow_offsets: np.ndarray,
    col_offsets: np.ndarray,
    edge_dst_type: np.ndarray,
    node_out_edges: np.ndarray,
    node_out_offsets: np.ndarray,
    start_id: int,
    start_type: int,
    tail_id: int,
    tail_type: int,
    min_len: int,
    max_len: int,
    max_attempts: int,
    seed: int,
    out_buffer: np.ndarray,
    out_lengths: np.ndarray,
) -> None:
    """Run ``max_attempts`` uniform random walks from one start node.

    Parameters
    ----------
    crow_all, col_all, crow_offsets, col_offsets, edge_dst_type,
    node_out_edges, node_out_offsets : np.ndarray
        Flat integer graph arrays from
        :func:`~anyburl.walk._numba_graph.build_numba_graph_view`.
    start_id, start_type : int
        Local node id and node-type id of the walk origin.
    tail_id, tail_type : int
        Local node id and node-type id of the target tail.
    min_len, max_len : int
        Inclusive minimum and maximum number of steps.
    max_attempts : int
        Number of independent walk attempts.
    seed : int
        Seed for this batch's random state.
    out_buffer : np.ndarray
        Int buffer of shape ``(max_attempts, max_len + 1, 3)`` to fill
        with ``(node_id, node_type_id, edge_type_id)`` per step.
    out_lengths : np.ndarray
        Int buffer of shape ``(max_attempts,)``; set to the number of
        recorded steps for a successful attempt or ``-1`` otherwise.
    """
    np.random.seed(seed)
    for attempt in range(max_attempts):
        out_lengths[attempt] = WALK_FAILED
        current_id = start_id
        current_type = start_type

        for step in range(max_len):
            out_start = node_out_offsets[current_type]
            num_out = node_out_offsets[current_type + 1] - out_start
            if num_out == 0:
                break
            edge = node_out_edges[out_start + np.random.randint(0, num_out)]

            crow_base = crow_offsets[edge]
            row_start = crow_all[crow_base + current_id]
            num_neighbors = crow_all[crow_base + current_id + 1] - row_start
            if num_neighbors == 0:
                break
            next_id = col_all[
                col_offsets[edge] + row_start + np.random.randint(0, num_neighbors)
            ]

            out_buffer[attempt, step, 0] = current_id
            out_buffer[attempt, step, 1] = current_type
            out_buffer[attempt, step, 2] = edge

            current_id = next_id
            current_type = edge_dst_type[edge]
            num_steps = step + 1

            if (
                current_id == tail_id
                and current_type == tail_type
                and num_steps >= min_len
            ):
                out_buffer[attempt, num_steps, 0] = current_id
                out_buffer[attempt, num_steps, 1] = current_type
                out_buffer[attempt, num_steps, 2] = NO_RELATION_ID
                out_lengths[attempt] = num_steps + 1
                break
