"""Numba random-walk kernel over flat integer graph arrays.

From a start node the kernel takes up to ``max_len`` steps, at each step
choosing a candidate edge type from the step tables and then a neighbour
uniformly, and succeeds when it reaches the target no earlier than
``min_len`` steps. Results are written into caller-provided buffers; see
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
    step_edges: np.ndarray,
    step_offsets: np.ndarray,
    step_cumulative: np.ndarray,
    num_node_types: int,
    max_tracked_steps: int,
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
    """Run ``max_attempts`` random walks from one start node.

    Parameters
    ----------
    crow_all, col_all, crow_offsets, col_offsets, edge_dst_type : np.ndarray
        Flat integer graph arrays from
        :func:`~anyburl.walk._numba_graph.build_numba_graph_view`.
    step_edges, step_offsets, step_cumulative : np.ndarray
        Candidate edge types per (target type, node type, steps remaining)
        from :func:`~anyburl.walk._reachability.build_step_tables`. An
        unpruned table reproduces uniform selection exactly; a pruned one
        drops edge types that cannot reach ``tail_type`` in time.
        ``step_cumulative`` holds each block's running selection weight,
        ending at 1.0, so a uniform block is evenly spaced.
    num_node_types, max_tracked_steps : int
        Shape constants for decoding a block index in ``step_offsets``.
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
            steps_left = min(max_len - step, max_tracked_steps)
            block = (tail_type * num_node_types + current_type) * (
                max_tracked_steps + 1
            ) + steps_left
            out_start = step_offsets[block]
            num_out = step_offsets[block + 1] - out_start
            if num_out == 0:
                break
            draw = np.random.random()
            choice = out_start
            while choice < out_start + num_out - 1:
                if step_cumulative[choice] >= draw:
                    break
                choice += 1
            edge = step_edges[choice]

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
