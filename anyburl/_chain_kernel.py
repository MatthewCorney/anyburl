"""Numba ``@njit`` kernel counting a body chain's groundings without building it.

Rule evaluation needs two integers per source row: how many distinct entities
the body chain reaches, and how many of those the head relation already
connects. The sparse-matmul route derives them by materialising the whole
product --- 93.2 million non-zeros for BIOKG's worst chain, costing 1.3-1.6 GB
of matmul workspace that the allocator never returns.

This kernel walks the chain a row at a time instead. Each level's reachable
set is deduplicated with a *timestamp* array: a node belongs to the current
level when its mark equals the current stamp, so nothing has to be cleared
between rows or levels. That reproduces sparse-matmul semantics exactly, since
``nnz`` counts distinct ``(row, col)`` pairs and ``to_torch_csr_tensor``
coalesces duplicate edges.

The stamp advances **per level**, not per row. This is load-bearing: on
``interacts_with -> interacts_with -> is_annotated_to`` the node type
``protein`` appears at levels 0, 1 and 2, so a single stamp per row would
suppress a node at level 2 merely because level 1 had already reached it.

Scratch is a few arrays sized by node count --- about 1.6 MB on BIOKG.
"""

from __future__ import annotations

import numpy as np
from numba import njit

SEEN_UNSTAMPED: int = 0
"""Initial marker value, distinct from every stamp the kernel issues."""

FRONTIER_SLOTS: int = 2
"""Ping-pong buffers: the current level's frontier and the next one's."""


@njit(cache=True)
def scan_chain_rows(
    crow_all: np.ndarray,
    col_all: np.ndarray,
    crow_offsets: np.ndarray,
    col_offsets: np.ndarray,
    chain_edges: np.ndarray,
    head_crow: np.ndarray,
    head_col: np.ndarray,
    source_rows: np.ndarray,
    stamp_base: int,
    marker: np.ndarray,
    head_marker: np.ndarray,
    frontier: np.ndarray,
    out_predictions: np.ndarray,
    out_support: np.ndarray,
) -> None:
    """Count reached entities and head-relation hits for each source row.

    Parameters
    ----------
    crow_all, col_all, crow_offsets, col_offsets : np.ndarray
        Concatenated graph CSR tables from
        :func:`~anyburl._csr_tables.build_csr_tables`.
    chain_edges : np.ndarray
        Edge-type ids of the body chain, in order.
    head_crow, head_col : np.ndarray
        CSR structure of the head relation, rows indexed by source entity.
    source_rows : np.ndarray
        Source entity ids to scan.
    stamp_base : int
        Monotone offset, advanced by the caller between calls that share
        scratch, so stale marks can never match a fresh stamp.
    marker, head_marker : np.ndarray
        Timestamp scratch, sized to the widest node type involved and to the
        head relation's destination count respectively.
    frontier : np.ndarray
        ``(2, width)`` ping-pong buffer indexed by level parity.
    out_predictions, out_support : np.ndarray
        One entry per source row, written by this call.
    """
    num_levels = chain_edges.shape[0]
    stamps_per_row = num_levels + 1

    for i in range(source_rows.shape[0]):
        row = source_rows[i]
        head_stamp = stamp_base + i * stamps_per_row + 1

        for slot in range(head_crow[row], head_crow[row + 1]):
            head_marker[head_col[slot]] = head_stamp

        current = 0
        frontier[current, 0] = row
        size = 1

        for level in range(num_levels):
            edge = chain_edges[level]
            crow_base = crow_offsets[edge]
            col_base = col_offsets[edge]
            level_stamp = head_stamp + level + 1
            nxt = 1 - current
            next_size = 0

            for position in range(size):
                node = frontier[current, position]
                for slot in range(
                    crow_all[crow_base + node], crow_all[crow_base + node + 1]
                ):
                    neighbour = col_all[col_base + slot]
                    if marker[neighbour] != level_stamp:
                        marker[neighbour] = level_stamp
                        frontier[nxt, next_size] = neighbour
                        next_size += 1

            current = nxt
            size = next_size
            if size == 0:
                break

        out_predictions[i] = size
        support = 0
        for position in range(size):
            if head_marker[frontier[current, position]] == head_stamp:
                support += 1
        out_support[i] = support
