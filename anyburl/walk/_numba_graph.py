"""Integer-encoded, flat-array view of a :class:`HeteroGraph` for Numba.

Numba's ``@njit`` kernels cannot operate on Python strings, dicts, or
heterogeneous tuples. This module lowers a :class:`HeteroGraph` into a
bundle of NumPy integer arrays that a JIT-compiled walk kernel can index
directly, plus the lookup tables needed to decode integer walk results
back into string-typed :class:`~anyburl.rule.PathStep` tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..graph import EdgeTypeTuple, HeteroGraph

NO_RELATION_ID: int = -1
"""Sentinel edge-type id marking the final step of a walk (empty relation)."""


@dataclass(frozen=True, slots=True)
class NumbaGraphView:
    """Flat integer arrays describing a heterogeneous graph for walk kernels.

    All node ids are local to their node type, matching
    :class:`~anyburl.graph.HeteroGraph` and
    :class:`~anyburl.sampler.Triple` conventions.

    Parameters
    ----------
    crow_all : NDArray[np.int64]
        Concatenated CSR row-offset arrays for every edge type.
    col_all : NDArray[np.int64]
        Concatenated CSR column-index arrays for every edge type.
    crow_offsets : NDArray[np.int64]
        Start index of each edge type's block within ``crow_all``.
    col_offsets : NDArray[np.int64]
        Start index of each edge type's block within ``col_all``.
    edge_dst_type : NDArray[np.int64]
        Destination node-type id for each edge type.
    node_out_edges : NDArray[np.int64]
        Flattened outgoing edge-type ids grouped by source node type.
    node_out_offsets : NDArray[np.int64]
        Start index of each node type's block within ``node_out_edges``.
    node_type_names : tuple[str, ...]
        Node-type name for each node-type id (decode table).
    edge_relation_names : tuple[str, ...]
        Relation name for each edge-type id (decode table).
    node_type_to_id : dict[str, int]
        Inverse of ``node_type_names``.
    """

    crow_all: NDArray[np.int64]
    col_all: NDArray[np.int64]
    crow_offsets: NDArray[np.int64]
    col_offsets: NDArray[np.int64]
    edge_dst_type: NDArray[np.int64]
    node_out_edges: NDArray[np.int64]
    node_out_offsets: NDArray[np.int64]
    node_type_names: tuple[str, ...]
    edge_relation_names: tuple[str, ...]
    node_type_to_id: dict[str, int]


def build_numba_graph_view(graph: HeteroGraph) -> NumbaGraphView:
    """Lower a :class:`HeteroGraph` into flat integer arrays for Numba.

    Parameters
    ----------
    graph : HeteroGraph
        The source graph. Only edge types with at least one edge are
        included.

    Returns
    -------
    NumbaGraphView
        Integer-encoded view ready for the walk kernel.
    """
    node_type_names = graph.node_types
    node_type_to_id = {name: i for i, name in enumerate(node_type_names)}

    edge_types = [et for et in graph.edge_types if _has_csr(graph, et)]
    edge_type_to_id = {et: i for i, et in enumerate(edge_types)}

    crow_blocks, col_blocks = _collect_csr_blocks(graph, edge_types)
    crow_offsets = _block_offsets(crow_blocks)
    col_offsets = _block_offsets(col_blocks)

    edge_dst_type = np.array(
        [node_type_to_id[et[2]] for et in edge_types], dtype=np.int64
    )
    edge_relation_names = tuple(et[1] for et in edge_types)

    node_out_edges, node_out_offsets = _build_outgoing_index(
        graph, node_type_names, edge_type_to_id
    )

    return NumbaGraphView(
        crow_all=_concat(crow_blocks),
        col_all=_concat(col_blocks),
        crow_offsets=crow_offsets,
        col_offsets=col_offsets,
        edge_dst_type=edge_dst_type,
        node_out_edges=node_out_edges,
        node_out_offsets=node_out_offsets,
        node_type_names=node_type_names,
        edge_relation_names=edge_relation_names,
        node_type_to_id=node_type_to_id,
    )


def _has_csr(graph: HeteroGraph, edge_type: EdgeTypeTuple) -> bool:
    """Return ``True`` if the edge type has a CSR index (non-empty)."""
    return edge_type in graph.edge_types and graph.edge_count(edge_type) > 0


def _collect_csr_blocks(
    graph: HeteroGraph,
    edge_types: list[EdgeTypeTuple],
) -> tuple[list[NDArray[np.int64]], list[NDArray[np.int64]]]:
    """Extract per-edge-type CSR row-offset and column-index arrays."""
    crow_blocks: list[NDArray[np.int64]] = []
    col_blocks: list[NDArray[np.int64]] = []
    for et in edge_types:
        idx = graph._get_csr_index(et)
        crow_blocks.append(idx.crow_indices.numpy().astype(np.int64))
        col_blocks.append(idx.col_indices.numpy().astype(np.int64))
    return crow_blocks, col_blocks


def _build_outgoing_index(
    graph: HeteroGraph,
    node_type_names: tuple[str, ...],
    edge_type_to_id: dict[EdgeTypeTuple, int],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Build the flattened per-node-type outgoing edge-type index."""
    flat: list[int] = []
    offsets: list[int] = [0]
    for node_type in node_type_names:
        for et in graph.outgoing_edge_types(node_type):
            flat.append(edge_type_to_id[et])
        offsets.append(len(flat))
    return (
        np.array(flat, dtype=np.int64),
        np.array(offsets, dtype=np.int64),
    )


def _block_offsets(blocks: list[NDArray[np.int64]]) -> NDArray[np.int64]:
    """Return the start offset of each block in a concatenation."""
    offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
    for i, block in enumerate(blocks):
        offsets[i + 1] = offsets[i] + block.shape[0]
    return offsets


def _concat(blocks: list[NDArray[np.int64]]) -> NDArray[np.int64]:
    """Concatenate blocks, returning an empty array when there are none."""
    if not blocks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(blocks)
