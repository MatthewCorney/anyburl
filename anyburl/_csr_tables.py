"""Flat integer CSR tables for the whole graph, shared by Numba kernels.

Numba kernels cannot index a Python list of torch tensors, so a rule body has
to reach them as plain arrays. Flattening *per rule* would rebuild those arrays
for every rule and store a matrix twice whenever a chain repeats an edge type
--- BIOKG's worst chain uses ``interacts_with`` at two levels. Flattening the
*graph* once avoids both: a body chain then reduces to an array of edge-type
ids, so per-rule setup is a three-element array.

The layout mirrors :mod:`anyburl.walk._numba_graph`: every edge type's CSR
block is concatenated into one array, with an offset table giving each block's
start.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.utils import to_torch_csr_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .graph import EdgeTypeTuple, HeteroGraph

__all__ = ["CsrDirection", "CsrTables", "build_csr_tables"]


class CsrDirection(StrEnum):
    """Which way the tables traverse each edge type.

    Attributes
    ----------
    FORWARD : str
        Edges as stored, keyed by their own edge type.
    REVERSE : str
        Every edge type transposed and keyed by its swapped tuple
        ``(dst_type, relation, src_type)``. Needed to ground an
        object-grounded AC1 rule, which propagates from its constant tail
        back towards candidate heads.
    """

    FORWARD = "forward"
    REVERSE = "reverse"


@dataclass(frozen=True, slots=True)
class CsrTables:
    """Concatenated CSR structure of every non-empty edge type.

    Row ``r`` of edge type ``e`` occupies
    ``col_all[col_offsets[e] + crow_all[crow_offsets[e] + r] :
    col_offsets[e] + crow_all[crow_offsets[e] + r + 1]]``.

    Parameters
    ----------
    crow_all : NDArray[np.int64]
        Concatenated CSR row-offset arrays.
    col_all : NDArray[np.int64]
        Concatenated CSR column-index arrays.
    crow_offsets : NDArray[np.int64]
        Start of each edge type's block within ``crow_all``, length
        ``num_edge_types + 1``.
    col_offsets : NDArray[np.int64]
        Start of each edge type's block within ``col_all``, same length.
    num_src_nodes : NDArray[np.int64]
        Source node count per edge type.
    num_dst_nodes : NDArray[np.int64]
        Destination node count per edge type.
    edge_type_to_id : dict[EdgeTypeTuple, int]
        Position of each edge type in the tables.
    """

    crow_all: NDArray[np.int64]
    col_all: NDArray[np.int64]
    crow_offsets: NDArray[np.int64]
    col_offsets: NDArray[np.int64]
    num_src_nodes: NDArray[np.int64]
    num_dst_nodes: NDArray[np.int64]
    edge_type_to_id: dict[EdgeTypeTuple, int]

    def edge_ids(self, signature: Sequence[EdgeTypeTuple]) -> NDArray[np.int64]:
        """Return the table ids of a body chain's edge types, in order.

        Parameters
        ----------
        signature : Sequence[EdgeTypeTuple]
            Body edge types in chain order.

        Returns
        -------
        NDArray[np.int64]
            One id per edge type.

        Raises
        ------
        ValueError
            If an edge type is absent from the tables, which happens when it
            has no edges.
        """
        missing = [et for et in signature if et not in self.edge_type_to_id]
        if missing:
            raise ValueError(f"edge types absent from the graph tables: {missing!r}")
        return np.array([self.edge_type_to_id[et] for et in signature], dtype=np.int64)


def build_csr_tables(
    graph: HeteroGraph,
    *,
    direction: CsrDirection = CsrDirection.FORWARD,
) -> CsrTables:
    """Lower a graph's CSR indices into concatenated integer arrays.

    Edge types with no edges are skipped, matching
    :func:`~anyburl.walk._numba_graph.build_numba_graph_view`.

    Parameters
    ----------
    graph : HeteroGraph
        The source graph.
    direction : CsrDirection
        Whether to store each edge type as-is or transposed.

    Returns
    -------
    CsrTables
        Tables ready to index from a kernel.
    """
    stored = [et for et in graph.edge_types if graph.edge_count(et) > 0]

    crow_blocks: list[NDArray[np.int64]] = []
    col_blocks: list[NDArray[np.int64]] = []
    num_src: list[int] = []
    num_dst: list[int] = []
    edge_types: list[EdgeTypeTuple] = []
    for edge_type in stored:
        if direction is CsrDirection.FORWARD:
            index = graph._get_csr_index(edge_type)
            crow, col = _as_int64(index.crow_indices), _as_int64(index.col_indices)
            rows, cols = index.num_src, index.num_dst
            edge_types.append(edge_type)
        else:
            src_type, relation, dst_type = edge_type
            rows = graph.node_count(dst_type)
            cols = graph.node_count(src_type)
            crow, col = _reverse_csr(graph.edge_index(edge_type), rows, cols)
            edge_types.append((dst_type, relation, src_type))
        crow_blocks.append(crow)
        col_blocks.append(col)
        num_src.append(rows)
        num_dst.append(cols)

    return CsrTables(
        crow_all=_concat(crow_blocks),
        col_all=_concat(col_blocks),
        crow_offsets=_block_offsets(crow_blocks),
        col_offsets=_block_offsets(col_blocks),
        num_src_nodes=np.array(num_src, dtype=np.int64),
        num_dst_nodes=np.array(num_dst, dtype=np.int64),
        edge_type_to_id={et: i for i, et in enumerate(edge_types)},
    )


def _reverse_csr(
    edge_index: torch.Tensor,
    num_rows: int,
    num_cols: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the CSR structure of an edge type with its endpoints swapped."""
    flipped = torch.stack([edge_index[1], edge_index[0]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
        matrix = to_torch_csr_tensor(flipped, size=(num_rows, num_cols))
    return _as_int64(matrix.crow_indices()), _as_int64(matrix.col_indices())


def _as_int64(tensor: object) -> NDArray[np.int64]:
    """Return a tensor's buffer as a contiguous int64 array without copying.

    ``to_torch_csr_tensor`` already produces int64 indices on CPU, so this is
    a view. ``astype`` would copy roughly 18 MB per BIOKG graph for nothing.
    """
    array: NDArray[np.int64] = np.ascontiguousarray(tensor.numpy())  # type: ignore[attr-defined]
    return array


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
    concatenated: NDArray[np.int64] = np.concatenate(blocks)
    return concatenated
