"""Driver for the chain-counting kernel: validation, scratch and stamps.

Mirrors the role :mod:`anyburl.walk.numba_engine` plays for the walk kernel ---
it owns the flat tables, allocates the scratch buffers, keeps the stamp
counter monotone across calls, and hands back plain Python results.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from ._chain_kernel import FRONTIER_SLOTS, scan_chain_rows
from ._csr_tables import build_csr_tables

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .graph import EdgeTypeTuple, HeteroGraph

__all__ = ["ChainScanner"]


class ChainScanner:
    """Counts body-chain groundings for a graph without materialising products.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to ground against.
    """

    def __init__(self, graph: HeteroGraph) -> None:
        self._graph = graph
        self._tables = build_csr_tables(graph)
        self._stamp_base = 0

    def scan_all_rows(
        self,
        signature: Sequence[EdgeTypeTuple],
        head_edge_type: EdgeTypeTuple,
    ) -> tuple[int, int]:
        """Return total predictions and support over every source entity.

        Parameters
        ----------
        signature : Sequence[EdgeTypeTuple]
            Body edge types in chain order.
        head_edge_type : EdgeTypeTuple
            The relation the rule predicts.

        Returns
        -------
        tuple[int, int]
            ``(num_predictions, support)`` summed over all rows.
        """
        num_rows = self._graph.node_count(head_edge_type[0])
        rows = np.arange(num_rows, dtype=np.int64)
        predictions, support = self.scan_rows(signature, head_edge_type, rows)
        return int(predictions.sum()), int(support.sum())

    def scan_rows(
        self,
        signature: Sequence[EdgeTypeTuple],
        head_edge_type: EdgeTypeTuple,
        rows: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return per-row predictions and support for the given source rows.

        Parameters
        ----------
        signature : Sequence[EdgeTypeTuple]
            Body edge types in chain order.
        head_edge_type : EdgeTypeTuple
            The relation the rule predicts.
        rows : NDArray[np.int64]
            Source entity ids to scan.

        Returns
        -------
        tuple[NDArray[np.int64], NDArray[np.int64]]
            Distinct entities reached per row, and how many of those the head
            relation already connects.
        """
        self._validate(signature, head_edge_type)
        edge_ids = self._tables.edge_ids(signature)
        head_crow, head_col = self._head_arrays(head_edge_type)

        predictions = np.zeros(rows.shape[0], dtype=np.int64)
        support = np.zeros(rows.shape[0], dtype=np.int64)
        if rows.shape[0] == 0:
            return predictions, support

        width = self._scratch_width(edge_ids, head_edge_type)
        scan_chain_rows(
            self._tables.crow_all,
            self._tables.col_all,
            self._tables.crow_offsets,
            self._tables.col_offsets,
            edge_ids,
            head_crow,
            head_col,
            np.ascontiguousarray(rows, dtype=np.int64),
            self._stamp_base,
            np.zeros(width, dtype=np.int64),
            np.zeros(self._graph.node_count(head_edge_type[2]), dtype=np.int64),
            np.zeros((FRONTIER_SLOTS, width), dtype=np.int64),
            predictions,
            support,
        )
        self._stamp_base += rows.shape[0] * (len(signature) + 1) + 1
        return predictions, support

    def _scratch_width(
        self,
        edge_ids: NDArray[np.int64],
        head_edge_type: EdgeTypeTuple,
    ) -> int:
        """Return the widest node count any frontier or marker must hold."""
        widths = [int(self._tables.num_dst_nodes[edge]) for edge in edge_ids]
        widths.append(self._graph.node_count(head_edge_type[0]))
        return max(widths)

    def _head_arrays(
        self, head_edge_type: EdgeTypeTuple
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return the head relation's CSR arrays, padded when it has no edges.

        A head type present in the graph but carrying no edges has no CSR
        index, yet the kernel still needs a row-offset array to read.
        """
        num_rows = self._graph.node_count(head_edge_type[0])
        if self._graph.edge_count(head_edge_type) == 0:
            return (
                np.zeros(num_rows + 1, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )
        head_id = self._tables.edge_type_to_id[head_edge_type]
        crow = self._tables.crow_all[
            self._tables.crow_offsets[head_id] : self._tables.crow_offsets[head_id + 1]
        ]
        col = self._tables.col_all[
            self._tables.col_offsets[head_id] : self._tables.col_offsets[head_id + 1]
        ]
        return crow, col

    @staticmethod
    def _validate(
        signature: Sequence[EdgeTypeTuple],
        head_edge_type: EdgeTypeTuple,
    ) -> None:
        """Reject chains that cannot be walked or compared against the head.

        Raises
        ------
        ValueError
            If the chain is empty, does not join end to end, or its endpoints
            disagree with the head relation's.
        """
        if not signature:
            raise ValueError("body chain must contain at least one edge type")

        for earlier, later in pairwise(signature):
            if earlier[2] != later[0]:
                raise ValueError(
                    f"body chain does not join: {earlier!r} ends in "
                    f"{earlier[2]!r} but {later!r} starts at {later[0]!r}"
                )

        if signature[0][0] != head_edge_type[0]:
            raise ValueError(
                f"chain starts at {signature[0][0]!r} but head "
                f"{head_edge_type!r} starts at {head_edge_type[0]!r}"
            )
        if signature[-1][2] != head_edge_type[2]:
            raise ValueError(
                f"chain ends at {signature[-1][2]!r} but head "
                f"{head_edge_type!r} ends at {head_edge_type[2]!r}"
            )
