"""Driver for the chain-counting kernel.

Owns the flat CSR tables, validates chains, allocates scratch buffers, keeps
the stamp counter monotone across calls and returns plain Python results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..exceptions import InvalidRuleError
from ..graph import mirrored_edge_type, validate_chain
from ._kernel import FRONTIER_SLOTS, scan_chain_rows
from .tables import CsrDirection, CsrTables, build_csr_tables

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..graph import EdgeTypeTuple, HeteroGraph

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
        self._reverse_tables: CsrTables | None = None
        self._stamp_base = 0

    def scan_reversed_rows(
        self,
        signature: Sequence[EdgeTypeTuple],
        head_edge_type: EdgeTypeTuple,
        rows: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Scan a chain backwards, from its tail endpoint towards its head.

        Used for object-grounded AC1 rules, whose constant sits on the tail.
        The chain and the head relation are both transposed.

        Parameters
        ----------
        signature : Sequence[EdgeTypeTuple]
            Body edge types in *forward* chain order.
        head_edge_type : EdgeTypeTuple
            The relation the rule predicts, in its forward orientation.
        rows : NDArray[np.int64]
            Tail entity ids to scan from.

        Returns
        -------
        tuple[NDArray[np.int64], NDArray[np.int64]]
            Distinct heads reached per tail, and how many of those the head
            relation already connects.
        """
        return self._scan(
            self._reversed(signature),
            mirrored_edge_type(head_edge_type),
            rows,
            self._reverse(),
        )

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
        return self._scan(signature, head_edge_type, rows, self._tables)

    def reachable_sources(
        self, signature: Sequence[EdgeTypeTuple]
    ) -> NDArray[np.bool_]:
        """Return which source entities the chain connects to anything.

        Parameters
        ----------
        signature : Sequence[EdgeTypeTuple]
            Body edge types in chain order.

        Returns
        -------
        NDArray[np.bool_]
            One flag per entity of the chain's starting node type.
        """
        rows = np.arange(self._graph.node_count(signature[0][0]), dtype=np.int64)
        predictions, _ = self._scan(signature, None, rows, self._tables)
        return predictions > 0

    def reachable_targets(
        self, signature: Sequence[EdgeTypeTuple]
    ) -> NDArray[np.bool_]:
        """Return which target entities the chain connects from anything.

        Parameters
        ----------
        signature : Sequence[EdgeTypeTuple]
            Body edge types in forward chain order.

        Returns
        -------
        NDArray[np.bool_]
            One flag per entity of the chain's ending node type.
        """
        rows = np.arange(self._graph.node_count(signature[-1][2]), dtype=np.int64)
        predictions, _ = self._scan(
            self._reversed(signature), None, rows, self._reverse()
        )
        return predictions > 0

    def _reverse(self) -> CsrTables:
        """Return the reverse tables, building them on first use."""
        if self._reverse_tables is None:
            self._reverse_tables = build_csr_tables(
                self._graph, direction=CsrDirection.REVERSE
            )
        return self._reverse_tables

    @staticmethod
    def _reversed(
        signature: Sequence[EdgeTypeTuple],
    ) -> tuple[EdgeTypeTuple, ...]:
        """Return the chain walked backwards, each edge type swapped."""
        return tuple(mirrored_edge_type(et) for et in reversed(tuple(signature)))

    def _scan(
        self,
        signature: Sequence[EdgeTypeTuple],
        head_edge_type: EdgeTypeTuple | None,
        rows: NDArray[np.int64],
        tables: CsrTables,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Run the kernel over one direction's tables.

        ``head_edge_type`` may be ``None`` when only reachability is wanted,
        in which case the support counts come back zero.
        """
        self._validate(signature, head_edge_type)
        edge_ids = tables.edge_ids(signature)
        head_crow, head_col = self._head_arrays(head_edge_type, signature, tables)

        predictions = np.zeros(rows.shape[0], dtype=np.int64)
        support = np.zeros(rows.shape[0], dtype=np.int64)
        if rows.shape[0] == 0:
            return predictions, support

        width = self._scratch_width(edge_ids, signature[0][0], tables)
        scan_chain_rows(
            tables.crow_all,
            tables.col_all,
            tables.crow_offsets,
            tables.col_offsets,
            edge_ids,
            head_crow,
            head_col,
            np.ascontiguousarray(rows, dtype=np.int64),
            self._stamp_base,
            np.zeros(width, dtype=np.int64),
            np.zeros(
                self._head_marker_width(head_edge_type, signature), dtype=np.int64
            ),
            np.zeros((FRONTIER_SLOTS, width), dtype=np.int64),
            predictions,
            support,
        )
        self._stamp_base += rows.shape[0] * (len(signature) + 1) + 1
        return predictions, support

    def _scratch_width(
        self,
        edge_ids: NDArray[np.int64],
        source_type: str,
        tables: CsrTables,
    ) -> int:
        """Return the widest node count any frontier or marker must hold."""
        widths = [int(tables.num_dst_nodes[edge]) for edge in edge_ids]
        widths.append(self._graph.node_count(source_type))
        return max(widths)

    def _head_marker_width(
        self,
        head_edge_type: EdgeTypeTuple | None,
        signature: Sequence[EdgeTypeTuple],
    ) -> int:
        """Return the size of the head-marker scratch for this scan."""
        if head_edge_type is None:
            return self._graph.node_count(signature[-1][2])
        return self._graph.node_count(head_edge_type[2])

    def _head_arrays(
        self,
        head_edge_type: EdgeTypeTuple | None,
        signature: Sequence[EdgeTypeTuple],
        tables: CsrTables,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return the head relation's CSR arrays, padded when it has no edges.

        The kernel indexes the row-offset array by every scanned row and does
        no bounds checking, so when there is no head relation, or it has no
        edges, an all-zero array spanning every source row is returned.
        """
        if head_edge_type is None or head_edge_type not in tables.edge_type_to_id:
            source_rows = self._graph.node_count(signature[0][0])
            return (
                np.zeros(source_rows + 1, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )
        head_id = tables.edge_type_to_id[head_edge_type]
        crow = tables.crow_all[
            tables.crow_offsets[head_id] : tables.crow_offsets[head_id + 1]
        ]
        col = tables.col_all[
            tables.col_offsets[head_id] : tables.col_offsets[head_id + 1]
        ]
        return crow, col

    @staticmethod
    def _validate(
        signature: Sequence[EdgeTypeTuple],
        head_edge_type: EdgeTypeTuple | None,
    ) -> None:
        """Reject chains that cannot be walked or compared against the head.

        Raises
        ------
        InvalidRuleError
            If the chain is empty, does not join end to end, or its endpoints
            disagree with the head relation's.
        """
        validate_chain(signature)
        if head_edge_type is None:
            return
        if signature[0][0] != head_edge_type[0]:
            raise InvalidRuleError(
                f"chain starts at {signature[0][0]!r} but head "
                f"{head_edge_type!r} starts at {head_edge_type[0]!r}"
            )
        if signature[-1][2] != head_edge_type[2]:
            raise InvalidRuleError(
                f"chain ends at {signature[-1][2]!r} but head "
                f"{head_edge_type!r} ends at {head_edge_type[2]!r}"
            )
