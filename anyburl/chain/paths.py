"""Driver for the path-counting kernel: one row of a chain's product at a time."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..graph import mirrored_edge_type, validate_chain
from ._kernel import FRONTIER_SLOTS, count_chain_paths
from .tables import CsrTables, forward_csr_tables, reverse_csr_tables

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from ..core import EdgeTypeTuple
    from ..graph import HeteroGraph

__all__ = ["PathCounter"]


class PathCounter:
    """Counts the paths a body chain takes from one entity to every other.

    Equivalent to reading one row (or column) of the product of the chain's
    adjacency matrices, without building the product. One instance serves
    every chain on a graph; share it via ``graph.cached(PathCounter)``. Its
    scratch buffers make it unsafe to call from several threads at once.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph whose chains are counted.
    """

    def __init__(self, graph: HeteroGraph) -> None:
        self._graph = graph
        width = max(graph.node_count(node_type) for node_type in graph.node_types)
        self._counts = np.zeros((FRONTIER_SLOTS, width), dtype=np.int64)
        self._frontier = np.zeros((FRONTIER_SLOTS, width), dtype=np.int64)
        self._out_nodes = np.zeros(width, dtype=np.int64)
        self._out_counts = np.zeros(width, dtype=np.int64)

    def check(self, chain: Sequence[EdgeTypeTuple]) -> None:
        """Raise unless ``chain`` can be counted on this graph.

        Parameters
        ----------
        chain : Sequence[EdgeTypeTuple]
            Body edge types in chain order.

        Raises
        ------
        InvalidRuleError
            If ``chain`` is empty or does not join end to end.
        GraphSchemaError
            If an edge type of ``chain`` is missing from, or empty in, the
            graph.
        """
        validate_chain(chain)
        forward_csr_tables(self._graph).edge_ids(chain)

    def from_source(
        self, chain: Sequence[EdgeTypeTuple], source: int
    ) -> tuple[Tensor, Tensor]:
        """Return the entities ``chain`` reaches from ``source``, with path counts.

        Parameters
        ----------
        chain : Sequence[EdgeTypeTuple]
            Body edge types in chain order.
        source : int
            An entity of the chain's starting node type.

        Returns
        -------
        tuple[Tensor, Tensor]
            Reached entity ids, ascending, and the float path count to each.

        Raises
        ------
        InvalidRuleError
            If ``chain`` is empty or does not join end to end.
        """
        validate_chain(chain)
        return self._count(chain, source, forward_csr_tables(self._graph))

    def into_target(
        self, chain: Sequence[EdgeTypeTuple], target: int
    ) -> tuple[Tensor, Tensor]:
        """Return the entities ``chain`` reaches ``target`` from, with path counts.

        Parameters
        ----------
        chain : Sequence[EdgeTypeTuple]
            Body edge types in *forward* chain order.
        target : int
            An entity of the chain's ending node type.

        Returns
        -------
        tuple[Tensor, Tensor]
            Source entity ids, ascending, and the float path count from each.

        Raises
        ------
        InvalidRuleError
            If ``chain`` is empty or does not join end to end.
        """
        validate_chain(chain)
        backwards = tuple(mirrored_edge_type(et) for et in reversed(tuple(chain)))
        return self._count(backwards, target, reverse_csr_tables(self._graph))

    def _count(
        self, chain: Sequence[EdgeTypeTuple], source: int, tables: CsrTables
    ) -> tuple[Tensor, Tensor]:
        """Run the kernel over one direction's tables and sort its output."""
        size = count_chain_paths(
            tables.crow_all,
            tables.col_all,
            tables.crow_offsets,
            tables.col_offsets,
            tables.edge_ids(chain),
            source,
            self._counts,
            self._frontier,
            self._out_nodes,
            self._out_counts,
        )
        order = np.argsort(self._out_nodes[:size], kind="stable")
        nodes = self._out_nodes[:size][order]
        counts = self._out_counts[:size][order]
        return (
            torch.from_numpy(nodes),
            torch.from_numpy(counts).to(torch.float32),
        )
