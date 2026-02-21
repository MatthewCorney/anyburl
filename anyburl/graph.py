"""HeteroGraph wrapper over PyG HeteroData with CSR-backed operations."""

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_torch_csr_tensor

from ._logging import get_logger

logger = get_logger(__name__)

EdgeTypeTuple = tuple[str, str, str]
"""(source_node_type, relation, destination_node_type)."""


@dataclass(frozen=True, slots=True)
class _EdgeTypeIndex:
    """CSR index data for a single edge type.

    Parameters
    ----------
    crow_indices : Tensor
        Compressed row offsets (length = num_src + 1).
    col_indices : Tensor
        Column indices for each non-zero entry.
    num_src : int
        Number of source nodes.
    num_dst : int
        Number of destination nodes.
    """

    crow_indices: Tensor
    col_indices: Tensor
    num_src: int
    num_dst: int


class HeteroGraph:
    """Wraps a PyG ``HeteroData`` with precomputed CSR indices.

    Provides efficient neighbor lookup, CSR matrix retrieval for sparse
    matmul-based rule grounding, and edge sampling support.

    Parameters
    ----------
    data : HeteroData
        A PyG heterogeneous graph. Must have at least one edge type
        with a non-empty edge index.

    Raises
    ------
    ValueError
        If the graph has no edge types or all edge indices are empty.
    """

    def __init__(self, data: HeteroData) -> None:
        edge_types: list[EdgeTypeTuple] = data.edge_types
        if not edge_types:
            raise ValueError("HeteroData has no edge types")

        self._node_counts: dict[str, int] = {}
        for node_type in data.node_types:
            self._node_counts[node_type] = int(data[node_type].num_nodes)

        self._edge_indices: dict[EdgeTypeTuple, Tensor] = {}
        self._edge_counts: dict[EdgeTypeTuple, int] = {}
        self._csr_index: dict[EdgeTypeTuple, _EdgeTypeIndex] = {}

        has_edges = False
        for et in edge_types:
            src_type, _, dst_type = et
            edge_index: Tensor = data[et].edge_index
            num_edges = edge_index.size(1)
            self._edge_indices[et] = edge_index
            self._edge_counts[et] = num_edges
            if num_edges == 0:
                continue
            has_edges = True
            num_src = self._node_counts[src_type]
            num_dst = self._node_counts[dst_type]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*Sparse CSR tensor support.*"
                )
                csr = to_torch_csr_tensor(edge_index, size=(num_src, num_dst))
            self._csr_index[et] = _EdgeTypeIndex(
                crow_indices=csr.crow_indices(),
                col_indices=csr.col_indices(),
                num_src=num_src,
                num_dst=num_dst,
            )

        if not has_edges:
            raise ValueError("All edge types have empty edge indices")

        self._outgoing_edge_types: dict[str, tuple[EdgeTypeTuple, ...]] = {}
        for node_type in data.node_types:
            matching = tuple(
                et for et in edge_types if et[0] == node_type and et in self._csr_index
            )
            self._outgoing_edge_types[node_type] = matching

        self._csr_matrix_cache: dict[EdgeTypeTuple, Tensor] = {}

        logger.debug(
            "Built HeteroGraph: %d node types, %d edge types, %d total edges",
            len(self._node_counts),
            len(self._csr_index),
            sum(self._edge_counts.values()),
        )

    def get_neighbors(self, node_id: int, edge_type: EdgeTypeTuple) -> Tensor:
        """Return neighbor IDs reachable from ``node_id`` via ``edge_type``.

        Parameters
        ----------
        node_id : int
            Source node index.
        edge_type : EdgeTypeTuple
            The ``(src_type, relation, dst_type)`` edge type.

        Returns
        -------
        Tensor
            1-D tensor of destination node IDs.

        Raises
        ------
        ValueError
            If ``edge_type`` is unknown or has no edges.
        """
        idx = self._get_csr_index(edge_type)
        start = int(idx.crow_indices[node_id].item())
        end = int(idx.crow_indices[node_id + 1].item())
        return idx.col_indices[start:end]

    def get_csr_matrix(self, edge_type: EdgeTypeTuple) -> Tensor:
        """Return a float CSR sparse tensor for ``edge_type``.

        The matrix has shape ``(num_src, num_dst)`` with ones at
        positions corresponding to edges. Cached after first call.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The ``(src_type, relation, dst_type)`` edge type.

        Returns
        -------
        Tensor
            Sparse CSR float tensor.

        Raises
        ------
        ValueError
            If ``edge_type`` is unknown or has no edges.
        """
        cached = self._csr_matrix_cache.get(edge_type)
        if cached is not None:
            return cached

        idx = self._get_csr_index(edge_type)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            matrix = torch.sparse_csr_tensor(
                idx.crow_indices,
                idx.col_indices,
                torch.ones(idx.col_indices.size(0), dtype=torch.float32),
                size=(idx.num_src, idx.num_dst),
            )
        self._csr_matrix_cache[edge_type] = matrix
        return matrix

    def node_count(self, node_type: str) -> int:
        """Return the number of nodes of the given type.

        Parameters
        ----------
        node_type : str
            The node type name.

        Returns
        -------
        int
            Number of nodes.

        Raises
        ------
        ValueError
            If ``node_type`` is unknown.
        """
        count = self._node_counts.get(node_type)
        if count is None:
            raise ValueError(f"Unknown node type: {node_type!r}")
        return count

    def edge_count(self, edge_type: EdgeTypeTuple) -> int:
        """Return the number of edges of the given type.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The ``(src_type, relation, dst_type)`` edge type.

        Returns
        -------
        int
            Number of edges.

        Raises
        ------
        ValueError
            If ``edge_type`` is unknown.
        """
        count = self._edge_counts.get(edge_type)
        if count is None:
            raise ValueError(f"Unknown edge type: {edge_type!r}")
        return count

    def total_edge_count(self) -> int:
        """Return the total number of edges across all edge types.

        Returns
        -------
        int
            Sum of edge counts.
        """
        return sum(self._edge_counts.values())

    def outgoing_edge_types(self, node_type: str) -> tuple[EdgeTypeTuple, ...]:
        """Return edge types originating from ``node_type``.

        Only includes edge types that have at least one edge (i.e. have
        a CSR index built).

        Parameters
        ----------
        node_type : str
            The source node type.

        Returns
        -------
        tuple[EdgeTypeTuple, ...]
            Edge types with ``src_type == node_type``.

        Raises
        ------
        ValueError
            If ``node_type`` is unknown.
        """
        result = self._outgoing_edge_types.get(node_type)
        if result is None:
            raise ValueError(f"Unknown node type: {node_type!r}")
        return result

    def edge_index(self, edge_type: EdgeTypeTuple) -> Tensor:
        """Return the raw ``[2, N]`` edge index tensor.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The ``(src_type, relation, dst_type)`` edge type.

        Returns
        -------
        Tensor
            Shape ``[2, num_edges]`` with source and destination IDs.

        Raises
        ------
        ValueError
            If ``edge_type`` is unknown.
        """
        ei = self._edge_indices.get(edge_type)
        if ei is None:
            raise ValueError(f"Unknown edge type: {edge_type!r}")
        return ei

    @property
    def node_types(self) -> tuple[str, ...]:
        """Return all node types in the graph.

        Returns
        -------
        tuple[str, ...]
            Node type names.
        """
        return tuple(self._node_counts.keys())

    @property
    def edge_types(self) -> tuple[EdgeTypeTuple, ...]:
        """Return all edge types in the graph.

        Returns
        -------
        tuple[EdgeTypeTuple, ...]
            Edge type tuples.
        """
        return tuple(self._edge_counts.keys())

    def _get_csr_index(self, edge_type: EdgeTypeTuple) -> _EdgeTypeIndex:
        """Look up the CSR index for an edge type, raising on unknown.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The edge type to look up.

        Returns
        -------
        _EdgeTypeIndex
            The CSR index data.

        Raises
        ------
        ValueError
            If ``edge_type`` is unknown or has no edges.
        """
        idx = self._csr_index.get(edge_type)
        if idx is None:
            if edge_type in self._edge_counts:
                raise ValueError(f"Edge type {edge_type!r} exists but has no edges")
            raise ValueError(f"Unknown edge type: {edge_type!r}")
        return idx
