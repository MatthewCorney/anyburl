"""Adjacency lookups over a :class:`HeteroGraph`, for test oracles.

Not a fixture module: tests import these helpers directly.
"""

import torch

from anyburl.core import EdgeTypeTuple
from anyburl.graph import HeteroGraph, suppress_sparse_csr_warning


def neighbors(
    graph: HeteroGraph, node_id: int, edge_type: EdgeTypeTuple
) -> torch.Tensor:
    """Return the destination ids reachable from ``node_id`` via ``edge_type``."""
    index = graph.csr_index(edge_type)
    start = int(index.crow_indices[node_id].item())
    end = int(index.crow_indices[node_id + 1].item())
    return index.col_indices[start:end]


def adjacency_matrix(graph: HeteroGraph, edge_type: EdgeTypeTuple) -> torch.Tensor:
    """Return ``edge_type`` as a float sparse CSR matrix with a one per edge."""
    index = graph.csr_index(edge_type)
    with suppress_sparse_csr_warning():
        return torch.sparse_csr_tensor(
            index.crow_indices,
            index.col_indices,
            torch.ones(index.col_indices.size(0), dtype=torch.float32),
            size=(index.num_src, index.num_dst),
        )
