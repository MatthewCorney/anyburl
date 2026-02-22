"""Benchmarks for HeteroGraph construction, neighbour lookup, and CSR cache."""

from anyburl.graph import HeteroGraph

ET_AB = ("A", "r_ab", "B")


def test_graph_construction(benchmark, raw_hetero_data):
    """Benchmark HeteroGraph construction (CSR index build) from raw HeteroData."""
    benchmark(HeteroGraph, raw_hetero_data)


def test_neighbor_lookup(benchmark, graph):
    """Benchmark single-node neighbour retrieval via the CSR index."""
    benchmark(graph.get_neighbors, 0, ET_AB)


def test_csr_matrix_warm(benchmark, graph):
    """Benchmark CSR float-matrix retrieval with a warm cache."""
    graph.get_csr_matrix(ET_AB)  # populate cache before timing
    benchmark(graph.get_csr_matrix, ET_AB)
