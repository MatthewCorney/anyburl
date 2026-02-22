"""Benchmarks for the WalkEngine at walk lengths 2 and 4, both strategies."""

import pytest

from anyburl.anyburl import build_walk_engine
from anyburl.sampler import Triple
from anyburl.walk import WalkConfig, WalkStrategy

# A triple that exists in the synthetic graph topology: head A:0, tail C:0.
# The walk only requires head_id / head_type as starting point; reaching
# tail_id is governed by max_attempts and graph connectivity.
BENCH_TRIPLE = Triple(
    head_id=0,
    tail_id=0,
    head_type="A",
    tail_type="C",
    relation="r_ac",
)


@pytest.mark.parametrize("max_length", [2, 4])
@pytest.mark.parametrize("strategy", [WalkStrategy.UNIFORM, WalkStrategy.RELATION_WEIGHTED])
def test_walk(benchmark, graph, strategy, max_length):
    """Benchmark walk_from_triple over varying walk lengths and edge-selection strategies."""
    config = WalkConfig(max_length=max_length, strategy=strategy)
    engine = build_walk_engine(graph, config)
    benchmark(engine.walk_from_triple, BENCH_TRIPLE)
