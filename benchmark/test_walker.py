"""Benchmarks for walk engines at walk lengths 2 and 4, both strategies."""

import pytest

from anyburl.factories import build_walk_engine
from anyburl.sampler import Triple
from anyburl.walk import WalkConfig, WalkStrategy

BENCH_TRIPLE = Triple(
    head_id=0,
    tail_id=0,
    head_type="A",
    tail_type="C",
    relation="r_ac",
)


@pytest.mark.parametrize("max_length", [2, 4])
@pytest.mark.parametrize(
    "strategy", [WalkStrategy.UNIFORM, WalkStrategy.RELATION_WEIGHTED]
)
def test_walk(benchmark, graph, strategy, max_length):
    """Benchmark walk_from_triple across walk lengths and strategies."""
    config = WalkConfig(max_length=max_length, strategy=strategy)
    engine = build_walk_engine(graph, config)
    benchmark(engine.walk_from_triple, BENCH_TRIPLE)
