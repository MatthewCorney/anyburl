"""Benchmarks for triple sampling at 3 sample sizes and 3 strategies."""

import pytest

from anyburl.anyburl import build_triple_sampler
from anyburl.sampler import SamplerConfig, SamplingStrategy

SAMPLE_SIZES = [100, 1_000, 10_000]
STRATEGIES = [
    SamplingStrategy.UNIFORM,
    SamplingStrategy.RELATION_PROPORTIONAL,
    SamplingStrategy.RELATION_INVERSE,
]


@pytest.mark.parametrize("sample_size", SAMPLE_SIZES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_sampling(benchmark, graph, strategy, sample_size):
    """Benchmark triple sampling at various sample sizes and relation-weighting strategies."""
    config = SamplerConfig(sample_size=sample_size, strategy=strategy)
    sampler = build_triple_sampler(graph, config)
    benchmark(sampler.sample)
