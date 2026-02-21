"""Triple sampling submodule for AnyBURL."""

from .base import BaseTripleSampler, SamplerConfig, SamplingStrategy, Triple
from .uniform_sampler import UniformTripleSampler
from .weighted_sampler import WeightedTripleSampler

__all__ = [
    "BaseTripleSampler",
    "SamplerConfig",
    "SamplingStrategy",
    "Triple",
    "UniformTripleSampler",
    "WeightedTripleSampler",
]
