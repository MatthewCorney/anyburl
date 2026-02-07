"""Configuration for the triple sampler."""

from dataclasses import dataclass

from anyburl.enums import SamplingStrategy

__all__ = ["SamplerConfig"]

DEFAULT_SAMPLE_SIZE: int = 1000
DEFAULT_RANDOM_SEED: int = 42


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """Configuration for sampling target triples from the knowledge graph.

    Parameters
    ----------
    sample_size : int
        Number of triples to sample per iteration.
    strategy : SamplingStrategy
        How to weight triple selection.
    seed : int
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If ``sample_size`` is not positive.
    """

    sample_size: int = DEFAULT_SAMPLE_SIZE
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.sample_size < 1:
            raise ValueError(f"sample_size must be positive, got {self.sample_size}")
