"""Configuration for the random walk engine."""

from dataclasses import dataclass
from enum import StrEnum

DEFAULT_MAX_WALK_LENGTH: int = 5
DEFAULT_MIN_WALK_LENGTH: int = 2
DEFAULT_MAX_WALK_ATTEMPTS: int = 100
DEFAULT_RANDOM_SEED: int = 42


class WalkStrategy(StrEnum):
    """Strategy for selecting the next edge during a random walk.

    Attributes
    ----------
    UNIFORM : str
        Select uniformly at random among all outgoing edges.
    RELATION_WEIGHTED : str
        Weight edges by inverse frequency of their relation type,
        favoring rarer relations to discover more diverse rules.
    """

    UNIFORM = "uniform"
    RELATION_WEIGHTED = "relation_weighted"


@dataclass(frozen=True, slots=True)
class WalkConfig:
    """Configuration for random walks on the knowledge graph.

    Parameters
    ----------
    max_length : int
        Maximum number of steps in a walk.
    min_length : int
        Minimum number of steps for a walk to be considered valid.
    max_attempts : int
        Maximum walk attempts per target triple before giving up.
    strategy : WalkStrategy
        How to select the next edge during a walk.
    seed : int
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.
    """

    max_length: int = DEFAULT_MAX_WALK_LENGTH
    min_length: int = DEFAULT_MIN_WALK_LENGTH
    max_attempts: int = DEFAULT_MAX_WALK_ATTEMPTS
    strategy: WalkStrategy = WalkStrategy.UNIFORM
    seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_length < 1:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.min_length < 1:
            raise ValueError(f"min_length must be positive, got {self.min_length}")
        if self.min_length > self.max_length:
            raise ValueError(
                f"min_length ({self.min_length}) must be <= "
                f"max_length ({self.max_length})"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be positive, got {self.max_attempts}")
