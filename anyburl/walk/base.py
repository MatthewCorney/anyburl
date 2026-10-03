"""Configuration for the random walk engine."""

from dataclasses import dataclass
from enum import StrEnum

from ..exceptions import ConfigurationError

DEFAULT_MAX_WALK_LENGTH: int = 5
DEFAULT_MIN_WALK_LENGTH: int = 2
DEFAULT_MAX_WALK_ATTEMPTS: int = 100
DEFAULT_RANDOM_SEED: int = 42

EMPTY_RELATION: str = ""
"""Relation of a walk's final step, which has no outgoing edge."""


class WalkStrategy(StrEnum):
    """Strategy for selecting the next edge during a random walk.

    Attributes
    ----------
    UNIFORM : str
        Select uniformly at random among all outgoing edges.
    RELATION_WEIGHTED : str
        Weight edges by inverse frequency of their relation type,
        favoring rarer relations to discover more diverse rules.
    REACHABILITY_PRUNED : str
        Like ``UNIFORM``, but only among edge types whose destination node
        type can still reach the walk's target type within the remaining
        steps. Removes only hopeless continuations, though it changes the
        distribution over the successful paths that remain.
    """

    UNIFORM = "uniform"
    RELATION_WEIGHTED = "relation_weighted"
    REACHABILITY_PRUNED = "reachability_pruned"


class EdgeWeighting(StrEnum):
    """How much weight each candidate edge type gets at a step.

    Orthogonal to :class:`WalkStrategy`, which decides which edge types are
    candidates at all. Applies to :class:`~anyburl.walk.NumbaWalkEngine`.

    Attributes
    ----------
    UNIFORM : str
        Every candidate edge type is equally likely.
    INVERSE_FREQUENCY : str
        Weight a candidate by the reciprocal of how many edges its type
        has, so abundant relations do not crowd out rare ones.
    """

    UNIFORM = "uniform"
    INVERSE_FREQUENCY = "inverse_frequency"


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
        Which edge types are candidates at each step.
    edge_weighting : EdgeWeighting
        How candidate edge types are weighted.
    seed : int
        Random seed for reproducibility.

    Raises
    ------
    ConfigurationError
        If any parameter is out of its valid range.
    """

    max_length: int = DEFAULT_MAX_WALK_LENGTH
    min_length: int = DEFAULT_MIN_WALK_LENGTH
    max_attempts: int = DEFAULT_MAX_WALK_ATTEMPTS
    strategy: WalkStrategy = WalkStrategy.UNIFORM
    edge_weighting: EdgeWeighting = EdgeWeighting.UNIFORM
    seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_length < 1:
            raise ConfigurationError(
                f"max_length must be positive, got {self.max_length}"
            )
        if self.min_length < 1:
            raise ConfigurationError(
                f"min_length must be positive, got {self.min_length}"
            )
        if self.min_length > self.max_length:
            raise ConfigurationError(
                f"min_length ({self.min_length}) must be <= "
                f"max_length ({self.max_length})"
            )
        if self.max_attempts < 1:
            raise ConfigurationError(
                f"max_attempts must be positive, got {self.max_attempts}"
            )
