"""Quality floors a rule must clear to be retained."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..exceptions import ConfigurationError
from .model import RuleType

__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_MIN_HEAD_COVERAGE",
    "DEFAULT_MIN_SUPPORT",
    "RuleConfig",
    "RuleThresholds",
]

DEFAULT_MIN_SUPPORT: int = 2
DEFAULT_MIN_CONFIDENCE: float = 0.01
DEFAULT_MIN_HEAD_COVERAGE: float = 0.01


@dataclass(frozen=True, slots=True)
class RuleThresholds:
    """Quality floors a rule must clear to be retained.

    Parameters
    ----------
    min_support : int
        Minimum support threshold for a rule to be retained.
    min_confidence : float
        Minimum confidence threshold in [0.0, 1.0].
    min_head_coverage : float
        Minimum head coverage threshold in [0.0, 1.0].

    Raises
    ------
    ConfigurationError
        If any parameter is out of its valid range.
    """

    min_support: int = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_head_coverage: float = DEFAULT_MIN_HEAD_COVERAGE

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.min_support < 1:
            raise ConfigurationError(
                f"min_support must be positive, got {self.min_support}"
            )
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ConfigurationError(
                f"min_confidence must be in [0.0, 1.0], got {self.min_confidence}"
            )
        if not (0.0 <= self.min_head_coverage <= 1.0):
            raise ConfigurationError(
                f"min_head_coverage must be in [0.0, 1.0], got {self.min_head_coverage}"
            )


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Configuration for rule generalization and filtering.

    The top-level floors apply to every rule type unless ``per_type``
    overrides them. Head coverage in particular is not comparable across
    types: an AC1 rule is pinned to one entity, so its support cannot exceed
    that entity's degree, and it usually needs a much lower floor than a
    cyclic rule.

    Parameters
    ----------
    min_support : int
        Default minimum support threshold.
    min_confidence : float
        Default minimum confidence threshold in [0.0, 1.0].
    min_head_coverage : float
        Default minimum head coverage threshold in [0.0, 1.0].
    per_type : Mapping[RuleType, RuleThresholds]
        Floors that replace the defaults for the named rule types.

    Raises
    ------
    ConfigurationError
        If any default parameter is out of its valid range.
    """

    min_support: int = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_head_coverage: float = DEFAULT_MIN_HEAD_COVERAGE
    per_type: Mapping[RuleType, RuleThresholds] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self.default_thresholds()

    def default_thresholds(self) -> RuleThresholds:
        """Return the floors applied to rule types without an override."""
        return RuleThresholds(
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            min_head_coverage=self.min_head_coverage,
        )

    def thresholds_for(self, rule_type: RuleType) -> RuleThresholds:
        """Return the floors a rule of this type must clear.

        Parameters
        ----------
        rule_type : RuleType
            The rule type being filtered.

        Returns
        -------
        RuleThresholds
            The override for this type, else the defaults.
        """
        override = self.per_type.get(rule_type)
        if override is not None:
            return override
        return self.default_thresholds()
