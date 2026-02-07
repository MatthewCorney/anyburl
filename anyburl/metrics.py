"""Rule quality evaluation: support, confidence, and head coverage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anyburl.rule import Rule

logger = logging.getLogger(__name__)

__all__ = ["MetricsConfig", "RuleEvaluator", "RuleMetrics"]

# Default thresholds for metrics filtering
DEFAULT_MIN_SUPPORT: int = 2
DEFAULT_MIN_CONFIDENCE: float = 0.01
DEFAULT_MIN_HEAD_COVERAGE: float = 0.01


@dataclass(frozen=True, slots=True)
class RuleMetrics:
    """Computed quality metrics for a single rule.

    Parameters
    ----------
    support : int
        Number of known triples correctly predicted by the rule.
    confidence : float
        ``support / (support + incorrect_predictions)``.
        In ``[0.0, 1.0]``. Higher is better.
    head_coverage : float
        ``support / total_triples_with_head_relation``.
        In ``[0.0, 1.0]``. Higher is better.
    num_predictions : int
        Total predictions made by the rule (correct + incorrect).
    """

    support: int
    confidence: float
    head_coverage: float
    num_predictions: int

    @property
    def is_trivial(self) -> bool:
        """Return ``True`` if the rule makes zero predictions."""
        return self.num_predictions == 0

    def passes_thresholds(
        self,
        *,
        min_support: int,
        min_confidence: float,
        min_head_coverage: float,
    ) -> bool:
        """Check whether this rule meets all quality thresholds.

        Parameters
        ----------
        min_support : int
            Minimum required support.
        min_confidence : float
            Minimum required confidence.
        min_head_coverage : float
            Minimum required head coverage.

        Returns
        -------
        bool
            ``True`` if all thresholds are met.
        """
        return (
            self.support >= min_support
            and self.confidence >= min_confidence
            and self.head_coverage >= min_head_coverage
        )


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Configuration for the rule evaluator.

    Parameters
    ----------
    min_support : int
        Minimum support for a rule to be considered valid.
    min_confidence : float
        Minimum confidence threshold in ``[0.0, 1.0]``.
    min_head_coverage : float
        Minimum head coverage threshold in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.
    """

    min_support: int = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_head_coverage: float = DEFAULT_MIN_HEAD_COVERAGE

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.min_support < 1:
            raise ValueError(f"min_support must be positive, got {self.min_support}")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {self.min_confidence}"
            )
        if not (0.0 <= self.min_head_coverage <= 1.0):
            raise ValueError(
                f"min_head_coverage must be in [0.0, 1.0], got {self.min_head_coverage}"
            )


class RuleEvaluator:
    """Evaluates rule quality against a knowledge graph.

    Computes support, confidence, and head coverage for rules by
    applying them to the graph and counting predictions.

    Implementation is deferred until the graph backend is selected
    (PyG, igraph, or DGL) after benchmarking.

    Parameters
    ----------
    config : MetricsConfig
        Evaluation configuration with quality thresholds.
    """

    def __init__(self, config: MetricsConfig) -> None:
        self._config = config

    def evaluate(self, rule: Rule) -> RuleMetrics:
        """Compute quality metrics for a single rule.

        Parameters
        ----------
        rule : Rule
            The rule to evaluate.

        Returns
        -------
        RuleMetrics
            The computed metrics.

        Raises
        ------
        NotImplementedError
            Always. Implementation deferred to post-benchmarking.
        """
        raise NotImplementedError(
            "RuleEvaluator.evaluate is not yet implemented. "
            "Awaiting graph backend selection from benchmarking."
        )

    def evaluate_batch(self, rules: Sequence[Rule]) -> list[tuple[Rule, RuleMetrics]]:
        """Evaluate multiple rules, returning those that pass thresholds.

        Parameters
        ----------
        rules : Sequence[Rule]
            The rules to evaluate.

        Returns
        -------
        list[tuple[Rule, RuleMetrics]]
            Rules paired with their metrics, filtered to only those
            passing the configured thresholds.

        Raises
        ------
        NotImplementedError
            Always. Implementation deferred to post-benchmarking.
        """
        raise NotImplementedError(
            "RuleEvaluator.evaluate_batch is not yet implemented. "
            "Awaiting graph backend selection from benchmarking."
        )
