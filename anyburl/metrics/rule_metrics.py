"""Rule quality metrics and their noisy-or aggregation."""

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "NO_PREDICTIONS",
    "RuleMetrics",
    "aggregate_confidence",
    "metrics_from_counts",
]


@dataclass(frozen=True, slots=True)
class RuleMetrics:
    """Computed quality metrics for a single rule.

    Parameters
    ----------
    support : int
        Number of known triples correctly predicted by the rule.
    confidence : float
        ``support / num_predictions``, in ``[0.0, 1.0]``.
    head_coverage : float
        ``support / total_triples_with_head_relation``, in ``[0.0, 1.0]``.
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


NO_PREDICTIONS: RuleMetrics = RuleMetrics(
    support=0, confidence=0.0, head_coverage=0.0, num_predictions=0
)
"""Metrics of a rule whose body has no groundings."""


def aggregate_confidence(confidences: Sequence[float]) -> float:
    """Aggregate confidences from multiple rules via noisy-or.

    Computes ``1 - prod(1 - c_i)``, treating each rule as an independent
    chance of the triple being true.

    Parameters
    ----------
    confidences : Sequence[float]
        Individual rule confidences, each in ``[0.0, 1.0]``.

    Returns
    -------
    float
        Aggregated confidence in ``[0.0, 1.0]``; ``0.0`` for an empty
        sequence.
    """
    result = 1.0
    for confidence in confidences:
        result *= 1.0 - confidence
    return 1.0 - result


def metrics_from_counts(
    num_predictions: int,
    support: int,
    total_head_triples: int,
) -> RuleMetrics:
    """Build metrics from a rule's prediction and support counts.

    Parameters
    ----------
    num_predictions : int
        Distinct pairs the rule predicts.
    support : int
        How many of those the head relation already contains.
    total_head_triples : int
        Triples carrying the head relation, for head coverage.

    Returns
    -------
    RuleMetrics
        Computed metrics.
    """
    if num_predictions == 0:
        return NO_PREDICTIONS
    return RuleMetrics(
        support=support,
        confidence=support / num_predictions,
        head_coverage=support / total_head_triples if total_head_triples > 0 else 0.0,
        num_predictions=num_predictions,
    )
