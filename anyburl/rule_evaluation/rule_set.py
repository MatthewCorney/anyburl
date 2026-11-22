"""Learned rules paired with their metrics, grouped by the relation they predict."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..core import EdgeTypeTuple
from ..rule import Rule, RuleMetrics

__all__ = ["RuleSet"]


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Rules that passed their quality thresholds, with their metrics.

    Parameters
    ----------
    results : tuple[tuple[Rule, RuleMetrics], ...]
        Rules paired with their metrics, in the order they were learned.
    """

    results: tuple[tuple[Rule, RuleMetrics], ...]

    @classmethod
    def of(cls, results: Iterable[tuple[Rule, RuleMetrics]]) -> "RuleSet":
        """Build a rule set from ``(rule, metrics)`` pairs.

        Parameters
        ----------
        results : Iterable[tuple[Rule, RuleMetrics]]
            Rules paired with their metrics.

        Returns
        -------
        RuleSet
            The pairs, in the given order.
        """
        return cls(results=tuple(results))

    def __len__(self) -> int:
        """Return the number of rules."""
        return len(self.results)

    def __iter__(self) -> Iterator[tuple[Rule, RuleMetrics]]:
        """Iterate ``(rule, metrics)`` pairs in learned order."""
        return iter(self.results)

    def head_edge_types(self) -> tuple[EdgeTypeTuple, ...]:
        """Return the relations the rules predict, in order of first appearance."""
        return tuple(dict.fromkeys(rule.head.edge_signature for rule, _ in self))

    def for_edge_type(self, edge_type: EdgeTypeTuple) -> "RuleSet":
        """Return the rules predicting ``edge_type``, in learned order.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The head relation.

        Returns
        -------
        RuleSet
            Its rules; empty when none were learned.
        """
        return RuleSet.of(
            (rule, metrics)
            for rule, metrics in self
            if rule.head.edge_signature == edge_type
        )
