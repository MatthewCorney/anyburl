"""Warning for rule types that head coverage alone eliminates."""

from collections import defaultdict
from collections.abc import Sequence

from .._logging import get_logger
from ..rule import Rule, RuleConfig, RuleType
from .rule_metrics import RuleMetrics

__all__ = ["HeadCoverageWarning"]

logger = get_logger(__name__)


class HeadCoverageWarning:
    """Logs once per rule type when head coverage removes all its viable rules.

    Head coverage is not comparable across rule types, so a single floor can
    silently remove every AC1 rule while cyclic rules pass.

    Parameters
    ----------
    config : RuleConfig
        The quality floors being applied.
    """

    def __init__(self, config: RuleConfig) -> None:
        self._config = config
        self._warned_types: set[RuleType] = set()

    def check(
        self,
        rules: Sequence[Rule],
        metrics_by_rule: dict[Rule, RuleMetrics],
    ) -> None:
        """Warn when head coverage alone removes every rule of a type.

        Only rules that clear support and confidence are considered, so a
        type of uniformly poor rules is not reported.

        Parameters
        ----------
        rules : Sequence[Rule]
            The rules that were evaluated.
        metrics_by_rule : dict[Rule, RuleMetrics]
            Their computed metrics.
        """
        by_type: dict[RuleType, list[RuleMetrics]] = defaultdict(list)
        for rule in rules:
            by_type[rule.rule_type].append(metrics_by_rule[rule])

        for rule_type, metrics in by_type.items():
            if rule_type in self._warned_types:
                continue
            eliminated = self._eliminated_by_head_coverage(rule_type, metrics)
            if not eliminated:
                continue
            self._warned_types.add(rule_type)
            logger.warning(
                "All %d %s rules that met support and confidence were removed "
                "by min_head_coverage=%.5f; the best any of them reached was "
                "%.5f. Head coverage is not comparable across rule types -- "
                "give this one its own floor via RuleConfig.per_type.",
                len(eliminated),
                rule_type.value,
                self._config.thresholds_for(rule_type).min_head_coverage,
                max(m.head_coverage for m in eliminated),
            )

    def _eliminated_by_head_coverage(
        self,
        rule_type: RuleType,
        metrics: Sequence[RuleMetrics],
    ) -> list[RuleMetrics]:
        """Return the rules of a type that failed only on head coverage.

        Returns an empty list unless every rule clearing support and
        confidence then fails the head coverage floor.
        """
        thresholds = self._config.thresholds_for(rule_type)
        if thresholds.min_head_coverage <= 0.0:
            return []
        otherwise_eligible = [
            m
            for m in metrics
            if m.support >= thresholds.min_support
            and m.confidence >= thresholds.min_confidence
        ]
        if any(
            m.head_coverage >= thresholds.min_head_coverage for m in otherwise_eligible
        ):
            return []
        return otherwise_eligible
