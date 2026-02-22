"""Benchmarks for RuleEvaluator on CYCLIC, AC1, and AC2 rules."""

import pytest

from anyburl.metrics import RuleEvaluator
from anyburl.rule import RuleConfig

from ._helpers import make_ac1_rule, make_ac2_rule, make_cyclic_rule

RULE_FACTORIES = [make_cyclic_rule, make_ac1_rule, make_ac2_rule]

PERMISSIVE_CONFIG = RuleConfig(
    min_support=1,
    min_confidence=0.0,
    min_head_coverage=0.0,
)


@pytest.mark.parametrize(
    "rule_fn",
    RULE_FACTORIES,
    ids=["cyclic", "ac1", "ac2"],
)
def test_evaluate_rule(benchmark, eval_graph, rule_fn):
    """Benchmark evaluating a single rule (CYCLIC / AC1 / AC2) on small and medium graphs."""
    evaluator = RuleEvaluator(eval_graph, PERMISSIVE_CONFIG)
    benchmark(evaluator.evaluate, rule_fn())


def test_evaluate_batch(benchmark, eval_graph):
    """Benchmark evaluate_batch on a mixed set of three rules."""
    evaluator = RuleEvaluator(eval_graph, PERMISSIVE_CONFIG)
    rules = [make_cyclic_rule(), make_ac1_rule(), make_ac2_rule()]
    benchmark(evaluator.evaluate_batch, rules)
