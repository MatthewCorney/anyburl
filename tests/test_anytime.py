"""Tests for the anytime mining loop."""

from collections.abc import Sequence

import pytest

from anyburl.anytime import (
    AnytimeConfig,
    AnytimeLearner,
    MiningStages,
    PathWalker,
)
from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleEvaluator
from anyburl.rule import PathStep, RuleConfig, RuleGeneralizer
from anyburl.sampler import Triple

TARGET = Triple(
    head_id=0, tail_id=1, head_type="person", tail_type="city", relation="lives_in"
)


class _FixedWalker:
    """Returns the same paths every call, so batches repeat themselves."""

    def __init__(self, paths: list[list[PathStep]]) -> None:
        self.paths = paths
        self.calls = 0

    def walk_from_triple(self, triple: Triple) -> list[list[PathStep]]:
        del triple
        self.calls += 1
        return self.paths


def _stages(
    graph: HeteroGraph,
    walker: PathWalker,
    *,
    batch: Sequence[Triple] = (TARGET,),
) -> MiningStages:
    config = RuleConfig(min_support=1, min_confidence=0.0, min_head_coverage=0.0)
    return MiningStages(
        sample_batch=lambda index: list(batch),
        walker_for_length=lambda length: walker,
        generalizer=RuleGeneralizer(config),
        evaluator=RuleEvaluator(graph, config),
    )


@pytest.fixture
def repeating_paths() -> list[list[PathStep]]:
    return [[(0, "person", "born_in"), (0, "city", "near"), (1, "city", "")]]


def test_saturation_stops_a_length_once_rules_repeat(
    evaluator_graph: HeteroGraph, repeating_paths: list[list[PathStep]]
) -> None:
    """A walker that repeats itself must not be mined forever."""
    walker = _FixedWalker(repeating_paths)
    config = AnytimeConfig(total_seconds=30.0, min_length=2, max_length=2)

    _, report = AnytimeLearner(_stages(evaluator_graph, walker), config).learn()

    assert report.lengths[0].saturated
    assert report.lengths[0].batches == 2
    assert report.completed


def test_min_batches_per_length_is_respected(
    evaluator_graph: HeteroGraph, repeating_paths: list[list[PathStep]]
) -> None:
    """One unlucky batch must not be able to end a length early."""
    walker = _FixedWalker(repeating_paths)
    config = AnytimeConfig(
        total_seconds=30.0, min_length=2, max_length=2, min_batches_per_length=4
    )

    _, report = AnytimeLearner(_stages(evaluator_graph, walker), config).learn()

    assert report.lengths[0].batches == 4


def test_every_length_in_range_is_mined(
    evaluator_graph: HeteroGraph, repeating_paths: list[list[PathStep]]
) -> None:
    walker = _FixedWalker(repeating_paths)
    config = AnytimeConfig(total_seconds=30.0, min_length=2, max_length=4)

    _, report = AnytimeLearner(_stages(evaluator_graph, walker), config).learn()

    assert [r.length for r in report.lengths] == [2, 3, 4]


def test_budget_cuts_off_a_length_that_cannot_saturate(
    evaluator_graph: HeteroGraph, repeating_paths: list[list[PathStep]]
) -> None:
    """With saturation held off, only the wall clock can end the run."""
    walker = _FixedWalker(repeating_paths)
    config = AnytimeConfig(
        total_seconds=0.05,
        min_length=2,
        max_length=2,
        min_batches_per_length=10**9,
    )

    _, report = AnytimeLearner(_stages(evaluator_graph, walker), config).learn()

    assert not report.completed
    assert not report.lengths[0].saturated
    assert report.seconds >= 0.05
    assert walker.calls > 1


def test_rules_are_deduplicated_across_batches(
    evaluator_graph: HeteroGraph, repeating_paths: list[list[PathStep]]
) -> None:
    """The same path seen twice must not yield the rule twice."""
    walker = _FixedWalker(repeating_paths)
    config = AnytimeConfig(
        total_seconds=30.0, min_length=2, max_length=2, min_batches_per_length=3
    )

    results, _ = AnytimeLearner(_stages(evaluator_graph, walker), config).learn()

    assert len(results) == len({str(rule) for rule, _ in results})


def test_a_walker_returning_nothing_saturates_immediately(
    evaluator_graph: HeteroGraph,
) -> None:
    """Walks that never reach the target will not start doing so."""
    config = AnytimeConfig(total_seconds=30.0, min_length=2, max_length=2)

    _, report = AnytimeLearner(
        _stages(evaluator_graph, _FixedWalker([])), config
    ).learn()

    assert report.lengths[0].saturated
    assert report.lengths[0].new_rules == 0


def test_report_describes_the_run(
    evaluator_graph: HeteroGraph, repeating_paths: list[list[PathStep]]
) -> None:
    config = AnytimeConfig(total_seconds=30.0, min_length=2, max_length=2)

    _, report = AnytimeLearner(
        _stages(evaluator_graph, _FixedWalker(repeating_paths)), config
    ).learn()

    assert "len 2" in report.describe()
    assert report.new_rules == report.lengths[0].new_rules


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_seconds": 0.0}, "total_seconds must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"min_length": 0}, "min_length must be positive"),
        ({"min_length": 4, "max_length": 2}, "is below min_length"),
        ({"saturation": 1.5}, "saturation must be in"),
        ({"min_batches_per_length": 0}, "min_batches_per_length must be positive"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AnytimeConfig(**kwargs)  # type: ignore[arg-type]
