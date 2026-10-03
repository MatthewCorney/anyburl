"""The anytime mining loop: mine rules by length within a time budget.

Mining proceeds by rule length, shortest first. Within a length, batches are
drawn until the rules coming back are overwhelmingly ones already known
(*saturation*), then mining moves to the next length. The run is bounded by a
wall-clock budget, and each batch is evaluated as it lands, so the rule set is
usable whenever the run stops.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ._logging import get_logger
from .exceptions import ConfigurationError
from .metrics import RuleEvaluator, RuleMetrics
from .rule import PathStep, Rule, RuleGeneralizer
from .sampler import Triple

logger = get_logger(__name__)

__all__ = [
    "AnytimeConfig",
    "AnytimeLearner",
    "AnytimeReport",
    "LengthReport",
    "MiningStages",
    "PathWalker",
]

DEFAULT_TOTAL_SECONDS: float = 10.0
"""Wall-clock budget for a mining run when none is given."""

DEFAULT_SATURATION: float = 0.01
"""New-rule fraction below which a length is considered exhausted."""

DEFAULT_BATCH_SIZE: int = 500
DEFAULT_MIN_RULE_LENGTH: int = 2
DEFAULT_MAX_RULE_LENGTH: int = 4
DEFAULT_MIN_BATCHES_PER_LENGTH: int = 1
DEFAULT_SEED: int = 42


class PathWalker(Protocol):
    """Produces walk paths connecting a triple's endpoints."""

    def walk_from_triple(self, triple: Triple) -> list[list[PathStep]]:
        """Return paths from the triple's head to its tail."""
        ...


@dataclass(frozen=True, slots=True)
class AnytimeConfig:
    """Budget and stopping rules for an anytime mining run.

    Parameters
    ----------
    total_seconds : float
        Wall-clock budget for the whole run. Checked between batches, so a
        single slow batch may overrun it.
    batch_size : int
        Target triples sampled per batch.
    min_length : int
        Shortest rule body to mine.
    max_length : int
        Longest rule body to mine.
    saturation : float
        Stop mining a length once the share of genuinely new rules in a
        batch falls below this. ``0.01`` means "fewer than 1% new".
    min_batches_per_length : int
        Batches to run at a length before saturation may end it, so one
        unlucky batch cannot cut a length short.
    seed : int
        Base seed; each batch derives its own from this.

    Raises
    ------
    ConfigurationError
        If a budget or bound is non-positive, the length range is
        inverted, or ``saturation`` is outside [0.0, 1.0].
    """

    total_seconds: float = DEFAULT_TOTAL_SECONDS
    batch_size: int = DEFAULT_BATCH_SIZE
    min_length: int = DEFAULT_MIN_RULE_LENGTH
    max_length: int = DEFAULT_MAX_RULE_LENGTH
    saturation: float = DEFAULT_SATURATION
    min_batches_per_length: int = DEFAULT_MIN_BATCHES_PER_LENGTH
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.total_seconds <= 0.0:
            raise ConfigurationError(
                f"total_seconds must be positive, got {self.total_seconds}"
            )
        if self.batch_size < 1:
            raise ConfigurationError(
                f"batch_size must be positive, got {self.batch_size}"
            )
        if self.min_length < 1:
            raise ConfigurationError(
                f"min_length must be positive, got {self.min_length}"
            )
        if self.max_length < self.min_length:
            raise ConfigurationError(
                f"max_length {self.max_length} is below min_length {self.min_length}"
            )
        if not 0.0 <= self.saturation <= 1.0:
            raise ConfigurationError(
                f"saturation must be in [0.0, 1.0], got {self.saturation}"
            )
        if self.min_batches_per_length < 1:
            raise ConfigurationError(
                "min_batches_per_length must be positive, got "
                f"{self.min_batches_per_length}"
            )

    @property
    def lengths(self) -> range:
        """Return the rule body lengths to mine, shortest first."""
        return range(self.min_length, self.max_length + 1)


@dataclass(frozen=True, slots=True)
class LengthReport:
    """What one rule length cost and yielded.

    Parameters
    ----------
    length : int
        The rule body length mined.
    batches : int
        Batches run at this length.
    new_rules : int
        Candidate rules first seen at this length.
    seconds : float
        Wall-clock time spent.
    saturated : bool
        ``True`` if mining stopped because new rules dried up, ``False``
        if the budget ran out first.
    """

    length: int
    batches: int
    new_rules: int
    seconds: float
    saturated: bool


@dataclass(frozen=True, slots=True)
class AnytimeReport:
    """A record of how a mining run spent its budget.

    Parameters
    ----------
    lengths : tuple[LengthReport, ...]
        Per-length detail, in the order mined.
    seconds : float
        Total wall-clock time.
    completed : bool
        ``True`` if every length saturated within the budget. ``False``
        means the run was cut off, and a larger budget would find more.
    """

    lengths: tuple[LengthReport, ...]
    seconds: float
    completed: bool

    @property
    def new_rules(self) -> int:
        """Return the total candidate rules found across all lengths."""
        return sum(report.new_rules for report in self.lengths)

    def describe(self) -> str:
        """Return a short human-readable summary of the run."""
        status = "saturated" if self.completed else "budget exhausted"
        per_length = ", ".join(
            f"len {r.length}: {r.new_rules} rules in {r.batches} batch(es)"
            f"{'' if r.saturated else ' (cut off)'}"
            for r in self.lengths
        )
        return f"{self.seconds:.1f}s, {status}; {per_length}"


@dataclass(frozen=True, slots=True)
class MiningStages:
    """The collaborators the loop drives once per batch.

    Parameters
    ----------
    sample_batch : Callable[[int], Sequence[Triple]]
        Returns a fresh batch of target triples given a batch index.
    walker_for_length : Callable[[int], PathWalker]
        Returns a walker restricted to paths of exactly this length.
    generalizer : RuleGeneralizer
        Turns paths into candidate rules.
    evaluator : RuleEvaluator
        Scores candidate rules and applies quality thresholds.
    """

    sample_batch: Callable[[int], Sequence[Triple]]
    walker_for_length: Callable[[int], PathWalker]
    generalizer: RuleGeneralizer
    evaluator: RuleEvaluator


class AnytimeLearner:
    """Mines rules by increasing length until saturated or out of time.

    Parameters
    ----------
    stages : MiningStages
        The per-batch collaborators.
    config : AnytimeConfig
        Budget and stopping rules.
    """

    def __init__(self, stages: MiningStages, config: AnytimeConfig) -> None:
        self._stages = stages
        self._config = config
        self._seen: set[str] = set()
        self._batch_index = 0
        self._last_produced = 0

    def learn(self) -> tuple[list[tuple[Rule, RuleMetrics]], AnytimeReport]:
        """Run the mining loop within the configured budget.

        Returns
        -------
        tuple[list[tuple[Rule, RuleMetrics]], AnytimeReport]
            Rules that passed thresholds, paired with their metrics, and
            a record of how the budget was spent.
        """
        started = time.perf_counter()
        deadline = started + self._config.total_seconds

        results: list[tuple[Rule, RuleMetrics]] = []
        reports: list[LengthReport] = []

        for length in self._config.lengths:
            if time.perf_counter() >= deadline:
                break
            report = self._mine_length(length, deadline, results)
            reports.append(report)

        elapsed = time.perf_counter() - started
        completed = len(reports) == len(self._config.lengths) and all(
            report.saturated for report in reports
        )
        anytime_report = AnytimeReport(
            lengths=tuple(reports), seconds=elapsed, completed=completed
        )
        logger.info("Anytime mining: %s", anytime_report.describe())
        return results, anytime_report

    def _mine_length(
        self,
        length: int,
        deadline: float,
        results: list[tuple[Rule, RuleMetrics]],
    ) -> LengthReport:
        """Mine one rule length until saturated or the deadline passes.

        Parameters
        ----------
        length : int
            Rule body length to mine.
        deadline : float
            ``perf_counter`` value at which to stop.
        results : list[tuple[Rule, RuleMetrics]]
            Accumulated passing rules, extended in place so the caller
            holds a usable model at every point in the run.

        Returns
        -------
        LengthReport
            What this length cost and yielded.
        """
        started = time.perf_counter()
        walker = self._stages.walker_for_length(length)
        batches = 0
        new_rules = 0
        saturated = False

        while time.perf_counter() < deadline:
            candidates = self._run_batch(walker)
            batches += 1
            new_rules += len(candidates)

            if candidates:
                results.extend(self._stages.evaluator.evaluate_batch(candidates))

            if batches >= self._config.min_batches_per_length and self._is_saturated(
                candidates
            ):
                saturated = True
                break

        return LengthReport(
            length=length,
            batches=batches,
            new_rules=new_rules,
            seconds=time.perf_counter() - started,
            saturated=saturated,
        )

    def _run_batch(self, walker: PathWalker) -> list[Rule]:
        """Sample, walk and generalize one batch, keeping only unseen rules.

        Parameters
        ----------
        walker : PathWalker
            The walker for the current length.

        Returns
        -------
        list[Rule]
            Rules not produced by any earlier batch.
        """
        triples = self._stages.sample_batch(self._batch_index)
        self._batch_index += 1
        self._last_produced = 0

        fresh: list[Rule] = []
        for triple in triples:
            for path in walker.walk_from_triple(triple):
                fresh.extend(self._generalize_path(path, triple))
        return fresh

    def _generalize_path(self, path: list[PathStep], triple: Triple) -> list[Rule]:
        """Generalize one path, returning only rules not yet seen."""
        candidates = self._stages.generalizer.generalize(
            path,
            target_relation=triple.relation,
            head_type=triple.head_type,
            tail_type=triple.tail_type,
        )

        fresh: list[Rule] = []
        for rule in candidates:
            self._last_produced += 1
            key = str(rule)
            if key not in self._seen:
                self._seen.add(key)
                fresh.append(rule)
        return fresh

    def _is_saturated(self, fresh: Sequence[Rule]) -> bool:
        """Return whether the latest batch produced too few new rules.

        A batch that produced nothing at all is treated as saturated: the
        walks are not reaching their targets at this length, and repeating
        them will not change that.

        Parameters
        ----------
        fresh : Sequence[Rule]
            Rules from the latest batch that had not been seen before.

        Returns
        -------
        bool
            ``True`` when the length is exhausted.
        """
        if self._last_produced == 0:
            return True
        return len(fresh) / self._last_produced < self._config.saturation
