"""End-to-end AnyBURL pipeline: sample, walk, generalize, evaluate."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

from torch_geometric.data import HeteroData

from ._logging import get_logger, progress
from .acyclic import AcyclicStep, AcyclicStepSampler, generalize_acyclic
from .aggregation import AggregatedScorer, AggregationSetup, fit_aggregated_scorer
from .anytime import (
    AnytimeConfig,
    AnytimeLearner,
    AnytimeReport,
    MiningStages,
)
from .core import DEFAULT_RANDOM_SEED, EdgeTypeTuple, Triple
from .evaluation import (
    EvaluationConfig,
    LinkPredictionMetrics,
    rank_queries_by_edge_type,
    summarize_ranks,
)
from .exceptions import ConfigurationError, NotFittedError
from .graph import HeteroGraph
from .prediction import MultiRelationPredictor, Prediction, PredictionConfig
from .rule import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_HEAD_COVERAGE,
    DEFAULT_MIN_SUPPORT,
    PathStep,
    Rule,
    RuleConfig,
    RuleGeneralizer,
    RuleThresholds,
    RuleType,
)
from .rule_evaluation import RuleEvaluator, RuleSet
from .sampler import SamplerConfig, SamplingStrategy, build_triple_sampler
from .walk import (
    EdgeWeighting,
    NumbaWalkEngine,
    PathWalker,
    WalkConfig,
    WalkStrategy,
)
from .walk.base import DEFAULT_MIN_WALK_LENGTH

__all__ = ["AnyBURL", "AnyBURLConfig"]

logger = get_logger(__name__)

DEFAULT_PIPELINE_SAMPLE_SIZE: int = 2000
DEFAULT_PIPELINE_MAX_WALK_LENGTH: int = 4
DEFAULT_PIPELINE_MAX_WALK_ATTEMPTS: int = 700
NO_ACYCLIC_SAMPLES: int = 0


@dataclass(frozen=True, slots=True)
class AnyBURLConfig:
    """Configuration for the AnyBURL pipeline.

    Parameters
    ----------
    sample_size : int
        Number of triples to sample per iteration.
    sampling_strategy : SamplingStrategy
        How to weight triple selection.
    target_edge_type : EdgeTypeTuple | None
        When set, only sample triples from this edge type.
    max_walk_length : int
        Maximum number of steps per random walk.
    min_walk_length : int
        Minimum number of steps for a valid walk.
    max_walk_attempts : int
        Maximum walk attempts per triple before giving up.
    walk_strategy : WalkStrategy
        Which edge types are candidates at each walk step.
    edge_weighting : EdgeWeighting
        How candidate edge types are weighted at each walk step.
    min_support : int
        Minimum support threshold for rule filtering.
    min_confidence : float
        Minimum confidence threshold for rule filtering.
    min_head_coverage : float
        Minimum head coverage threshold for rule filtering.
    per_type_thresholds : Mapping[RuleType, RuleThresholds]
        Floors replacing the defaults for the named rule types.
    acyclic_samples_per_end : int
        One-step walks drawn from each endpoint of every sampled triple and
        generalized into acyclic rules. Zero learns no acyclic rules.
    seed : int
        Random seed for reproducibility.

    Raises
    ------
    ConfigurationError
        If ``acyclic_samples_per_end`` is negative.
    """

    sample_size: int = DEFAULT_PIPELINE_SAMPLE_SIZE
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    target_edge_type: EdgeTypeTuple | None = None

    max_walk_length: int = DEFAULT_PIPELINE_MAX_WALK_LENGTH
    min_walk_length: int = DEFAULT_MIN_WALK_LENGTH
    max_walk_attempts: int = DEFAULT_PIPELINE_MAX_WALK_ATTEMPTS
    walk_strategy: WalkStrategy = WalkStrategy.UNIFORM
    edge_weighting: EdgeWeighting = EdgeWeighting.UNIFORM

    min_support: int = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_head_coverage: float = DEFAULT_MIN_HEAD_COVERAGE
    per_type_thresholds: Mapping[RuleType, RuleThresholds] = field(default_factory=dict)

    acyclic_samples_per_end: int = NO_ACYCLIC_SAMPLES

    seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.acyclic_samples_per_end < 0:
            raise ConfigurationError(
                "acyclic_samples_per_end must be non-negative, "
                f"got {self.acyclic_samples_per_end}"
            )

    def sampler_config(self, *, seed_offset: int = 0) -> SamplerConfig:
        """Return the sampler configuration, seeded with ``seed + seed_offset``."""
        return SamplerConfig(
            sample_size=self.sample_size,
            strategy=self.sampling_strategy,
            seed=self.seed + seed_offset,
            target_edge_type=self.target_edge_type,
        )

    def walk_config(self, *, min_length: int, max_length: int) -> WalkConfig:
        """Return the walk configuration for the given length bounds."""
        return WalkConfig(
            max_length=max_length,
            min_length=min_length,
            max_attempts=self.max_walk_attempts,
            strategy=self.walk_strategy,
            edge_weighting=self.edge_weighting,
            seed=self.seed,
        )

    def rule_config(self) -> RuleConfig:
        """Return the rule quality configuration."""
        return RuleConfig(
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            min_head_coverage=self.min_head_coverage,
            per_type=self.per_type_thresholds,
        )


class AnyBURL:
    """End-to-end AnyBURL rule learning pipeline.

    Parameters
    ----------
    config : AnyBURLConfig
        Pipeline configuration controlling all stages.

    Attributes
    ----------
    graph : HeteroGraph | None
        The wrapped graph, set by ``fit``.
    triples : list[Triple]
        Sampled target triples.
    paths : list[tuple[list[PathStep], Triple]]
        Walk paths paired with their source triples.
    acyclic_steps : list[AcyclicStep]
        One-step walks generalized into acyclic rules.
    rules : list[Rule]
        Deduplicated candidate rules from generalization.
    results : RuleSet
        Rules that passed quality thresholds, with their metrics.
    report : AnytimeReport | None
        How the budget was spent, set by ``fit_anytime``.
    """

    def __init__(self, config: AnyBURLConfig) -> None:
        self.config = config

        self.graph: HeteroGraph | None = None
        self.triples: list[Triple] = []
        self.paths: list[tuple[list[PathStep], Triple]] = []
        self.acyclic_steps: list[AcyclicStep] = []
        self.rules: list[Rule] = []
        self.results: RuleSet = RuleSet.of(())
        self.report: AnytimeReport | None = None

    def fit(self, data: HeteroData | HeteroGraph) -> Self:
        """Sample triples, walk, generalize and evaluate, in that order.

        Parameters
        ----------
        data : HeteroData | HeteroGraph
            The graph to mine. Pass a :class:`HeteroGraph` to share its
            indices across several fits on the same graph.

        Returns
        -------
        AnyBURL
            ``self``, for method chaining.
        """
        self.graph = _as_graph(data)
        self.triples = build_triple_sampler(
            self.graph, self.config.sampler_config()
        ).sample()
        self._walk(self.graph)
        self._sample_acyclic(self.graph)
        self._generalize()
        evaluator = RuleEvaluator(self.graph, self.config.rule_config())
        self.results = evaluator.evaluate_batch(self.rules)
        return self

    def fit_anytime(
        self,
        data: HeteroData | HeteroGraph,
        anytime_config: AnytimeConfig | None = None,
    ) -> Self:
        """Mine rules by increasing length within a wall-clock budget.

        ``sample_size`` and ``max_walk_attempts`` size a single batch rather
        than the whole run. :attr:`report` records how the budget was spent.

        Parameters
        ----------
        data : HeteroData | HeteroGraph
            The graph to mine.
        anytime_config : AnytimeConfig | None
            Budget, rule lengths and stopping rules. ``None`` uses the
            default budget and this pipeline's walk lengths.

        Returns
        -------
        AnyBURL
            ``self``, for method chaining.

        Raises
        ------
        ConfigurationError
            If ``acyclic_samples_per_end`` is set; the anytime loop does not
            mine acyclic rules.
        """
        cfg = self.config
        if cfg.acyclic_samples_per_end != NO_ACYCLIC_SAMPLES:
            raise ConfigurationError(
                "fit_anytime does not mine acyclic rules; "
                "set acyclic_samples_per_end=0 or use fit"
            )
        if anytime_config is None:
            anytime_config = AnytimeConfig(
                min_length=cfg.min_walk_length, max_length=cfg.max_walk_length
            )

        self.graph = _as_graph(data)
        learner = AnytimeLearner(self._mining_stages(self.graph), anytime_config)
        self.results, self.report = learner.learn()
        self.rules = list(learner.candidates)
        return self

    def predict(
        self,
        *,
        filter_known: bool = False,
        prediction: PredictionConfig | None = None,
    ) -> list[Prediction]:
        """Score every candidate pair with the learned rules.

        Parameters
        ----------
        filter_known : bool
            If ``True``, exclude predictions already present in the graph.
        prediction : PredictionConfig | None
            Scoring and grounding settings. ``None`` uses the defaults.

        Returns
        -------
        list[Prediction]
            Predictions of every relation with rules, sorted by score
            descending.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called or produced no rules.
        """
        return self.predictor(prediction).predict(filter_known=filter_known)

    def evaluate_predictions(
        self,
        test_triples: Sequence[Triple],
        *,
        evaluation: EvaluationConfig | None = None,
        prediction: PredictionConfig | None = None,
        known: HeteroData | HeteroGraph | None = None,
    ) -> LinkPredictionMetrics:
        """Evaluate link prediction quality on test triples.

        Each relation is ranked with its own rules; a relation without rules
        scores every candidate zero.

        Parameters
        ----------
        test_triples : Sequence[Triple]
            Test triples to evaluate, of any number of relations.
        evaluation : EvaluationConfig | None
            Hits@K thresholds, filtering and tie handling. ``None`` uses the
            defaults.
        prediction : PredictionConfig | None
            Scoring and grounding settings. ``None`` uses the defaults.
        known : HeteroData | HeteroGraph | None
            The known triples to filter. The standard filtered setting uses
            every known edge, including validation and test edges. ``None``
            filters only the fitted graph's edges.

        Returns
        -------
        LinkPredictionMetrics
            Aggregated MRR and Hits@K metrics.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called or produced no rules.
        """
        predictor = self.predictor(prediction)
        config = evaluation if evaluation is not None else EvaluationConfig()
        known_graph = self._require_graph() if known is None else _as_graph(known)
        ranks = rank_queries_by_edge_type(
            predictor.scorer_for, known_graph, test_triples, config
        )
        return summarize_ranks([query.rank for query in ranks], config.k_values)

    def predictor(
        self, prediction: PredictionConfig | None = None
    ) -> MultiRelationPredictor:
        """Return a predictor over the learned rules of every relation.

        Parameters
        ----------
        prediction : PredictionConfig | None
            Scoring and grounding settings. ``None`` uses the defaults.

        Returns
        -------
        MultiRelationPredictor
            Scores queries of any relation against the fitted graph.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called or produced no rules.
        """
        return MultiRelationPredictor(
            self._require_rules(), self.results, config=prediction
        )

    def fit_aggregator(
        self,
        train_triples: Sequence[Triple],
        *,
        edge_type: EdgeTypeTuple | None = None,
        config: AggregationSetup | None = None,
    ) -> AggregatedScorer:
        """Fit a learned aggregator over one relation's rules.

        Parameters
        ----------
        train_triples : Sequence[Triple]
            Triples of one relation held out of the fitted graph, such as
            validation triples.
        edge_type : EdgeTypeTuple | None
            The relation to fit. ``None`` takes it from ``train_triples``.
        config : AggregationSetup | None
            Featurizer, sampling and model settings. ``None`` uses the
            defaults.

        Returns
        -------
        AggregatedScorer
            Ranks queries of the relation against the fitted graph.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called or produced no rules.
        ConfigurationError
            As raised by :func:`~anyburl.aggregation.fit_aggregated_scorer`.
        """
        return fit_aggregated_scorer(
            self._require_rules(),
            self.results,
            train_triples,
            edge_type=edge_type,
            setup=config,
        )

    def _require_rules(self) -> HeteroGraph:
        """Return the graph, raising unless ``fit`` produced rules."""
        graph = self._require_graph()
        if not self.results:
            raise NotFittedError(
                "No rules found. Call fit(data) first and ensure rules pass thresholds."
            )
        return graph

    def _require_graph(self) -> HeteroGraph:
        """Return the graph, raising if ``fit`` has not been called."""
        if self.graph is None:
            raise NotFittedError("Pipeline must be fitted first. Call fit(data).")
        return self.graph

    def _mining_stages(self, graph: HeteroGraph) -> MiningStages:
        """Wire the per-batch collaborators the anytime loop drives."""
        cfg = self.config

        def sample_batch(batch_index: int) -> Sequence[Triple]:
            sampler_config = cfg.sampler_config(seed_offset=batch_index)
            return build_triple_sampler(graph, sampler_config).sample()

        def walker_for_length(length: int) -> PathWalker:
            walk_config = cfg.walk_config(min_length=length, max_length=length)
            return NumbaWalkEngine(graph, walk_config)

        return MiningStages(
            sample_batch=sample_batch,
            walker_for_length=walker_for_length,
            generalizer=RuleGeneralizer(),
            evaluator=RuleEvaluator(graph, cfg.rule_config()),
        )

    def _walk(self, graph: HeteroGraph) -> None:
        """Run random walks from each sampled triple."""
        walk_config = self.config.walk_config(
            min_length=self.config.min_walk_length,
            max_length=self.config.max_walk_length,
        )
        walker = NumbaWalkEngine(graph, walk_config)
        self.paths = [
            (path, triple)
            for triple in progress(self.triples, desc="Building paths")
            for path in walker.walk_from_triple(triple)
        ]
        logger.info(
            "Total triples: %d | Total paths: %d", len(self.triples), len(self.paths)
        )

    def _sample_acyclic(self, graph: HeteroGraph) -> None:
        """Draw one-step walks from both endpoints of each sampled triple."""
        samples = self.config.acyclic_samples_per_end
        if samples == NO_ACYCLIC_SAMPLES:
            return
        sampler = AcyclicStepSampler(graph, self.config.seed)
        self.acyclic_steps = [
            step for triple in self.triples for step in sampler.sample(triple, samples)
        ]
        logger.info("Total acyclic steps: %d", len(self.acyclic_steps))

    def _generalize(self) -> None:
        """Generalize walk paths into deduplicated Horn rules."""
        generalizer = RuleGeneralizer()
        unique: dict[Rule, None] = {}
        for step in self.acyclic_steps:
            for rule in generalize_acyclic(step):
                unique.setdefault(rule, None)
        for path, triple in progress(self.paths, desc="Generalizing rules"):
            for rule in generalizer.generalize(
                path,
                target_relation=triple.relation,
                head_type=triple.head_type,
                tail_type=triple.tail_type,
            ):
                unique.setdefault(rule, None)
        self.rules = list(unique)


def _as_graph(data: HeteroData | HeteroGraph) -> HeteroGraph:
    """Return ``data`` as a :class:`HeteroGraph`, wrapping it if needed."""
    match data:
        case HeteroGraph():
            return data
        case _:
            return HeteroGraph(data)
