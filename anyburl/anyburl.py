"""End-to-end AnyBURL pipeline: sample, walk, generalize, evaluate."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self, assert_never

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from ._logging import get_logger
from .anytime import (
    AnytimeConfig,
    AnytimeLearner,
    AnytimeReport,
    MiningStages,
    PathWalker,
)
from .evaluation import (
    DEFAULT_K_VALUES,
    EvaluationConfig,
    LinkPredictionEvaluator,
    LinkPredictionMetrics,
    TieHandling,
)
from .exceptions import NotFittedError
from .graph import EdgeTypeTuple, HeteroGraph
from .metrics import RuleEvaluator, RuleMetrics
from .prediction import Prediction, PredictionConfig, RulePredictor
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
from .sampler import (
    BaseTripleSampler,
    EntityBalancedTripleSampler,
    SamplerConfig,
    SamplingStrategy,
    Triple,
    UniformTripleSampler,
    WeightedTripleSampler,
)
from .walk import (
    NumbaWalkEngine,
    RelationWeightedEdgeSelector,
    WalkConfig,
    WalkEngine,
    WalkStrategy,
)
from .walk.base import DEFAULT_MIN_WALK_LENGTH, DEFAULT_RANDOM_SEED

logger = get_logger(__name__)

DEFAULT_PIPELINE_SAMPLE_SIZE: int = 2000
DEFAULT_PIPELINE_MAX_WALK_LENGTH: int = 4
DEFAULT_PIPELINE_MAX_WALK_ATTEMPTS: int = 700


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
        Strategy for selecting the next step during walks.
    min_support : int
        Minimum support threshold for rule filtering.
    min_confidence : float
        Minimum confidence threshold for rule filtering.
    min_head_coverage : float
        Minimum head coverage threshold for rule filtering.
    per_type_thresholds : Mapping[RuleType, RuleThresholds]
        Floors replacing the defaults for the named rule types.
    seed : int
        Random seed for reproducibility.
    """

    sample_size: int = DEFAULT_PIPELINE_SAMPLE_SIZE
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    target_edge_type: EdgeTypeTuple | None = None

    max_walk_length: int = DEFAULT_PIPELINE_MAX_WALK_LENGTH
    min_walk_length: int = DEFAULT_MIN_WALK_LENGTH
    max_walk_attempts: int = DEFAULT_PIPELINE_MAX_WALK_ATTEMPTS
    walk_strategy: WalkStrategy = WalkStrategy.UNIFORM

    min_support: int = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_head_coverage: float = DEFAULT_MIN_HEAD_COVERAGE
    per_type_thresholds: Mapping[RuleType, RuleThresholds] = field(default_factory=dict)

    seed: int = DEFAULT_RANDOM_SEED

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


def build_triple_sampler(
    graph: HeteroGraph,
    config: SamplerConfig,
) -> BaseTripleSampler:
    """Build a triple sampler for the configured strategy.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    config : SamplerConfig
        Sampler configuration.

    Returns
    -------
    BaseTripleSampler
        A sampler instance.
    """
    match config.strategy:
        case SamplingStrategy.UNIFORM:
            return UniformTripleSampler(graph, config)
        case SamplingStrategy.ENTITY_BALANCED:
            return EntityBalancedTripleSampler(graph, config)
        case SamplingStrategy.RELATION_PROPORTIONAL:
            weights = torch.ones(len(_non_empty_edge_counts(graph)))
            return WeightedTripleSampler(graph, config, weights)
        case SamplingStrategy.RELATION_INVERSE:
            inverse = 1.0 / _non_empty_edge_counts(graph)
            return WeightedTripleSampler(graph, config, inverse / inverse.sum())
        case _ as unreachable:
            assert_never(unreachable)


def _non_empty_edge_counts(graph: HeteroGraph) -> torch.Tensor:
    """Return the edge count of every edge type that has edges, in graph order."""
    return torch.tensor(
        [graph.edge_count(et) for et in graph.edge_types if graph.edge_count(et) > 0],
        dtype=torch.float32,
    )


def build_walk_engine(
    graph: HeteroGraph,
    config: WalkConfig,
) -> WalkEngine | NumbaWalkEngine:
    """Build a walk engine for the configured strategy.

    Uniform and reachability-pruned walks use :class:`NumbaWalkEngine`;
    relation-weighted walks use :class:`WalkEngine`.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    config : WalkConfig
        Walk configuration.

    Returns
    -------
    WalkEngine | NumbaWalkEngine
        A walk engine instance.
    """
    match config.strategy:
        case WalkStrategy.UNIFORM | WalkStrategy.REACHABILITY_PRUNED:
            return NumbaWalkEngine(graph, config)
        case WalkStrategy.RELATION_WEIGHTED:
            generator = torch.Generator().manual_seed(config.seed)
            selector = RelationWeightedEdgeSelector(graph, generator)
            return WalkEngine(graph, config, selector)
        case _ as unreachable:
            assert_never(unreachable)


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
    rules : list[Rule]
        Deduplicated candidate rules from generalization.
    results : list[tuple[Rule, RuleMetrics]]
        Rules that passed quality thresholds, with their metrics.
    report : AnytimeReport | None
        How the budget was spent, set by ``fit_anytime``.
    """

    def __init__(self, config: AnyBURLConfig) -> None:
        self.config = config

        self.graph: HeteroGraph | None = None
        self.triples: list[Triple] = []
        self.paths: list[tuple[list[PathStep], Triple]] = []
        self.rules: list[Rule] = []
        self.results: list[tuple[Rule, RuleMetrics]] = []
        self.report: AnytimeReport | None = None

    def fit(self, data: HeteroData) -> Self:
        """Sample triples, walk, generalize and evaluate, in that order.

        Parameters
        ----------
        data : HeteroData
            A PyTorch Geometric heterogeneous graph.

        Returns
        -------
        AnyBURL
            ``self``, for method chaining.
        """
        self.graph = HeteroGraph(data)
        self.triples = build_triple_sampler(
            self.graph, self.config.sampler_config()
        ).sample()
        self._walk(self.graph)
        self._generalize()
        evaluator = RuleEvaluator(self.graph, self.config.rule_config())
        self.results = evaluator.evaluate_batch(self.rules)
        return self

    def fit_anytime(
        self, data: HeteroData, anytime_config: AnytimeConfig | None = None
    ) -> Self:
        """Mine rules by increasing length within a wall-clock budget.

        ``sample_size`` and ``max_walk_attempts`` size a single batch rather
        than the whole run. :attr:`report` records how the budget was spent.

        Parameters
        ----------
        data : HeteroData
            A PyTorch Geometric heterogeneous graph.
        anytime_config : AnytimeConfig | None
            Budget and stopping rules. ``None`` uses this pipeline's batch
            size, walk lengths and seed.

        Returns
        -------
        AnyBURL
            ``self``, for method chaining.
        """
        cfg = self.config
        if anytime_config is None:
            anytime_config = AnytimeConfig(
                batch_size=cfg.sample_size,
                min_length=cfg.min_walk_length,
                max_length=cfg.max_walk_length,
                seed=cfg.seed,
            )

        self.graph = HeteroGraph(data)
        learner = AnytimeLearner(self._mining_stages(self.graph), anytime_config)
        self.results, self.report = learner.learn()
        self.rules = [rule for rule, _ in self.results]
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
            Predictions sorted by score descending.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called or produced no rules.
        """
        return self._build_predictor(prediction).predict(filter_known=filter_known)

    def evaluate_predictions(
        self,
        test_triples: Sequence[Triple],
        *,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
        filter_known: bool = True,
        tie_handling: TieHandling = TieHandling.AVERAGE,
        prediction: PredictionConfig | None = None,
    ) -> LinkPredictionMetrics:
        """Evaluate link prediction quality on test triples.

        Parameters
        ----------
        test_triples : Sequence[Triple]
            Test triples to evaluate.
        k_values : tuple[int, ...]
            Hits@K thresholds.
        filter_known : bool
            If ``True``, filter known triples when computing ranks.
        tie_handling : TieHandling
            How to rank a target tied with other candidates.
        prediction : PredictionConfig | None
            Scoring and grounding settings. ``None`` uses the defaults.

        Returns
        -------
        LinkPredictionMetrics
            Aggregated MRR and Hits@K metrics.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called or produced no rules.
        """
        predictor = self._build_predictor(prediction)
        config = EvaluationConfig(
            k_values=k_values,
            filter_known=filter_known,
            tie_handling=tie_handling,
        )
        evaluator = LinkPredictionEvaluator(predictor, self._require_graph(), config)
        return evaluator.evaluate(test_triples)

    def _require_graph(self) -> HeteroGraph:
        """Return the graph, raising if ``fit`` has not been called."""
        if self.graph is None:
            raise NotFittedError("Pipeline must be fitted first. Call fit(data).")
        return self.graph

    def _build_predictor(self, prediction: PredictionConfig | None) -> RulePredictor:
        """Return a predictor over the learned rules."""
        graph = self._require_graph()
        if not self.results:
            raise NotFittedError(
                "No rules found. Call fit(data) first and ensure rules pass thresholds."
            )
        settings = prediction if prediction is not None else PredictionConfig()
        return RulePredictor(
            graph,
            self.results,
            scoring_strategy=settings.scoring_strategy,
            grounding_mode=settings.grounding_mode,
        )

    def _mining_stages(self, graph: HeteroGraph) -> MiningStages:
        """Wire the per-batch collaborators the anytime loop drives."""
        cfg = self.config

        def sample_batch(batch_index: int) -> Sequence[Triple]:
            sampler_config = cfg.sampler_config(seed_offset=batch_index)
            return build_triple_sampler(graph, sampler_config).sample()

        def walker_for_length(length: int) -> PathWalker:
            walk_config = cfg.walk_config(min_length=length, max_length=length)
            return build_walk_engine(graph, walk_config)

        return MiningStages(
            sample_batch=sample_batch,
            walker_for_length=walker_for_length,
            generalizer=RuleGeneralizer(cfg.rule_config()),
            evaluator=RuleEvaluator(graph, cfg.rule_config()),
        )

    def _walk(self, graph: HeteroGraph) -> None:
        """Run random walks from each sampled triple."""
        walk_config = self.config.walk_config(
            min_length=self.config.min_walk_length,
            max_length=self.config.max_walk_length,
        )
        walker = build_walk_engine(graph, walk_config)
        self.paths = [
            (path, triple)
            for triple in tqdm(self.triples, desc="Building paths")
            for path in walker.walk_from_triple(triple)
        ]
        logger.info(
            "Total triples: %d | Total paths: %d", len(self.triples), len(self.paths)
        )

    def _generalize(self) -> None:
        """Generalize walk paths into deduplicated Horn rules."""
        generalizer = RuleGeneralizer(self.config.rule_config())
        unique: dict[str, Rule] = {}
        for path, triple in tqdm(self.paths, desc="Generalizing rules"):
            for rule in generalizer.generalize(
                path,
                target_relation=triple.relation,
                head_type=triple.head_type,
                tail_type=triple.tail_type,
            ):
                unique.setdefault(str(rule), rule)
        self.rules = list(unique.values())
