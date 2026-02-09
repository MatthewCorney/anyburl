"""End-to-end AnyBURL pipeline: sample, walk, generalize, evaluate."""
from dataclasses import dataclass

from .graph import HeteroGraph
from .metrics import RuleEvaluator, RuleMetrics
from .prediction import Prediction, RulePredictor
from .rule import Rule, RuleConfig, RuleGeneralizer
from .sampler import SamplerConfig, SamplingStrategy, Triple, UniformTripleSampler, WeightedTripleSampler, \
    BaseTripleSampler
from .walk import WalkConfig, WalkEngine, WalkStrategy, RelationWeightedEdgeSelector, UniformEdgeSelector
from torch_geometric.data import HeteroData
import torch
from .rule import PathStep
from tqdm import tqdm
from ._logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnyBURLConfig:
    """Configuration for the AnyBURL pipeline.

    Parameters
    ----------
    sample_size : int
        Number of triples to sample per iteration.
    sampling_strategy : SamplingStrategy
        How to weight triple selection.
    target_edge_type : tuple[str, str, str] | None
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
    seed : int
        Random seed for reproducibility.
    """

    sample_size: int = 2000
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    target_edge_type: tuple[str, str, str] | None = None

    max_walk_length: int = 4
    min_walk_length: int = 2
    max_walk_attempts: int = 700
    walk_strategy: WalkStrategy = WalkStrategy.UNIFORM

    min_support: int = 2
    min_confidence: float = 0.01
    min_head_coverage: float = 0.01

    seed: int = 42


def build_triple_sampler(
        graph: HeteroGraph,
        config: SamplerConfig,
) -> BaseTripleSampler:
    if config.strategy == SamplingStrategy.UNIFORM:
        return UniformTripleSampler(graph, config)

    edge_counts = torch.tensor(
        [graph.edge_count(et) for et in graph.edge_types if graph.edge_count(et) > 0],
        dtype=torch.float32,
    )

    if config.strategy == SamplingStrategy.RELATION_PROPORTIONAL:
        weights = torch.ones_like(edge_counts)
    elif config.strategy == SamplingStrategy.RELATION_INVERSE:
        weights = 1.0 / edge_counts
        weights = weights / weights.sum()
    else:
        raise ValueError('Pass')

    return WeightedTripleSampler(graph, config, weights)


def build_walk_engine(
        graph: HeteroGraph,
        config: WalkConfig,
) -> WalkEngine:
    """Factory for constructing a WalkEngine with the correct strategy."""
    generator = torch.Generator().manual_seed(config.seed)

    if config.strategy is WalkStrategy.UNIFORM:
        selector = UniformEdgeSelector(generator)

    elif config.strategy is WalkStrategy.RELATION_WEIGHTED:
        selector = RelationWeightedEdgeSelector(graph, generator)

    else:
        raise ValueError(f"Unsupported walk strategy: {config.strategy}")

    return WalkEngine(graph, config, selector)


class AnyBURL:
    """End-to-end AnyBURL rule learning pipeline.

    Wires together all four stages of the AnyBURL algorithm: sampling
    target triples, random walks, rule generalization, and evaluation.

    Parameters
    ----------
    config : AnyBURLConfig
        Pipeline configuration controlling all stages.

    Attributes
    ----------
    graph : HeteroGraph | None
        The wrapped graph, set after ``fit``.
    triples : list[Triple]
        Sampled target triples.
    paths : list[tuple[list[PathStep], Triple]]
        Walk paths paired with their source triples.
    rules : list[Rule]
        Deduplicated candidate rules from generalization.
    results : list[tuple[Rule, RuleMetrics]]
        Rules that passed quality thresholds, with their metrics.

    Examples
    --------
    >>> config = AnyBURLConfig(sample_size=500, seed=0)
    >>> pipeline = AnyBURL(config).fit(data)
    >>> len(pipeline.results)
    42
    """

    def __init__(self, config: AnyBURLConfig) -> None:
        self.config = config

        self.graph: HeteroGraph | None = None
        self.triples: list[Triple] = []
        self.paths: list[tuple[list[PathStep], Triple]] = []
        self.rules: list[Rule] = []
        self.results: list[tuple[Rule, RuleMetrics]] = []

    def fit(self, data: HeteroData) -> 'AnyBURL':
        """Run the full AnyBURL learning pipeline.

        Executes all four stages in order: sample triples, walk,
        generalize, and evaluate. Results are stored on the instance.

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

        self._sample_triples()
        self._walk()
        self._generalize()
        self._evaluate()

        return self

    def predict(self, *, filter_known: bool = False) -> list[Prediction]:
        """Generate predictions by grounding learned rules against the graph.

        Each rule is grounded via sparse matrix multiplication to find
        ``(head, tail)`` pairs, then confidence scores are aggregated
        across all rules that predict the same pair.

        Parameters
        ----------
        filter_known : bool
            If ``True``, exclude predictions that correspond to edges
            already present in the graph.

        Returns
        -------
        list[Prediction]
            Predictions sorted by aggregated confidence descending.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called yet or produced no results.
        """
        graph = self._require_graph()
        if not self.results:
            msg = (
                "No rules found. Call fit(data) first "
                "and ensure rules pass thresholds."
            )
            raise RuntimeError(msg)
        predictor = RulePredictor(graph)
        return predictor.predict(self.results, filter_known=filter_known)

    def _require_graph(self) -> HeteroGraph:
        """Return the graph, raising if ``fit`` has not been called."""
        if self.graph is None:
            msg = "Pipeline must be fitted first. Call fit(data)."
            raise RuntimeError(msg)
        return self.graph

    def _sample_triples(self) -> None:
        """Sample target triples from the graph."""
        graph = self._require_graph()
        cfg = self.config

        sampler_config = SamplerConfig(
            sample_size=cfg.sample_size,
            strategy=cfg.sampling_strategy,
            seed=cfg.seed,
            target_edge_type=cfg.target_edge_type,
        )
        sampler = build_triple_sampler(graph, sampler_config)
        self.triples = sampler.sample()

    def _walk(self) -> None:
        """Run random walks from each sampled triple."""
        graph = self._require_graph()
        cfg = self.config

        walk_config = WalkConfig(
            max_length=cfg.max_walk_length,
            min_length=cfg.min_walk_length,
            max_attempts=cfg.max_walk_attempts,
            strategy=cfg.walk_strategy,
            seed=cfg.seed,
        )
        walker = build_walk_engine(graph, walk_config)

        self.paths = []
        total_paths = 0
        for triple in tqdm(self.triples, desc="Building paths"):
            paths = walker.walk_from_triple(triple)

            n_paths = len(paths)
            total_paths += n_paths

            for path in paths:
                self.paths.append((path, triple))

        logger.info("Finished. Total triples: %d | Total paths: %d",
                    len(self.triples),
                    total_paths)

    def _generalize(self) -> None:
        """Generalize walk paths into typed Horn rules."""
        cfg = self.config

        rule_config = RuleConfig(
            min_support=cfg.min_support,
            min_confidence=cfg.min_confidence,
            min_head_coverage=cfg.min_head_coverage,
        )

        generalizer = RuleGeneralizer(rule_config)

        seen: set[str] = set()
        self.rules = []

        for path, triple in tqdm(self.paths,desc="Generalizing Rules"):
            try:
                rules = generalizer.generalize(
                    path,
                    target_relation=triple.relation,
                    head_type=triple.head_type,
                    tail_type=triple.tail_type,
                )
            except ValueError:
                continue

            for rule in rules:
                key = str(rule)
                if key not in seen:
                    seen.add(key)
                    self.rules.append(rule)

    def _evaluate(self) -> None:
        """Evaluate candidate rules against quality thresholds."""
        graph = self._require_graph()
        cfg = self.config

        evaluator = RuleEvaluator(
            graph,
            RuleConfig(
                min_support=cfg.min_support,
                min_confidence=cfg.min_confidence,
                min_head_coverage=cfg.min_head_coverage,
            ),
        )

        self.results = evaluator.evaluate_batch(self.rules)
