"""AnyBURL: Anytime Bottom-Up Rule Learning for knowledge graphs.

This package implements the AnyBURL algorithm for learning first-order
Horn rules from heterogeneous knowledge graphs backed by PyTorch Geometric.

Algorithm Pipeline
------------------
The learning process follows four stages:

1. **Sample** --- draw target triples according to a
   :class:`SamplingStrategy`.
2. **Walk** --- run bounded random walks from each triple's head entity,
   keeping paths that reach its tail.
3. **Generalize** --- :class:`RuleGeneralizer` replaces the entities in each
   path with variables, producing cyclic and AC1 :class:`Rule` objects.
4. **Evaluate** --- :class:`RuleEvaluator` computes support, confidence and
   head coverage (:class:`RuleMetrics`) by counting body-chain groundings,
   and keeps the rules passing their thresholds in a :class:`RuleSet`.

Learned rules are grounded into :class:`RuleEvidence`, combined into scores by
:class:`RulePredictor` or a learned :class:`AggregatedScorer`, and scored with
:class:`LinkPredictionEvaluator`; :func:`stratify_by_degree` breaks its
per-query ranks down by how well-connected each target is.

Everything exported here is the public API. Subpackage modules prefixed with
an underscore, and the :mod:`anyburl.chain` grounding engine, are internal.

References
----------
.. [1] Meilicke, C., Chekol, M. W., Ruffinelli, D., & Stuckenschmidt, H.
   (2019). Anytime Bottom-Up Rule Learning for Knowledge Graph Completion.
   *IJCAI*.
"""

from .aggregation import (
    AggregatedScorer,
    AggregationSetup,
    AggregatorConfig,
    CandidateFeaturizer,
    CandidateSampling,
    CandidateScope,
    FeaturizerKind,
    JointFeaturizer,
    KnownAnswers,
    LinearAggregator,
    ResidualAggregator,
    SharedFeaturizer,
    build_training_set,
    concat_training_sets,
    fit_aggregated_scorer,
    ranking_queries,
)
from .anytime import (
    AnytimeConfig,
    AnytimeLearner,
    AnytimeReport,
    LengthReport,
)
from .baselines import (
    MetaPathScorer,
    PopularityScorer,
    RandomScorer,
    ZeroScorer,
)
from .core import (
    BodyChain,
    EdgeTypeTuple,
    EntityScorer,
    QueryDirection,
    Triple,
)
from .evaluation import (
    EvaluationConfig,
    LinkPredictionEvaluator,
    LinkPredictionMetrics,
    QueryRank,
    TieHandling,
    rank_queries_by_edge_type,
    rank_queries_with_each,
    summarize_ranks,
)
from .exceptions import (
    AnyBURLError,
    ConfigurationError,
    GraphSchemaError,
    InvalidRuleError,
    NotFittedError,
)
from .graph import HeteroGraph
from .pipeline import (
    AnyBURL,
    AnyBURLConfig,
)
from .prediction import (
    ConfidenceAggregation,
    GroundingWeighting,
    MultiRelationPredictor,
    Prediction,
    PredictionConfig,
    RuleEvidence,
    RuleFiring,
    RulePredictor,
)
from .rule import (
    Atom,
    PathStep,
    Rule,
    RuleConfig,
    RuleGeneralizer,
    RuleMetrics,
    RuleThresholds,
    RuleType,
    Term,
    TermKind,
    aggregate_confidence,
)
from .rule_evaluation import (
    RuleEvaluator,
    RuleSet,
)
from .sampler import (
    BaseTripleSampler,
    EntityBalancedTripleSampler,
    SamplerConfig,
    SamplingStrategy,
    UniformTripleSampler,
    WeightedTripleSampler,
    build_triple_sampler,
)
from .split import (
    InverseEdgeHandling,
    SplitConfig,
    TripleSplit,
    ValidationSplit,
    split_target_edges,
    split_with_validation,
)
from .stratification import (
    DegreeBucket,
    RelationDegrees,
    StratificationConfig,
    relation_degrees,
    stratify_by_degree,
)
from .walk import (
    EdgeWeighting,
    NumbaWalkEngine,
    WalkConfig,
    WalkStrategy,
)

__all__ = [
    "AggregatedScorer",
    "AggregationSetup",
    "AggregatorConfig",
    "AnyBURL",
    "AnyBURLConfig",
    "AnyBURLError",
    "AnytimeConfig",
    "AnytimeLearner",
    "AnytimeReport",
    "Atom",
    "BaseTripleSampler",
    "BodyChain",
    "CandidateFeaturizer",
    "CandidateSampling",
    "CandidateScope",
    "ConfidenceAggregation",
    "ConfigurationError",
    "DegreeBucket",
    "EdgeTypeTuple",
    "EdgeWeighting",
    "EntityBalancedTripleSampler",
    "EntityScorer",
    "EvaluationConfig",
    "FeaturizerKind",
    "GraphSchemaError",
    "GroundingWeighting",
    "HeteroGraph",
    "InvalidRuleError",
    "InverseEdgeHandling",
    "JointFeaturizer",
    "KnownAnswers",
    "LengthReport",
    "LinearAggregator",
    "LinkPredictionEvaluator",
    "LinkPredictionMetrics",
    "MetaPathScorer",
    "MultiRelationPredictor",
    "NotFittedError",
    "NumbaWalkEngine",
    "PathStep",
    "PopularityScorer",
    "Prediction",
    "PredictionConfig",
    "QueryDirection",
    "QueryRank",
    "RandomScorer",
    "RelationDegrees",
    "ResidualAggregator",
    "Rule",
    "RuleConfig",
    "RuleEvaluator",
    "RuleEvidence",
    "RuleFiring",
    "RuleGeneralizer",
    "RuleMetrics",
    "RulePredictor",
    "RuleSet",
    "RuleThresholds",
    "RuleType",
    "SamplerConfig",
    "SamplingStrategy",
    "SharedFeaturizer",
    "SplitConfig",
    "StratificationConfig",
    "Term",
    "TermKind",
    "TieHandling",
    "Triple",
    "TripleSplit",
    "UniformTripleSampler",
    "ValidationSplit",
    "WalkConfig",
    "WalkStrategy",
    "WeightedTripleSampler",
    "ZeroScorer",
    "aggregate_confidence",
    "build_training_set",
    "build_triple_sampler",
    "concat_training_sets",
    "fit_aggregated_scorer",
    "rank_queries_by_edge_type",
    "rank_queries_with_each",
    "ranking_queries",
    "relation_degrees",
    "split_target_edges",
    "split_with_validation",
    "stratify_by_degree",
    "summarize_ranks",
]
