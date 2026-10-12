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
   path with variables, producing cyclic, AC1 and AC2 :class:`Rule` objects.
4. **Evaluate** --- :class:`RuleEvaluator` computes support, confidence and
   head coverage (:class:`RuleMetrics`) by counting body-chain groundings.

Learned rules are applied by :class:`RulePredictor` and scored with
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
)
from .evaluation import (
    EntityScorer,
    EvaluationConfig,
    LinkPredictionEvaluator,
    LinkPredictionMetrics,
    QueryDirection,
    QueryRank,
    TieHandling,
    summarize_ranks,
)
from .exceptions import (
    AnyBURLError,
    ConfigurationError,
    GraphSchemaError,
    InvalidRuleError,
    NotFittedError,
)
from .factories import (
    build_triple_sampler,
    build_walk_engine,
)
from .graph import (
    EdgeTypeTuple,
    HeteroGraph,
)
from .metrics import (
    RuleEvaluator,
    RuleMetrics,
    aggregate_confidence,
)
from .pipeline import (
    AnyBURL,
    AnyBURLConfig,
)
from .prediction import (
    ConfidenceAggregation,
    GroundingMode,
    Prediction,
    PredictionConfig,
    RuleFiring,
    RulePredictor,
    ScoringStrategy,
)
from .rule import (
    Atom,
    PathStep,
    Rule,
    RuleConfig,
    RuleGeneralizer,
    RuleThresholds,
    RuleType,
    Term,
    TermKind,
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
from .split import (
    InverseEdgeHandling,
    SplitConfig,
    TripleSplit,
    split_target_edges,
)
from .stratification import (
    DegreeBucket,
    RelationDegrees,
    StratificationConfig,
    relation_degrees,
    stratify_by_degree,
)
from .validation_split import ValidationSplit, split_with_validation
from .walk import (
    EdgeWeighting,
    NumbaWalkEngine,
    WalkConfig,
    WalkEngine,
    WalkStrategy,
)

__all__ = [
    "AnyBURL",
    "AnyBURLConfig",
    "AnyBURLError",
    "AnytimeConfig",
    "AnytimeLearner",
    "AnytimeReport",
    "Atom",
    "BaseTripleSampler",
    "ConfidenceAggregation",
    "ConfigurationError",
    "DegreeBucket",
    "EdgeTypeTuple",
    "EdgeWeighting",
    "EntityBalancedTripleSampler",
    "EntityScorer",
    "EvaluationConfig",
    "GraphSchemaError",
    "GroundingMode",
    "HeteroGraph",
    "InvalidRuleError",
    "InverseEdgeHandling",
    "LengthReport",
    "LinkPredictionEvaluator",
    "LinkPredictionMetrics",
    "MetaPathScorer",
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
    "Rule",
    "RuleConfig",
    "RuleEvaluator",
    "RuleFiring",
    "RuleGeneralizer",
    "RuleMetrics",
    "RulePredictor",
    "RuleThresholds",
    "RuleType",
    "SamplerConfig",
    "SamplingStrategy",
    "ScoringStrategy",
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
    "WalkEngine",
    "WalkStrategy",
    "WeightedTripleSampler",
    "aggregate_confidence",
    "build_triple_sampler",
    "build_walk_engine",
    "relation_degrees",
    "split_target_edges",
    "split_with_validation",
    "stratify_by_degree",
    "summarize_ranks",
]
