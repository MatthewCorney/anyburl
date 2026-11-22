"""AnyBURL: Anytime Bottom-Up Rule Learning for heterogeneous knowledge graphs.

Learns first-order Horn rules from a PyTorch Geometric ``HeteroData`` graph
and applies them to link prediction.

The top level exports the everyday API: :class:`AnyBURL` and its config,
graph and domain types, rules and their metrics, prediction, evaluation,
splits, anytime mining, the learned aggregator, and the exception hierarchy.
Advanced building blocks stay in their subpackages: :mod:`anyburl.rule`,
:mod:`anyburl.rule_evaluation`, :mod:`anyburl.sampler`, :mod:`anyburl.walk`,
:mod:`anyburl.prediction`, :mod:`anyburl.aggregation`,
:mod:`anyburl.evaluation`, :mod:`anyburl.anytime`, :mod:`anyburl.baselines`
and :mod:`anyburl.stratification`. Modules prefixed with an underscore, and
:mod:`anyburl.chain`, are internal.

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
    FeaturizerKind,
)
from .anytime import AnytimeConfig, AnytimeReport
from .core import EdgeTypeTuple, EntityScorer, QueryDirection, Triple
from .evaluation import (
    EvaluationConfig,
    LinkPredictionEvaluator,
    LinkPredictionMetrics,
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
from .graph import HeteroGraph
from .pipeline import AnyBURL, AnyBURLConfig
from .prediction import (
    ConfidenceAggregation,
    GroundingWeighting,
    MultiRelationPredictor,
    Prediction,
    PredictionConfig,
    RuleFiring,
    RulePredictor,
)
from .rule import Rule, RuleMetrics, RuleThresholds, RuleType
from .rule_evaluation import RuleSet
from .sampler import SamplingStrategy
from .split import (
    InverseEdgeHandling,
    SplitConfig,
    TripleSplit,
    ValidationSplit,
    split_target_edges,
    split_with_validation,
)
from .walk import EdgeWeighting, WalkStrategy

__all__ = [
    "AggregatedScorer",
    "AggregationSetup",
    "AggregatorConfig",
    "AnyBURL",
    "AnyBURLConfig",
    "AnyBURLError",
    "AnytimeConfig",
    "AnytimeReport",
    "ConfidenceAggregation",
    "ConfigurationError",
    "EdgeTypeTuple",
    "EdgeWeighting",
    "EntityScorer",
    "EvaluationConfig",
    "FeaturizerKind",
    "GraphSchemaError",
    "GroundingWeighting",
    "HeteroGraph",
    "InvalidRuleError",
    "InverseEdgeHandling",
    "LinkPredictionEvaluator",
    "LinkPredictionMetrics",
    "MultiRelationPredictor",
    "NotFittedError",
    "Prediction",
    "PredictionConfig",
    "QueryDirection",
    "Rule",
    "RuleFiring",
    "RuleMetrics",
    "RulePredictor",
    "RuleSet",
    "RuleThresholds",
    "RuleType",
    "SamplingStrategy",
    "SplitConfig",
    "TieHandling",
    "Triple",
    "TripleSplit",
    "ValidationSplit",
    "WalkStrategy",
    "split_target_edges",
    "split_with_validation",
    "summarize_ranks",
]
