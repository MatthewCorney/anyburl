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
:class:`LinkPredictionEvaluator`.

References
----------
.. [1] Meilicke, C., Chekol, M. W., Ruffinelli, D., & Stuckenschmidt, H.
   (2019). Anytime Bottom-Up Rule Learning for Knowledge Graph Completion.
   *IJCAI*.
"""

from .anyburl import AnyBURL, AnyBURLConfig
from .anytime import (
    AnytimeConfig,
    AnytimeLearner,
    AnytimeReport,
    LengthReport,
)
from .baselines import MetaPathScorer, PopularityScorer, RandomScorer
from .evaluation import (
    EntityScorer,
    EvaluationConfig,
    LinkPredictionEvaluator,
    LinkPredictionMetrics,
    TieHandling,
)
from .exceptions import (
    AnyBURLError,
    ConfigurationError,
    GraphSchemaError,
    InvalidRuleError,
    NotFittedError,
)
from .graph import HeteroGraph
from .metrics import (
    RuleEvaluator,
    RuleMetrics,
    aggregate_confidence,
)
from .prediction import (
    GroundingMode,
    Prediction,
    PredictionConfig,
    RuleFiring,
    RulePredictor,
    ScoringStrategy,
)
from .rule import (
    Atom,
    Rule,
    RuleConfig,
    RuleGeneralizer,
    RuleThresholds,
    RuleType,
    Term,
    TermKind,
)
from .sampler import (
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
from .walk import WalkConfig, WalkEngine, WalkStrategy

__all__ = [
    "AnyBURL",
    "AnyBURLConfig",
    "AnyBURLError",
    "AnytimeConfig",
    "AnytimeLearner",
    "AnytimeReport",
    "Atom",
    "ConfigurationError",
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
    "PopularityScorer",
    "Prediction",
    "PredictionConfig",
    "RandomScorer",
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
    "Term",
    "TermKind",
    "TieHandling",
    "Triple",
    "TripleSplit",
    "UniformTripleSampler",
    "WalkConfig",
    "WalkEngine",
    "WalkStrategy",
    "WeightedTripleSampler",
    "aggregate_confidence",
    "split_target_edges",
]
