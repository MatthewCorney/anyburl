"""AnyBURL: Anytime Bottom-Up Rule Learning for knowledge graphs.

This package implements the AnyBURL algorithm for learning first-order
Horn rules from heterogeneous knowledge graphs backed by PyTorch Geometric.

Algorithm Pipeline
------------------
The learning process follows four stages:

1. **Sample** --- :func:`build_triple_sampler` draws target triples from
   the graph according to a :class:`SamplingStrategy` (uniform, relation-
   proportional, or inverse-frequency weighted).

2. **Walk** --- a walk engine from :func:`build_walk_engine` performs
   bounded random walks from each sampled triple's head entity, searching
   for paths that reach the tail entity. Successful paths are returned as
   sequences of :class:`~anyburl.rule.PathStep`.

3. **Generalize** --- :class:`RuleGeneralizer` replaces concrete
   entities in each path with variables, producing typed Horn rules.
   Each path yields up to three :class:`Rule` variants:

   * **Cyclic** -- both head variables appear in the body chain.
   * **AC1** -- one head variable is grounded as a constant.
   * **AC2** -- one head variable is absent from the body entirely.

4. **Evaluate** --- :class:`RuleEvaluator` scores each rule via sparse
   CSR matrix multiplication against the graph, computing support,
   confidence, and head coverage (:class:`RuleMetrics`).

Module Layout
-------------
``graph``
    :class:`HeteroGraph` wraps PyG ``HeteroData`` with precomputed CSR
    indices for O(1) neighbor lookup and sparse matmul.
``sampler``
    :class:`SamplerConfig`, :class:`SamplingStrategy`, and the concrete
    samplers (:class:`UniformTripleSampler`, :class:`WeightedTripleSampler`).
``walk``
    :class:`WalkConfig`, :class:`WalkStrategy`, and the walk engines
    (:class:`NumbaWalkEngine`, :class:`WalkEngine`).
``rule``
    :class:`Rule`, :class:`Atom`, :class:`Term`, :class:`RuleGeneralizer`,
    and :class:`RuleConfig`.
``metrics``
    :class:`RuleEvaluator` and :class:`RuleMetrics`.
``evaluation``
    :class:`LinkPredictionEvaluator`, :class:`EvaluationConfig`,
    :class:`LinkPredictionMetrics`, and :class:`TieHandling`.

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
    "AnytimeConfig",
    "AnytimeLearner",
    "AnytimeReport",
    "Atom",
    "EntityBalancedTripleSampler",
    "EntityScorer",
    "EvaluationConfig",
    "GroundingMode",
    "HeteroGraph",
    "InverseEdgeHandling",
    "LengthReport",
    "LinkPredictionEvaluator",
    "LinkPredictionMetrics",
    "MetaPathScorer",
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
