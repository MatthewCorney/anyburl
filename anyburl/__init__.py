"""AnyBURL: Anytime Bottom-Up Rule Learning for knowledge graphs.

This package implements the AnyBURL algorithm for learning first-order
Horn rules from heterogeneous knowledge graphs backed by PyTorch Geometric.

Algorithm Pipeline
------------------
The learning process follows four stages:

1. **Sample** --- :class:`TripleSampler` draws target triples from the
   graph according to a :class:`SamplingStrategy` (uniform, relation-
   proportional, or inverse-frequency weighted).

2. **Walk** --- :class:`WalkEngine` performs bounded random walks from
   each sampled triple's head entity, searching for paths that reach
   the tail entity. Successful paths are returned as sequences of
   :class:`~anyburl.rule.PathStep`.

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
    :class:`TripleSampler` and :class:`SamplerConfig`.
``walk``
    :class:`WalkEngine` and :class:`WalkConfig`.
``rule``
    :class:`Rule`, :class:`Atom`, :class:`Term`, :class:`RuleGeneralizer`,
    and :class:`RuleConfig`.
``metrics``
    :class:`RuleEvaluator` and :class:`RuleMetrics`.
``evaluation``
    :class:`LinkPredictionEvaluator`, :class:`EvaluationConfig`,
    and :class:`LinkPredictionMetrics`.

References
----------
.. [1] Meilicke, C., Chekol, M. W., Ruffinelli, D., & Stuckenschmidt, H.
   (2019). Anytime Bottom-Up Rule Learning for Knowledge Graph Completion.
   *IJCAI*.
"""

from .anyburl import AnyBURL, AnyBURLConfig
from .evaluation import EvaluationConfig, LinkPredictionEvaluator, LinkPredictionMetrics
from .graph import HeteroGraph
from .metrics import RuleEvaluator, RuleMetrics, aggregate_confidence
from .prediction import Prediction, RulePredictor
from .rule import Atom, Rule, RuleConfig, RuleGeneralizer, RuleType, Term, TermKind
from .sampler import (
    SamplerConfig,
    SamplingStrategy,
    Triple,
    UniformTripleSampler,
    WeightedTripleSampler,
)
from .walk import WalkConfig, WalkEngine, WalkStrategy

__all__ = [
    "AnyBURL",
    "AnyBURLConfig",
    "Atom",
    "EvaluationConfig",
    "HeteroGraph",
    "LinkPredictionEvaluator",
    "LinkPredictionMetrics",
    "Prediction",
    "Rule",
    "RuleConfig",
    "RuleEvaluator",
    "RuleGeneralizer",
    "RuleMetrics",
    "RulePredictor",
    "RuleType",
    "SamplerConfig",
    "SamplingStrategy",
    "Term",
    "TermKind",
    "Triple",
    "UniformTripleSampler",
    "WalkConfig",
    "WalkEngine",
    "WalkStrategy",
    "WeightedTripleSampler",
    "aggregate_confidence",
]
