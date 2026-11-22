# anyburl

[![Tests](https://github.com/MatthewCorney/anyburl/actions/workflows/tests.yml/badge.svg)](https://github.com/MatthewCorney/anyburl/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

`anyburl` is a Python implementation of AnyBURL (Anytime Bottom-Up Rule
Learning, Meilicke et al., IJCAI 2019) for heterogeneous knowledge graphs
held as PyTorch Geometric `HeteroData`. It samples target edges, walks paths
between their endpoints, generalizes the paths into Horn rules, scores the
rules by counting their groundings, and uses the rules that pass their
quality thresholds for explainable link prediction.

## Scope

In scope:

- Heterogeneous PyG graphs; entities are identified by `(id, node_type)`.
- Cyclic and AC1 chain rules of any body length, plus length-one acyclic
  rules (`r(X, c) :- b(X, d)` and `r(X, c) :- b(X, A)`; opt-in through
  `AnyBURLConfig.acyclic_samples_per_end`).
- Bottom-up sampling with a Numba random walk engine.
- Fixed aggregation of rule confidences (noisy-or, maximum, lexicographic)
  and a learned aggregator fitted on held-out triples.
- Filtered MRR and Hits@K evaluation, per relation.
- Breaking ranks down by target degree.
- Budgeted anytime mining, length by length.

Out of scope:

- Rules with a head variable absent from the body are not learned.
- No rule-file I/O and no interoperability with the Java AnyBURL.
- No GPU or distributed execution; walks and grounding counts run on CPU.
- The torch walk engine is removed; walks use Numba only.

## Quickstart

The three blocks run in order in one session.

**1. Fit, predict and explain.**

```python
import torch
from torch_geometric.data import HeteroData

from anyburl import AnyBURL, AnyBURLConfig

LIVES_IN = ("person", "lives_in", "city")
persons = torch.arange(30)
cities = persons % 5
ring = torch.arange(5)

data = HeteroData()
data["person"].num_nodes = 30
data["city"].num_nodes = 5
data["person", "born_in", "city"].edge_index = torch.stack([persons, cities])
data[LIVES_IN].edge_index = torch.stack([persons, cities])
data["city", "near", "city"].edge_index = torch.stack([ring, (ring + 1) % 5])

config = AnyBURLConfig(
    target_edge_type=LIVES_IN, sample_size=20, min_walk_length=1, max_walk_length=2
)
pipeline = AnyBURL(config).fit(data)

for rule, metrics in pipeline.results:
    print(rule, f"conf={metrics.confidence:.2f}")

print(len(pipeline.predict()), "pairs scored")
for prediction in pipeline.predictor().predictor_for(LIVES_IN).top_tails(0, limit=2):
    print(prediction.tail_id, prediction.score)
    for firing in prediction.explanations:
        print("  ", firing.describe())
```

**2. Split, fit and evaluate on test triples.**

```python
from anyburl import SplitConfig, split_with_validation

split = split_with_validation(
    data,
    SplitConfig(target_edge_type=LIVES_IN, test_fraction=0.2),
    validation_fraction=0.2,
)
pipeline = AnyBURL(config).fit(split.fit_data)
metrics = pipeline.evaluate_predictions(split.test_triples, known=split.train_data)
print(f"MRR={metrics.mrr:.3f}", metrics.hits_at_k)
```

**3. Fit a learned aggregator on validation triples and evaluate it.**

```python
from anyburl import EvaluationConfig, HeteroGraph, LinkPredictionEvaluator

scorer = pipeline.fit_aggregator(split.validation_triples)
evaluator = LinkPredictionEvaluator(
    scorer, HeteroGraph(split.train_data), EvaluationConfig()
)
print(f"learned MRR={evaluator.evaluate(split.test_triples).mrr:.3f}")
```

## Public API

Everything in `anyburl.__all__` is the everyday API:

- **Entry:** `AnyBURL`, `AnyBURLConfig`, `HeteroGraph`, `Triple`, `EdgeTypeTuple`, `QueryDirection`, `EntityScorer`
- **Rules:** `Rule`, `RuleType`, `RuleMetrics`, `RuleSet`, `RuleThresholds`
- **Config enums:** `SamplingStrategy`, `WalkStrategy`, `EdgeWeighting`
- **Prediction:** `PredictionConfig`, `ConfidenceAggregation`, `GroundingWeighting`, `Prediction`, `RuleFiring`, `RulePredictor`, `MultiRelationPredictor`
- **Evaluation:** `EvaluationConfig`, `TieHandling`, `LinkPredictionMetrics`, `LinkPredictionEvaluator`, `summarize_ranks`
- **Splits:** `SplitConfig`, `TripleSplit`, `InverseEdgeHandling`, `split_target_edges`, `ValidationSplit`, `split_with_validation`
- **Anytime:** `AnytimeConfig`, `AnytimeReport`
- **Learned aggregator:** `AggregatedScorer`, `AggregatorConfig`, `AggregationSetup`, `FeaturizerKind`
- **Errors:** `AnyBURLError`, `ConfigurationError`, `GraphSchemaError`, `InvalidRuleError`, `NotFittedError`

Advanced building blocks are imported from their subpackages:
`anyburl.rule` (atoms, terms, `RuleGeneralizer`), `anyburl.rule_evaluation`
(`RuleEvaluator`), `anyburl.sampler` (triple samplers), `anyburl.walk`
(`NumbaWalkEngine`, `WalkConfig`), `anyburl.prediction` (`RuleEvidence`),
`anyburl.aggregation` (featurizers, aggregators, training sets),
`anyburl.evaluation` (per-query ranks), `anyburl.anytime` (`AnytimeLearner`),
`anyburl.baselines` (reference scorers) and `anyburl.stratification`
(degree buckets).

## Package Layout

Modules prefixed with an underscore, and the `anyburl.chain` package, are
internal.

| Module | Responsibility |
|--------|----------------|
| `core` | Shared domain types: `Triple`, `EdgeTypeTuple`, `BodyChain`, `QueryDirection`, `EntityScorer` |
| `pipeline` | `AnyBURL` and `AnyBURLConfig`: the end-to-end fit / predict / evaluate entry point |
| `graph` | `HeteroGraph` wrapper over PyG `HeteroData`, its derived-index cache, and edge-type helpers |
| `sampler` | Target triple samplers and `build_triple_sampler` |
| `walk/` | The Numba random walk engine and the `PathWalker` protocol |
| `rule/` | Rule representation (`model`), quality metrics (`quality`), quality floors (`config`), path generalization (`generalizer`) |
| `rule_evaluation/` | `RuleEvaluator`, which scores candidate rules, and the learned `RuleSet` |
| `acyclic/` | Length-one acyclic rules end to end: sampling, generalization, evaluation and scoring |
| `chain/` | Internal engine counting body-chain groundings for evaluation and prediction |
| `prediction/` | Applying rules: `RuleEvidence`, the fixed-aggregation `RulePredictor`, and `MultiRelationPredictor` |
| `aggregation/` | The learned aggregator: featurizers, models, training sets and `AggregatedScorer` |
| `anytime` | Budgeted, length-by-length mining loop |
| `evaluation` | Filtered link prediction metrics (MRR, Hits@K) and per-query ranks |
| `baselines` | Reference scorers for calibrating link prediction results |
| `split` | Train/validation/test splitting of a target relation |
| `stratification` | Breaking ranks down by target degree |
| `exceptions` | `AnyBURLError` hierarchy |

## Development Setup

```bash
poetry install
```

## Pre-Commit Checks

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy anyburl/
poetry run pytest
```
