# anyburl

[![Tests](https://github.com/MatthewCorney/anyburl/actions/workflows/tests.yml/badge.svg)](https://github.com/MatthewCorney/anyburl/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Package Layout

The public API is everything exported from `anyburl` itself. Modules prefixed
with an underscore, and the `anyburl.chain` package, are internal.

| Module | Responsibility |
|--------|----------------|
| `core` | Shared domain types: `Triple`, `EdgeTypeTuple`, `BodyChain`, `QueryDirection`, `EntityScorer` |
| `pipeline` | `AnyBURL` and `AnyBURLConfig`: the end-to-end fit / predict / evaluate entry point |
| `graph` | `HeteroGraph` wrapper over PyG `HeteroData`, its derived-index cache, and edge-type helpers |
| `sampler` | Target triple samplers and `build_triple_sampler` |
| `walk/` | The Numba random walk engine and the `PathWalker` protocol |
| `rule/` | Rule representation (`model`), quality metrics (`quality`: `RuleMetrics`), quality floors (`config`), path generalization (`generalizer`) |
| `rule_evaluation/` | The `RuleEvaluator`, which scores candidate rules, and the learned `RuleSet` |
| `acyclic/` | Length-one acyclic rules end to end: sampling, generalization, evaluation and scoring |
| `chain/` | Internal engine counting body-chain groundings for evaluation and prediction |
| `prediction/` | Applying rules: `RuleEvidence`, the fixed-aggregation `RulePredictor`, and `MultiRelationPredictor` |
| `aggregation/` | A learned aggregator that ranks candidates from rule evidence |
| `anytime` | Budgeted, length-by-length mining loop |
| `evaluation` | Link prediction metrics (MRR, Hits@K), per relation |
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
