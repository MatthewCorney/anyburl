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
| `pipeline` | `AnyBURL` and `AnyBURLConfig`: the end-to-end fit / predict / evaluate entry point |
| `factories` | Build samplers and walk engines from their configs |
| `graph` | `HeteroGraph` wrapper over PyG `HeteroData`, plus edge-type helpers |
| `sampler/` | Target triple samplers |
| `walk/` | Random walk engines (Numba kernel and reference torch engine) |
| `rule/` | Rule representation (`model`), quality floors (`config`), path generalization (`generalizer`) |
| `metrics/` | Rule quality metrics and the `RuleEvaluator` |
| `chain/` | Internal engine counting body-chain groundings for evaluation |
| `prediction/` | Applying rules: grounding, scoring and the `RulePredictor` |
| `anytime` | Budgeted, length-by-length mining loop |
| `evaluation` | Link prediction metrics (MRR, Hits@K) |
| `baselines` | Reference scorers for calibrating link prediction results |
| `split` | Train/test splitting of a target relation |
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
