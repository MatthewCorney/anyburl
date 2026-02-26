# anyburl

[![Tests](https://github.com/MatthewCorney/anyburl/actions/workflows/tests.yml/badge.svg)](https://github.com/MatthewCorney/anyburl/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

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
