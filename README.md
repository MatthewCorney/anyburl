# anyburl

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
