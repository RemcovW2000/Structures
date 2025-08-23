# Structures

Professional Python project scaffold with src/ layout, tests, and auto-generated docs from docstrings.

## Features
- src/ layout with package `structures`
- Tests with pytest and coverage
- Linting with ruff and formatting with black
- Type checking with mypy (strict)
- Pre-commit hooks
- MkDocs + mkdocstrings documentation (auto from docstrings)
- GitHub Actions CI for lint, type-check, tests, and docs deployment

## Quick start

1) Create and activate a virtual environment, then install dev dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[dev,docs]
```

2) Run linters, type-checker, tests:

```bash
ruff check .
black --check .
mypy
pytest --cov
```

3) Serve docs locally:

```bash
mkdocs serve
```

4) Install pre-commit hooks:

```bash
pre-commit install
```

## Makefile shortcuts

Common tasks are available via `make`:

- `make install` – install package + dev/docs deps
- `make fmt` – run black and ruff --fix
- `make lint` – run ruff in check mode
- `make typecheck` – run mypy
- `make test` – run pytest with coverage
- `make docs` – build docs
- `make docs-serve` – serve docs locally

## Releasing docs

Docs are deployed automatically from the `main` branch to GitHub Pages using GitHub Actions. Ensure repository Pages is set to deploy from `gh-pages` branch.
