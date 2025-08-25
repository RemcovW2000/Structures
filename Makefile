# Makefile with common dev tasks
.PHONY: install fmt lint test clean docs docs-serve precommit hooks

install:
	python -m pip install --upgrade pip
	pip install -e .[dev,docs]

fmt:
	ruff check . --fix
	black .

lint:
	ruff check .
	black --check .

test:
	pytest --cov --cov-report=term-missing

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov site
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

precommit:
	pre-commit run --all-files

hooks:
	pre-commit install

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve
