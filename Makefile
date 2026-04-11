.PHONY: install install-dev lint format format-check typecheck test coverage check pre-commit-install

install:
	python -m pip install -e .

install-dev:
	python -m pip install -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff check . --fix
	ruff format .

format-check:
	ruff format --check .

typecheck:
	mypy

test:
	pytest

coverage:
	coverage run -m pytest
	coverage report -m

check: lint format-check typecheck test

pre-commit-install:
	pre-commit install
