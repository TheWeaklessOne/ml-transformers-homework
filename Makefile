.PHONY: install install-dev lint format format-check typecheck test coverage smoke check pre-commit-install

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
	pytest

smoke:
	transformers-day1 --help
	python -m transformers_week.day01_tokenization --help

check: lint format-check typecheck test smoke

pre-commit-install:
	pre-commit install
