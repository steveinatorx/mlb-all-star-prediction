.PHONY: help setup install clean ingest build featurize train train-advanced eval report test lint format

help:
	@echo "MLB All-Star Prediction - Available commands:"
	@echo "  make setup      - Install dependencies"
	@echo "  make install    - Install package in development mode"
	@echo "  make clean      - Remove generated files and caches"
	@echo "  make ingest     - Download raw data"
	@echo "  make build      - Build processed dataset"
	@echo "  make featurize  - Generate features"
	@echo "  make train      - Train baseline models"
	@echo "  make train-advanced - Train models with SMOTE and class weights"
	@echo "  make eval       - Evaluate models and generate reports"
	@echo "  make report     - Generate markdown report"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"

setup:
	asdf install
	pipenv install --dev

install:
	pipenv install --dev

clean:
	rm -rf data/raw/* data/processed/* data/features/*
	rm -rf reports/figures/* reports/tables/*
	rm -rf experiments/*
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete

ingest:
	pipenv run python -m src.main ingest

build:
	pipenv run python -m src.main build-dataset

featurize:
	pipenv run python -m src.main featurize

train:
	pipenv run python -m src.main train

train-advanced:
	pipenv run python -m src.main train-advanced

eval:
	pipenv run python -m src.main evaluate

report:
	pipenv run python -m src.main report

test:
	pipenv run pytest tests/ -v

lint:
	pipenv run ruff check src/ tests/
	pipenv run mypy src/

format:
	pipenv run black --line-length 100 src/ tests/
	pipenv run ruff check --fix src/ tests/

