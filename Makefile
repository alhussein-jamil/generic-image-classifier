.DEFAULT_GOAL := help
SHELL := /bin/bash

UV ?= uv
PYTHON ?= python3
PORT ?= 7860
ZIP ?=
MODEL ?= mobilenetv2
EPOCHS ?= 10

.PHONY: help install dev sync run app train both test test-cov lint format check \
	pre-commit-install pre-commit build clean

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: sync ## Install runtime dependencies

sync: ## Sync runtime dependencies (uv)
	$(UV) sync

dev: ## Sync with dev dependencies
	$(UV) sync --dev

run: app ## Launch Gradio UI (alias: app)

app: ## Launch all-in-one Gradio UI
	$(UV) run image-classifier --port $(PORT)

train: ## Train only (set ZIP=path/to/dataset.zip)
	@test -n "$(ZIP)" || { echo "Usage: make train ZIP=dataset.zip"; exit 1; }
	$(UV) run image-classifier --mode train --zip "$(ZIP)" --model $(MODEL) --epochs $(EPOCHS)

both: ## Train then launch UI (set ZIP=path/to/dataset.zip)
	@test -n "$(ZIP)" || { echo "Usage: make both ZIP=dataset.zip"; exit 1; }
	$(UV) run image-classifier --mode both --zip "$(ZIP)" --model $(MODEL) --epochs $(EPOCHS) --port $(PORT)

test: ## Run tests
	$(UV) run pytest

test-cov: ## Run tests with coverage report
	$(UV) run pytest --cov=generic_image_classifier --cov-report=term-missing

lint: ## Ruff lint check
	$(UV) run ruff check src tests

format: ## Format code with Ruff
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

check: lint test ## Lint and test

pre-commit-install: ## Install git pre-commit hooks
	$(UV) run pre-commit install

pre-commit: ## Run all pre-commit hooks
	$(UV) run pre-commit run --all-files

build: ## Build wheel and sdist
	$(UV) build

clean: ## Remove build artifacts and caches
	rm -rf build dist .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
