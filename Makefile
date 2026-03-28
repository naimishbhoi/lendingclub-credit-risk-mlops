# =========================
# Project Configuration
# =========================
PROJECT_NAME := lendingclub-credit-risk-mlops
PYTHON := python
PIP := $(PYTHON) -m pip

SRC_DIR := src
TEST_DIR := tests
CONFIG_DIR := configs

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# =========================
# Help
# =========================
.PHONY: help install install-dev format lint test check clean precommit validate download
help:
	@echo "Available commands:"
	@echo "  make install        - Install project core dependencies"
	@echo "  make install-dev    - Install core + dev dependencies"
	@echo "  make format		 - Format code using Black"
	@echo "  make lint           - Lint code with ruff"
	@echo "  make test           - Run tests with pytest"
	@echo "  make check          - Run format, lint, and tests"
	@echo "  make validate       - Run validation pipeline"
	@echo "  make download       - Download datasets"
	@echo "  make clean          - Clean up build artifacts and caches"

# =========================
# Installation
# =========================
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

# =========================
# Code Quality
# =========================
format:
	ruff check $(SRC_DIR) $(TEST_DIR) --fix
	black $(SRC_DIR) $(TEST_DIR)

lint:
	ruff check --fix $(SRC_DIR) $(TEST_DIR)

# =========================
# Testing
# =========================
test:
	pytest $(TEST_DIR)

# ========================
# Check
# =========================
check: format lint test
	@echo "All checks passed."

# =========================
# Clean Up
# =========================
clean:
	rm -rf build dist .pytest_cache .ruff_cache *.egg-info

# =========================
# Pre-Commit
# =========================
precommit: format lint test
	@echo "Pre-commit checks passed."

# =========================
# Pipelines
# =========================
validate:
	$(PYTHON) -m pipelines.validate_pipeline --config-dir $(CONFIG_DIR)

# =========================
# Download Datasets
# =========================
download:
	$(PYTHON) -m scripts.download_datasets --config-dir $(CONFIG_DIR)
