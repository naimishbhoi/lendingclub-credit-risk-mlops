# =========================
# Project Configuration
# =========================
PROJECT_NAME := lendingclub-credit-risk-mlops
PYTHON := python
PIP := pip

SRC_DIR := src
TEST_DIR := tests

# =========================
# Help
# =========================
.PHONY: help install install-dev test lint format clean
help:
	@echo "Available commands:"
	@echo "  make install        - Install project core dependencies"
	@echo "  make install-dev    - Install core + dev dependencies"
	@echo "  make format		 - Format code using Black"
	@echo "  make lint           - Lint code with ruff"
	@echo "  make test           - Run tests with pytest"
	@echo "  make clean          - Clean up build artifacts and caches"

# =========================
# Installation
# =========================
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# =========================
# Code Quality
# =========================
format:
	black src tests

lint:
	ruff check src tests

# =========================
# Testing
test:
	pytest tests

# =========================
# Clean Up
# =========================
clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache

