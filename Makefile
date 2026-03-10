# GPM Makefile. Use: make install && make test. With venv: PYTHON=.venv/bin/python make test
.PHONY: install test test-all test-data test-prompts test-eval lint format benchmark quality-gate validate

PYTHON ?= python3
PYTEST = $(PYTHON) -m pytest

install:
	pip install -e ".[dev]"

test:
	$(PYTEST) tests/ -m "not slow" -v

test-all:
	$(PYTEST) tests/ -v

test-data:
	$(PYTEST) tests/ -m data -v

test-prompts:
	$(PYTEST) tests/ -m prompts -v

test-eval:
	$(PYTEST) tests/ -m eval -v

lint:
	ruff check .

format:
	ruff format .

benchmark:
	./scripts/benchmarks/run_all_benchmarks.sh 2>/dev/null || echo "Run: ./scripts/benchmarks/rhyme_bench/run_bench.py or rev_flux manually"

quality-gate:
	python scripts/data_generation/quality_gate.py data/educator_training/train.jsonl 2>/dev/null || echo "data/educator_training/train.jsonl not found"

validate: lint test
