
PROJECT := ml-2026
SHELL := bash
PATH += $(HOME)/.local/bin
ACTIVATE := source .venv/bin/activate
# PYTHONPATH := .

all:
	ls -l

.venv:
	which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv

install: .venv
	$(ACTIVATE) && uv sync
	$(ACTIVATE) && pre-commit install

STRICT = --strict --warn-unreachable --ignore-missing-imports --no-namespace-packages

ruff-check:
	$(ACTIVATE) && black . && ruff check --fix
lint: ruff-check
	$(ACTIVATE) && pyright .
	$(ACTIVATE) && mypy $(STRICT) .
	$(ACTIVATE) && isort .

docker-build: clean-caches
	rm -rf .venv/
	docker buildx build --tag $(PROJECT) .
docker-run:
	docker run -v .:/tmp/ml-2025 -p 8000:8000 -it $(PROJECT)



test:
	$(ACTIVATE) && python -m unittest */*_test.py

CACHES := .mypy_cache/ .pyre/ .pytype/ .ruff_cache/
clean-caches:
	rm -rf $(CACHES)
clean: clean-caches
	rm -rf .venv/

.PHONY: all install ruff-check lint mypy docker-build docker-run lab nb notebook test clean-caches clean .venv
