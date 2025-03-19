run:
	uv run src/main.py

run-debug:
	uv run src/main.py --debug

run-text:
	uv run src/main.py --debug --input="text" --output="text"

run-ja:
	uv run src/main.py --lang ja

test: ## Run unit tests
	uv run pytest

fmt: ## Run ruff format
	uv run ruff format

lint: ## Run ruff lint
	uv run ruff check

check: lint mypy ## Run check

fix: ## Run ruff check --fix
	uv run ruff check --fix

mypy: ## Run mypy
	uv run mypy . --ignore-missing-imports --no-namespace-packages

zip: ## Make zip archive
	zip -er archive.zip pyproject.toml Makefile README.md .gitignore .python-version src/ tests/

help: ## Display this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'
