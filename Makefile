PYTHON ?= .venv/bin/python
PYTEST ?= $(PYTHON) -m pytest
UVICORN ?= $(PYTHON) -m uvicorn
APP_PYTHONPATH ?= src
DATABASE_URL ?= postgresql+asyncpg://jarvis:jarvis@127.0.0.1:5432/jarvis
CONFIG_PROFILE ?= default
OLLAMA_CHAT_MODEL ?= qwen3.5:9b
OLLAMA_STRUCTURED_MODEL ?= qwen3.5:2b
OLLAMA_EMBED_MODEL ?= embeddinggemma:latest
TEST_DATABASE_URL ?= postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test
TEST_COMPOSE ?= docker compose -f infra/compose/test-postgres.yml
JARVIS_RUNTIME ?= $(PYTHON) scripts/dev/jarvis_runtime.py

.PHONY: migrate run run-ollama cli models-list models-pull local-smoke content-ingest jarvis-bootstrap jarvis-up jarvis-cli jarvis-status jarvis-logs jarvis-down jarvis-reset test test-unit test-contract test-integration test-golden test-architecture test-e2e test-db-up test-db-down

migrate:
	PYTHONPATH=$(APP_PYTHONPATH) DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m assistant_core.storage.migrations

run:
	PYTHONPATH=$(APP_PYTHONPATH) DATABASE_URL=$(DATABASE_URL) JARVIS_CONFIG_PROFILE=$(CONFIG_PROFILE) $(UVICORN) assistant_core.app_factory:create_asgi_app --factory --host 127.0.0.1 --port 8080

run-ollama:
	CONFIG_PROFILE=ollama $(MAKE) run

cli:
	PYTHONPATH=$(APP_PYTHONPATH) $(PYTHON) -m assistant_core.cli $(ARGS)

models-list:
	ollama list

models-pull:
	ollama pull $(OLLAMA_CHAT_MODEL)
	ollama pull $(OLLAMA_STRUCTURED_MODEL)
	ollama pull $(OLLAMA_EMBED_MODEL)

local-smoke:
	$(MAKE) cli ARGS='health'

content-ingest:
	$(MAKE) cli ARGS='content ingest'

jarvis-bootstrap:
	$(JARVIS_RUNTIME) bootstrap

jarvis-up:
	$(JARVIS_RUNTIME) up

jarvis-cli:
	$(JARVIS_RUNTIME) cli $(ARGS)

jarvis-status:
	$(JARVIS_RUNTIME) status

jarvis-logs:
	$(JARVIS_RUNTIME) logs

jarvis-down:
	$(JARVIS_RUNTIME) down

jarvis-reset:
	@if [ "$(CONFIRM)" != "YES" ]; then echo "error> destructive reset requires CONFIRM=YES"; exit 1; fi
	$(JARVIS_RUNTIME) reset --yes

test: test-unit test-contract test-integration test-golden test-architecture test-e2e

test-unit:
	$(PYTEST) -m unit tests/unit

test-db-up:
	$(TEST_COMPOSE) up -d postgres-test
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		$(TEST_COMPOSE) exec -T postgres-test pg_isready -U jarvis -d jarvis_test >/dev/null 2>&1 && exit 0; \
		sleep 1; \
	done; \
	exit 1

test-db-down:
	$(TEST_COMPOSE) down -v

test-contract: test-db-up
	JARVIS_RUN_DB_TESTS=1 DATABASE_URL=$(TEST_DATABASE_URL) $(PYTEST) --run-db -m contract tests/contract

test-integration: test-db-up
	JARVIS_RUN_DB_TESTS=1 DATABASE_URL=$(TEST_DATABASE_URL) $(PYTEST) --run-db -m integration tests/integration

test-golden:
	$(PYTEST) -m golden tests/golden

test-architecture:
	$(PYTEST) -m architecture tests/architecture

test-e2e: test-db-up
	JARVIS_RUN_DB_TESTS=1 DATABASE_URL=$(TEST_DATABASE_URL) $(PYTEST) --run-db -m e2e tests/e2e
