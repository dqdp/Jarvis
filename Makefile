PYTHON ?= .venv/bin/python
PYTEST ?= $(PYTHON) -m pytest
UVICORN ?= $(PYTHON) -m uvicorn
APP_PYTHONPATH ?= src
DATABASE_URL ?= postgresql+asyncpg://jarvis:jarvis@127.0.0.1:5432/jarvis
CONFIG_PROFILE ?= default
OLLAMA_CHAT_MODEL ?= qwen3.5:4b
OLLAMA_EMBED_MODEL ?= embeddinggemma:latest
TEST_DATABASE_URL ?= postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test
TEST_COMPOSE ?= docker compose -f infra/compose/test-postgres.yml

.PHONY: migrate run cli models-list models-pull test test-unit test-contract test-integration test-golden test-architecture test-e2e test-db-up test-db-down

migrate:
	PYTHONPATH=$(APP_PYTHONPATH) DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m assistant_core.storage.migrations

run:
	PYTHONPATH=$(APP_PYTHONPATH) DATABASE_URL=$(DATABASE_URL) JARVIS_CONFIG_PROFILE=$(CONFIG_PROFILE) $(UVICORN) assistant_core.app_factory:create_asgi_app --factory --host 127.0.0.1 --port 8080

cli:
	PYTHONPATH=$(APP_PYTHONPATH) $(PYTHON) -m assistant_core.cli $(ARGS)

models-list:
	ollama list

models-pull:
	ollama pull $(OLLAMA_CHAT_MODEL)
	ollama pull $(OLLAMA_EMBED_MODEL)

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
	DATABASE_URL=$(TEST_DATABASE_URL) $(PYTEST) -m contract tests/contract

test-integration: test-db-up
	DATABASE_URL=$(TEST_DATABASE_URL) $(PYTEST) -m integration tests/integration

test-golden:
	$(PYTEST) -m golden tests/golden

test-architecture:
	$(PYTEST) -m architecture tests/architecture

test-e2e: test-db-up
	DATABASE_URL=$(TEST_DATABASE_URL) $(PYTEST) -m e2e tests/e2e
