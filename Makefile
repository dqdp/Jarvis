PYTHON ?= .venv/bin/python
PYTEST ?= $(PYTHON) -m pytest
TEST_DATABASE_URL ?= postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test
TEST_COMPOSE ?= docker compose -f infra/compose/test-postgres.yml

.PHONY: test test-unit test-contract test-integration test-golden test-architecture test-e2e test-db-up test-db-down

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
