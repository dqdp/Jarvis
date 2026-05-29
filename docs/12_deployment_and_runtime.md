# 12 — Deployment and Runtime

## 1. Phase 1 Deployment

```text
postgres
assistant-api
inference-node
```

## 2. Docker Compose Baseline

PostgreSQL:

- pgvector-enabled image preferred;
- persistent volume;
- migrations through Alembic.

assistant-api:

- FastAPI app;
- production composition root via `assistant_core.app_factory:create_asgi_app`;
- in-process ModelRouter;
- in-process MemoryService adapter;
- connects to Postgres and inference-node.

inference-node:

- vLLM preferred;
- Ollama acceptable for experiments;
- OpenAI-compatible API.

## 3. No Mandatory Queue

Redis/NATS are not Phase 1 dependencies.

Future background/event workflow may introduce NATS JetStream.

## 4. Configuration

Config layers:

```text
default yaml
environment overrides
local secrets file / env vars
```

Important flags:

```text
cloud_models_enabled: false
allow_autonomous_memory_write: false
allow_tools: false
streaming_transport: sse
```

## 5. Runtime commands

Local daemon commands:

```text
make migrate
make run
make cli
make cli ARGS='health'
make models-list
make models-pull
```

`make migrate` applies Alembic migrations to `DATABASE_URL`. Migration
entrypoints require a local database host by default; remote migration runs
must set `JARVIS_ALLOW_REMOTE_MIGRATIONS=1` explicitly.

`make run` starts Uvicorn with:

```text
assistant_core.app_factory:create_asgi_app --factory
```

The factory loads YAML config, creates the PostgreSQL engine, wires storage
adapters, `ConfigPolicyEngine`, `DeterministicContextAssembler`, `ModelRouter`,
local providers and the FastAPI app.

For local Ollama dogfood, run with:

```text
CONFIG_PROFILE=ollama make run
```

The `ollama` profile uses the locally available chat model `qwen3.5:9b` and the
embedding model `embeddinggemma:latest`. This is a runtime
profile behind the existing `ModelRouterPort`; CI still uses fake model and
embedding providers and does not require real LLM calls.
The Ollama adapter sends anti-repeat generation options and terminates obvious
repeated-line loops before they exhaust the configured output-token budget.

The CLI talks to the HTTP API:

```text
make cli
make cli ARGS='health'
make cli ARGS='memory add --memory-type preference Prefer concise answers'
make cli ARGS='chat Ответь ровно одним словом: OK'
```

Running `make cli` without `ARGS`, or `make cli ARGS='chat'`, opens an
interactive terminal chat shell. Normal input is sent to the current
conversation. Typing `/` in a TTY shows the available slash commands before
submission. Slash commands are handled client-side:

```text
/help
/new [title]
/memory add <content>
/memory list
/exit
```

During streaming, Ctrl-C interrupts the local CLI and sends best-effort
`POST /v1/requests/{request_id}/cancel` to the daemon. The local Ollama profile
also caps chat output at 1024 tokens to bound runaway generations.

On Unix TTY, Up/Down browse in-session input history. History is intentionally
not persisted to disk by default to avoid storing raw prompts. `/memory add`
payloads and `secret` sensitivity sessions are not added to input history.

## 6. Restart

Postgres persists domain state.

In-flight request recovery is best-effort in Phase 1.
