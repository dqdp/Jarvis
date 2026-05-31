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
policy.tools_enabled: true
memory_augmented_answer.allow_tools: false
tool_react_loop.allow_tools: true
streaming_transport: sse
```

## 5. Runtime commands

Local daemon commands:

```text
make jarvis-bootstrap
make jarvis-up
make jarvis-status
make jarvis-cli
make jarvis-logs
make jarvis-down
make jarvis-reset CONFIRM=YES
make migrate
make run
make run-ollama
make cli
make cli ARGS='health'
make models-list
make models-pull
make local-smoke
```

The canonical local Jarvis runtime path is:

```text
make jarvis-up
make jarvis-cli
```

`make jarvis-up` starts a persistent local PostgreSQL service from
`infra/compose/jarvis-postgres.yml`, runs migrations, starts the daemon, writes
PID/log/redacted runtime metadata under `.run/jarvis/`, and waits for
`/v1/health`. `make jarvis-status`, `make jarvis-logs` and `make jarvis-down`
inspect or stop that owned daemon. `jarvis-down` verifies PID ownership before
sending signals, and status/runtime metadata must not persist raw database
credentials. `make jarvis-reset CONFIRM=YES` is the explicit destructive reset
path for the local Jarvis database volume.

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
make run-ollama
```

The `ollama` profile uses the locally available chat model `qwen3.5:9b`, the
structured/classifier model `qwen3.5:2b`, and the embedding model
`embeddinggemma:latest`. This is a runtime
profile behind the existing `ModelRouterPort`; CI still uses fake model and
embedding providers and does not require real LLM calls.
The classifier fast path is conservative: deterministic intent routing may skip
the structured model only when the deterministic confidence is greater than
`0.9`. Broad tool, project and live-state hints remain model-routed so the
selector does not silently bypass the classifier on ambiguous input. Smaller
structured-model candidates, including `qwen3.5:0.8b`, must be compared through
the local intent-routing evaluation before becoming the default. The threshold
can be tuned without a code change through
`JARVIS_LOOP_SELECTION__DETERMINISTIC_FAST_PATH_THRESHOLD`.
PM-08k is planned to simplify this classifier path further: the model-facing
structured output should become a small route classification, while runtime code
maps accepted routes into capabilities, tool names, risk classes and policy
semantics.
The Ollama adapter sends anti-repeat generation options and terminates obvious
repeated-line loops before they exhaust the configured output-token budget.
`GET /v1/health` also performs cheap provider readiness probes: Ollama uses
`/api/tags`, and local OpenAI-compatible providers use `/v1/models`. CI keeps
using fake providers/transports, so these probes do not make real LLM calls in
tests.

The CLI talks to the HTTP API:

```text
make cli
make cli ARGS='health'
make local-smoke
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

The interactive TTY shell shows a live status bar with an activity spinner while
a request is running. `--plain` disables terminal control for deterministic
scripts and accessibility/debugging. ANSI color defaults to auto-detection and
can be controlled with `--color auto|always|never`; `NO_COLOR`, `TERM=dumb` and
`--plain` disable color.

## 6. Restart

Postgres persists domain state.

In-flight request recovery is best-effort in Phase 1.
