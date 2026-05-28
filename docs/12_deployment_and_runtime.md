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

## 5. Restart

Postgres persists domain state.

In-flight request recovery is best-effort in Phase 1.
