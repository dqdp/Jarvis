# 01 — Target Architecture Overview

## 1. Purpose

Документ описывает целевую архитектуру Phase 1 с высоты компонентов: runtime, storage, memory, model routing, streaming и future extension points.

## 2. High-Level Architecture

```text
Client / CLI / Minimal Web
        |
        v
Assistant API
        |
        v
Agent Runtime / LangGraph
        |
        +--> ConversationStorePort --> PostgreSQL
        +--> EventLogPort -----------> PostgreSQL
        +--> MemoryReadPort ---------> Memory Subsystem Adapter
        +--> ModelRouterPort --------> Local Inference Node
        +--> PolicyPort -------------> ConfigPolicyEngine
```

Physical Phase 1 deployment:

```text
assistant-api process:
  FastAPI
  Agent Runtime
  Model Router module
  Memory Service module
  Storage adapters
  Policy module

postgres process:
  conversations
  messages
  events
  memories
  memory_candidates
  embeddings
  model_invocations
  runtime checkpoints

inference-node process:
  vLLM / Ollama / NIM-like OpenAI-compatible server
```

## 3. Component Boundaries

### Assistant API

Responsibilities:

- expose conversation/message endpoints;
- expose SSE streaming endpoint;
- validate input;
- attach request/correlation IDs;
- call Agent Runtime;
- return response and RuntimeStreamEvents.

### Agent Runtime

Responsibilities:

- select Loop Strategy;
- execute LangGraph workflow;
- load recent context;
- retrieve memory;
- build prompt;
- call ModelRouterPort;
- emit RuntimeStreamEvents;
- persist domain events.

Phase 1 runtime uses deterministic memory-augmented workflow.

### Conversation Store

Owns user-visible conversation state:

- conversations;
- messages;
- recent message window;
- conversation metadata.

### Event Log

Owns historical truth:

- user.message;
- assistant.message;
- model.request/model.response;
- memory.retrieved;
- policy.decision;
- runtime.error.

Event log is append-only by default.

### Memory Subsystem

Owns interpreted knowledge:

- semantic memory;
- episodic summaries;
- preferences;
- project facts;
- memory_candidates;
- embeddings/retrieval index.

Memory is replaceable behind MemoryReadPort/MemoryWritePort.

### Model Router

Owns model provider abstraction:

- local model profiles;
- disabled cloud profile;
- streaming;
- structured output mode;
- model invocation logging;
- policy check before external provider calls.

### Policy Engine

Phase 1 implementation is minimal:

- allow local model;
- deny cloud unless explicitly enabled;
- deny tools because tools are out of scope;
- deny autonomous memory writes by default.

## 4. State Taxonomy

The system distinguishes four state classes:

```text
Conversation state:
  messages and conversation metadata.

Historical event state:
  append-only event log.

Long-term memory:
  interpreted knowledge with provenance/confidence/status.

Runtime execution state:
  LangGraph checkpoints, graph thread state, intermediate values.
```

Rules:

- checkpoints are not memory;
- memory is not historical truth;
- event log can be used to rebuild memory;
- runtime state can be discarded/rebuilt more aggressively than events.

## 5. LangGraph Checkpoints

Decision:

- Phase 1 stores LangGraph checkpoints in PostgreSQL.
- They are physically colocated with domain data for simplicity.
- They are logically separate runtime state.
- Domain logic must not depend on checkpoint schema.

## 6. Queue/Event Bus

Decision:

- Phase 1 has no mandatory Redis/NATS dependency.
- User request path is synchronous.
- Future SchedulerPort/EventPublisherPort may exist as no-op/in-process adapters.
- Phase 2 may introduce NATS JetStream for durable event-driven workflows.

## 7. Streaming

Decision:

- Phase 1 primary streaming transport: SSE.
- WebSocket deferred to realtime/control/voice phases.
- RuntimeStreamEvent schema must be transport-agnostic.

## 8. Runtime Loop Strategy

Decision:

- LangGraph is execution substrate, not agent architecture.
- Phase 1 selected loop: deterministic memory-augmented workflow.
- ReAct-style tool loop is deferred until ToolGatewayPort exists.
- All future loop strategies must define budgets, stopping conditions, policy hooks and emitted events.

## 9. Extension Points

Future components:

```text
ToolGatewayPort
  MCP gateway
  shell sandbox
  search
  Spotify
  Telegram

SchedulerPort/EventPublisherPort
  background jobs
  reminders
  sleep/reflection
  proactive tasks

Voice Gateway
  wake word
  VAD
  STT
  TTS
  realtime model sessions

Advanced Memory
  hybrid retrieval
  Qdrant/Weaviate/Milvus
  reranking
  consolidation
  memory editor
```

## 10. Open Questions for Later Phases

Not decided in Phase 1:

- exact MCP gateway design;
- shell sandbox implementation;
- voice architecture;
- advanced permission model;
- NATS vs Redis Streams final choice for Phase 2;
- planner-executor design;
- autonomous task lifecycle.


## 10. Context Assembly Boundary

Current context management is a separate core subsystem.

Agent Runtime calls `ContextAssemblerPort`; it does not construct prompts directly.

```text
Agent Runtime
  -> ContextAssemblerPort
      -> ConversationStorePort
      -> MemoryReadPort
      -> PolicyPort
      -> PromptTemplateRegistry
  -> ModelRouterPort
```

This mirrors the memory subsystem design: stable facade, replaceable internals.

Phase 1 implementation is deterministic and minimal. Future implementations may add reranking, compression, rolling summaries and tool-aware context without changing Agent Runtime.
