# 01 — Target Architecture Overview

## 1. Purpose

Документ описывает целевую архитектуру Phase 1 с высоты компонентов: runtime, storage, memory, model routing, streaming и future extension points.

Current baseline: post-MVP Alpha. The original MVP baseline used only the
deterministic memory-augmented loop; PM-01..PM-07b add bounded tools, approvals
and project-docs retrieval while keeping the same ports/adapters shape.

## 2. High-Level Architecture

```text
Client / CLI / Minimal Web
        |
        v
Assistant API
        |
        v
Agent Runtime / custom deterministic workflow
        |
        +--> ConversationStorePort --> PostgreSQL
        +--> EventLogPort -----------> PostgreSQL
        +--> ContextAssemblerPort ---> Context Assembly module
        |       +--> ConversationStorePort
        |       +--> MemoryReadPort -> Memory Subsystem Adapter
        |       +--> PolicyPort
        +--> ModelRouterPort --------> Local Inference Node
        +--> PolicyPort -------------> ConfigPolicyEngine
```

Physical Phase 1 deployment:

```text
assistant-api process:
  FastAPI
  Agent Runtime
  Context Assembly module
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
  graph checkpoints deferred post-MVP

inference-node process:
  vLLM / Ollama / NIM-like OpenAI-compatible server
```

The implemented MVP uses FastAPI for the assistant API route contract.

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
- execute the deterministic MVP workflow;
- call ContextAssemblerPort for recent context, memory retrieval and prompt context assembly;
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

Phase 1 and post-MVP Alpha implementation is policy-gated:

- allow local model;
- deny cloud unless explicitly enabled;
- allow only configured tool capabilities through ToolGateway and approval policy;
- allow content retrieval, ingestion and indexing only through configured permission modes;
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
  in-process request task state during MVP; graph checkpoints deferred post-MVP.
```

Rules:

- checkpoints are not memory;
- memory is not historical truth;
- event log can be used to rebuild memory;
- runtime state can be discarded/rebuilt more aggressively than events.

## 5. Graph Checkpoints

Decision:

- MVP does not require graph checkpoint tables.
- LangGraph is deferred until graph-native branching or checkpoint replay is required.
- Future checkpoint tables may be physically colocated with domain data for simplicity.
- They remain logically separate runtime state.
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

- MVP runtime uses a custom deterministic workflow behind AgentRuntime.
- LangGraph is deferred; it may become an adapter/substrate for later loop strategies.
- Phase 1 selected loop: deterministic memory-augmented workflow.
- Current post-MVP Alpha adds bounded `tool_react_loop` after ToolGatewayPort
  and the loop-strategy boundary were introduced.
- Near-term priority is PM-08a through PM-08l: automatic loop selection, API
  lifecycle wiring, CLI mode controls, CLI tool/RAG/approval readiness,
  direct-answer hardening, request-quality gates and Codex-like interactive CLI
  shell UX, followed by canonical Jarvis runtime startup and PM-08k request
  handling cleanup. PM-08k makes the bounded agent loop the default
  natural-language path and removes classifier-first routing before voice.
  PM-08l hardens the agent-loop state machine, finalization, tool-observation
  recovery and DB-backed transcript-like API evidence. PM-09 voice gateway
  foundation remains blocked until the final PM-08l verification/review gate is
  green. LangGraph remains a follow-up for later durable workflows.
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
  modular local/external speech provider profiles
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


## 11. Context Assembly Boundary

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
