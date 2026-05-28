# 03 — Modularity and Ports

## 1. Decision

Phase 1 uses a modular monolith with ports/adapters boundaries.

This is not a microservice architecture, but it must preserve replaceability of key subsystems.

## 2. Why Modular Monolith

Benefits:

- faster MVP;
- fewer moving parts;
- easier local deployment;
- easier tests and debugging;
- no premature distributed system;
- still allows future extraction of services.

## 3. Rule

Upper-level components depend only on domain schemas and ports.

They must not depend on adapter details.

## 4. Required Ports

Phase 1:

- ConversationStorePort;
- EventLogPort;
- MemoryReadPort;
- MemoryWritePort;
- ContextAssemblerPort;
- ModelRouterPort;
- EmbeddingPort;
- PolicyPort.

Future:

- MemoryCandidatePort;
- MemoryConsolidationPort;
- ToolGatewayPort;
- SchedulerPort;
- EventPublisherPort;
- VoiceGatewayPort.

## 5. Forbidden Dependencies

Agent Runtime must not import:

- SQL queries;
- ORM models;
- pgvector-specific code;
- vLLM/Ollama/OpenAI clients;
- MCP clients;
- shell execution clients;
- concrete storage adapters.

## 6. Allowed Dependencies

Agent Runtime may import:

- port interfaces;
- domain request/response schemas;
- RuntimeStreamEvent;
- LoopStrategy definitions;
- policy decision schemas.

## 7. Contract Tests

Every replaceable adapter must satisfy contract tests.

Examples:

```text
MemoryReadPort:
  retrieve active memories
  filter by namespace/type
  exclude archived memories
  return provenance

ContextAssemblerPort:
  assemble provider-neutral context
  return ContextManifest
  exclude secret sources

EventLogPort:
  append ordered events
  query by conversation/correlation id
  preserve payload

EmbeddingPort:
  generate local embeddings
  keep memory subsystem provider-independent
  preserve model invocation audit

ModelRouterPort:
  call local model profile
  stream events
  log invocation metadata
  enforce policy on cloud profile
```

## 8. Extraction Path

A module can become a service later if:

- it has stable port contract;
- adapter already satisfies contract tests;
- no domain code imports implementation internals;
- data ownership is clear.

Likely extraction candidates:

- memory service;
- model router;
- tool gateway;
- scheduler/event bus;
- voice gateway.
