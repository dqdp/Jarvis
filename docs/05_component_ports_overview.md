# 05 — Component Ports Overview

## 1. Purpose

This document lists the core port boundaries for Phase 1 and future phases.

Ports define stable domain-level contracts. Adapters implement those contracts using PostgreSQL, pgvector, vLLM, Ollama, OpenAI-compatible APIs, or future services.

## 2. Phase 1 Ports

### ConversationStorePort

Owns conversation and message persistence.

```python
class ConversationStorePort(Protocol):
    async def create_conversation(self, command: CreateConversationCommand) -> Conversation: ...
    async def append_message(self, command: AppendMessageCommand) -> Message: ...
    async def load_recent_messages(self, query: RecentMessagesQuery) -> list[Message]: ...
```

### EventLogPort

Owns append-only event history.

```python
class EventLogPort(Protocol):
    async def append(self, event: EventRecord) -> EventId: ...
    async def query(self, filter: EventFilter) -> list[EventRecord]: ...
```

### MemoryReadPort / MemoryWritePort

Own long-term memory retrieval and controlled mutation.

See `04_memory_subsystem_api.md`.

### ContextAssemblerPort

Owns ephemeral prompt context construction for each model call.

Agent Runtime depends on this port and must not assemble prompts manually from storage, memory, templates and policies.

```python
class ContextAssemblerPort(Protocol):
    async def assemble(self, request: ContextAssemblyRequest) -> AssembledContext: ...
```

### ModelRouterPort

```python
class ModelRouterPort(Protocol):
    async def chat(self, request: ChatModelRequest) -> ChatModelResponse: ...
    async def stream_chat(self, request: ChatModelRequest) -> AsyncIterator[ModelStreamEvent]: ...
    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse: ...
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
```

### EmbeddingPort

`EmbeddingPort` is introduced in Phase 1 as a narrow facade for embedding generation.

Default implementation:

```text
EmbeddingPort -> ModelRouter.embed(local_embedding)
```

Purpose:

- keep Memory subsystem independent from concrete embedding providers;
- preserve model invocation audit for embeddings;
- allow future replacement of embedding backend without changing Memory subsystem.

`EmbeddingPort` is intentionally small and should not become a generic model router.

### PolicyPort

```python
class PolicyPort(Protocol):
    async def evaluate_model_request(self, request: ModelPolicyRequest) -> PolicyDecision: ...
    async def evaluate_memory_write(self, request: MemoryWritePolicyRequest) -> PolicyDecision: ...
    async def evaluate_context_inclusion(self, request: ContextPolicyRequest) -> PolicyDecision: ...
```

## 3. Future Ports

### ToolGatewayPort

For MCP, shell, search, Spotify, Telegram and other tools.

### SchedulerPort

For reminders, maintenance, sleep/reflection, proactive tasks.

### EventPublisherPort

For durable event bus integration.

### VoiceGatewayPort

For wake word, VAD, STT, TTS, realtime sessions.

## 4. Port Design Rules

- Ports use domain schemas.
- Ports do not expose SQL/ORM/provider-specific objects.
- Ports must be async-friendly.
- Ports must be testable through contract tests.
- Adapters can be in-process in Phase 1.
- Service extraction must not change Agent Runtime.
- Facade/internal separation is mandatory for Memory, Context Assembly and Model Router.

## Sensitivity-related port responsibilities

`PolicyPort` is responsible for minimal Phase 1 sensitivity enforcement:

- deny cloud model requests by default;
- reject memory writes with sensitivity `secret`;
- reject model requests whose assembled context contains `secret`.

`ContextAssemblerPort` must exclude `secret` sources from prompt context.

`MemoryWritePort` must reject `secret` memory records.

`EventLogPort` must persist sensitivity labels and redacted payloads for secret-bearing events.
