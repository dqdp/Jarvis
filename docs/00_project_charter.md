# 00 — Устав проекта Phase 1: Core Daemon

## 1. Назначение

Phase 1 закладывает минимальный, но архитектурно корректный фундамент для local-first runtime личного ассистента.

Цель первого этапа — не построить полноценного автономного ассистента сразу. Цель — создать надежный `Core Daemon`, который сможет эволюционировать в постоянно работающую агентную систему с долгосрочной памятью, локальным инференсом, опциональным доступом к внешним LLM, tools, MCP-интеграциями, Telegram, voice interaction и bounded self-reflection workflows.

Главный принцип:

> Сначала строим надежный runtime личного ассистента; автономность и широкий набор инструментов добавляем позже.

## 2. Продуктовая цель

Создать local-first систему личного ассистента, которая постоянно работает в приватном домашнем контуре и поддерживает широкий круг личных, инженерных, инфраструктурных, knowledge-management и automation задач.

В целевом состоянии ассистент должен поддерживать:

- interaction на естественном языке;
- долгосрочную персональную и проектную память;
- local LLM inference;
- optional external LLM fallback;
- controlled tool use;
- MCP integrations;
- Linux CLI через sandboxed policy layer;
- search;
- Telegram;
- Spotify и другие personal-service integrations;
- realtime voice chat;
- sleep/reflection cycles;
- scheduled/proactive tasks.

Phase 1 строит только core runtime, необходимый для этих будущих возможностей.

## 3. Миссия Phase 1

Phase 1 поставляет core daemon, который умеет:

1. принимать пользовательские сообщения через API и/или CLI client;
2. сохранять conversations, messages, events, memories, model invocations;
3. вызывать локальную LLM через model-router abstraction;
4. извлекать простую долгосрочную memory и подмешивать ее в prompt;
5. stream-ить ответы и runtime events;
6. переживать daemon restart без потери history;
7. предоставлять архитектуру, расширяемую tools, voice, MCP и sleep workflows.

## 4. Архитектурный замысел

Ассистент проектируется как long-running runtime system, а не stateless chatbot script.

Ключевые свойства:

- durable event log;
- explicit memory model;
- replaceable inference backend;
- local-first policy;
- cloud-optional model routing;
- auditable model invocations;
- graph-based runtime flow;
- streaming-first interaction model;
- modular monolith + ports/adapters;
- separation between runtime, storage, memory, context assembly, model routing, inference serving;
- replaceable Context Assembly subsystem behind `ContextAssemblerPort`;
- future extension points for tools, MCP, voice, scheduler, sleep/reflection.

## 5. Требование архитектурной модульности

Phase 1 использует modular monolith, но все ключевые подсистемы доступны только через explicit ports.

Agent Runtime не должен зависеть от:

- прямых SQL queries;
- ORM/SQLAlchemy models конкретного adapter;
- pgvector-specific retrieval code;
- прямых клиентов vLLM/Ollama/OpenAI;
- прямых клиентов MCP/shell/external services.

Agent Runtime может зависеть от:

- MemoryReadPort;
- ContextAssemblerPort;
- ModelRouterPort;
- EventLogPort;
- ConversationStorePort;
- PolicyPort;
- future ToolGatewayPort;
- future SchedulerPort.

Цель: сохранить быстрый MVP без архитектурного lock-in.

То же правило применяется к подсистеме текущего контекста. Agent Runtime не собирает prompt вручную из сообщений, memories, rules и templates. Он вызывает `ContextAssemblerPort`. Внутренняя реализация Context Assembler может эволюционировать от простого deterministic assembler в Phase 1 до более сложного context planner / reranker / compressor в будущих фазах без изменения agent loop contract.

## 6. Scope Phase 1

### In scope

- FastAPI assistant API;
- basic conversation lifecycle;
- CLI или minimal web client;
- deterministic memory-augmented conversation workflow на LangGraph;
- ContextAssembler as a first-class core component;
- PostgreSQL primary system of record;
- pgvector initial vector retrieval adapter;
- event log;
- message storage;
- model invocation audit;
- manual memory creation;
- simple memory retrieval;
- model-router abstraction;
- one local OpenAI-compatible inference backend;
- disabled external LLM profile;
- SSE response streaming;
- structured logging;
- Docker Compose local deployment;
- basic tests and contract tests.

### Out of scope

- unrestricted Linux shell tool execution;
- MCP gateway;
- Telegram;
- Spotify;
- realtime voice;
- wake word / VAD / STT / TTS;
- autonomous background tool execution;
- full sleep/reflection loop;
- complex multi-agent orchestration;
- production-grade permissions/user management;
- polished UI;
- distributed deployment;
- fine-tuning/training pipeline.

## 7. Non-goals

Phase 1 не предназначена для:

- fully autonomous assistant;
- максимизации benchmark performance;
- поддержки всех tools;
- multi-user SaaS deployment;
- complex permission system;
- polished consumer UI;
- production-grade distributed agent platform.

## 8. Согласованные архитектурные решения

1. PostgreSQL выбран как primary system of record.
2. pgvector используется как initial retrieval adapter.
3. Memory subsystem доступна через MemoryReadPort / MemoryWritePort.
4. memory_candidates входят в data model, но auto-extraction не входит в MVP acceptance.
5. LangGraph используется как execution substrate.
6. Phase 1 agent loop — deterministic memory-augmented workflow.
7. ReAct не используется в Phase 1.
8. ReAct-style tool loop откладывается до ToolGatewayPort.
9. LangGraph checkpoints хранятся в PostgreSQL как runtime state.
10. Redis/NATS не вводятся как обязательная зависимость Phase 1.
11. SSE — primary streaming transport.
12. RuntimeStreamEvent schema transport-agnostic.
13. Minimal PolicyPort вводится в Phase 1.
14. Model-router — internal module/package; inference node — external process.
15. Replaceable subsystems require contract tests.
16. Minimal memory namespaces: `user.preferences`, `user.working_style`, `project.personal_assistant`, `system.runtime_rules`, `environment.inference_node`.
17. Core memory types: `fact`, `preference`, `procedure`, `summary`.
18. Memory lifecycle statuses: `active`, `archived`, `superseded`.
19. Current context is ephemeral and managed by ContextAssembler, not by long-term memory lifecycle.
20. ContextAssembler facade/internal split is mandatory: AgentRuntime depends on `ContextAssemblerPort`, not on prompt assembly implementation.

## 9. Preferred Technology Baseline

```text
Language: Python
API: FastAPI
Runtime: LangGraph
Architecture: modular monolith + ports/adapters
Storage: PostgreSQL + pgvector
Inference serving: vLLM preferred, Ollama acceptable for early experiments
Streaming: SSE
Deployment: Docker Compose
Testing: pytest + integration tests + contract tests
Migrations: Alembic
Configuration: YAML + environment overrides
```

## 10. Acceptance Criteria

Phase 1 считается завершенной, когда:

1. user can create conversation;
2. user can send message;
3. assistant responds using local model;
4. response can be streamed through SSE;
5. conversations/messages/events/model invocations are persisted;
6. manual memory records can be created;
7. memory retrieval influences prompt through ContextAssembler;
8. ContextAssembler builds deterministic prompt context with recent messages, active memories, runtime rules and token budget;
9. daemon restart does not lose history;
10. ModelRouter hides concrete inference backend;
11. external LLM profile exists but disabled by default;
12. system runs through Docker Compose;
13. basic tests cover storage, model-router, context assembly, graph flow;
14. contract tests cover replaceable ports.

## 11. Основные риски

### Scope creep

Phase 1 может разрастись до tools, voice, autonomy и UI.

Mitigation: удерживать MVP вокруг core daemon, storage, memory, model routing, deterministic runtime.

### Memory contamination

Model может писать noisy/sensitive memories.

Mitigation: Phase 1 runtime only reads memory; writes are manual or controlled; auto-extraction deferred.

### Backend lock-in

Runtime может начать зависеть от vLLM/pgvector/PostgreSQL details.

Mitigation: ports/adapters + contract tests, including ContextAssemblerPort contract tests.

### Cloud leakage

Private data может уйти во external LLM.

Mitigation: cloud disabled by default; PolicyPort; audit.

### Agent self-loop

Agent может начать бесконтрольный loop.

Mitigation: Phase 1 no ReAct; all future loops must define budgets/stopping conditions.

## 12. Documentation Package

См. `README.md` и остальные документы пакета.


## 20. Event log и historical truth

Phase 1 использует append-only event log как immutable historical truth о действиях системы.

Это не full event sourcing: domain tables остаются operational read/write state, а event log обеспечивает audit, traceability и future reconstruction.

Каждое событие использует стабильный `EventEnvelope`. `request_id` связывает один user turn; `correlation_id` зарезервирован для long-running workflows; `causation_id` связывает прямую причинную цепочку.

Raw full prompts не хранятся по умолчанию. Для context assembly хранится `ContextManifest`. Raw message content хранится в `messages`, а events хранят refs/hashes/redacted snapshots.

## 21. Data sensitivity and privacy baseline

Phase 1 вводит минимальную модель чувствительности данных:

```text
public
project
personal
infra
secret
```

Все core artifacts должны иметь sensitivity label:

- events;
- messages;
- memories;
- model invocations;
- context manifests.

Hard rules:

- `secret` не сохраняется как long-term memory;
- `secret` не включается в prompt context;
- `secret` не отправляется во внешнюю LLM;
- `secret` не логируется raw;
- cloud model access denied by default для всех sensitivity classes в Phase 1.

Phase 1 не включает LLM-based sensitivity classifier, full PII detector, secret manager integration или policy DSL. Эти возможности отложены.


---

## 20. Hardening additions

Final reviewed package adds:

```text
28_mvp_acceptance_checklist.md
29_hardening_review_notes.md
AGENTS.md
ADR-016_context_representation_provider_neutral_post_mvp.md
```

`AGENTS.md` provides implementation rules for coding agents.
