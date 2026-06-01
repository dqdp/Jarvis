# 00 — Устав проекта Phase 1: Core Daemon

## 1. Назначение

Phase 1 закладывает минимальный, но архитектурно корректный фундамент для local-first runtime личного ассистента.

Цель первого этапа — не построить полноценного автономного ассистента сразу. Цель — создать надежный `Core Daemon`, который сможет эволюционировать в постоянно работающую агентную систему с долгосрочной памятью, локальным инференсом, опциональным доступом к внешним LLM, tools, MCP-интеграциями, Telegram, voice interaction и bounded self-reflection workflows.

Current baseline: post-MVP Alpha. Original Phase 1/MVP scope remains the
historical baseline, while PM-01..PM-07b have added bounded tools, approvals
and project-docs retrieval without changing the local-first/no-cloud default.

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
- deterministic runtime workflow;
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
- optional thin CLI or minimal web client for local/manual testing, with the HTTP API as the required MVP interface;
- deterministic memory-augmented conversation workflow in process;
- ContextAssembler as a first-class core component;
- PostgreSQL primary system of record;
- PostgreSQL-backed memory retrieval adapter, with pgvector as the preferred similarity-index implementation when available;
- event log;
- message storage;
- model invocation audit;
- manual memory creation;
- simple memory retrieval;
- model-router abstraction;
- one local OpenAI-compatible or native Ollama inference backend;
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
2. PostgreSQL-backed retrieval is the initial adapter; pgvector remains the preferred similarity-index implementation when available.
3. Memory subsystem доступна через MemoryReadPort / MemoryWritePort.
4. memory_candidates входят в data model, но auto-extraction не входит в MVP acceptance.
5. MVP runtime uses a custom deterministic workflow; LangGraph is deferred.
6. Phase 1 agent loop — deterministic memory-augmented workflow.
7. Original Phase 1 не использовал ReAct/tool loops.
8. Current baseline: post-MVP Alpha includes bounded `tool_react_loop` after
   `ToolGatewayPort`; planner-executor and autonomous ReAct remain deferred.
9. Graph checkpoints are deferred runtime state, not MVP storage scope.
10. Near-term priority is PM-08a through PM-08l: backend auto-selection, API
    lifecycle wiring, CLI mode controls, CLI tool/RAG/approval readiness,
    direct-answer hardening, request-quality gates and Codex-like interactive
    CLI shell UX, followed by canonical Jarvis runtime startup and PM-08k request
    handling cleanup. PM-08k makes the bounded agent loop the default
    natural-language path and removes classifier-first routing before PM-09 voice
    gateway foundation starts. PM-08l then hardens the agent-loop state machine,
    finalization and tool-observation recovery contracts before voice. LangGraph
    remains a follow-up for later durable workflows.
11. Redis/NATS не вводятся как обязательная зависимость Phase 1.
12. SSE — primary streaming transport.
13. RuntimeStreamEvent schema transport-agnostic.
14. Minimal PolicyPort вводится в Phase 1.
15. Model-router — internal module/package; inference node — external process.
16. Replaceable subsystems require contract tests.
17. Minimal memory namespaces: `user.preferences`, `user.working_style`, `project.personal_assistant`, `system.runtime_rules`, `environment.inference_node`.
18. Core memory types: `fact`, `preference`, `procedure`, `summary`.
19. Memory lifecycle statuses: `active`, `archived`, `superseded`.
20. Current context is ephemeral and managed by ContextAssembler, not by long-term memory lifecycle.
21. ContextAssembler facade/internal split is mandatory: AgentRuntime depends on `ContextAssemblerPort`, not on prompt assembly implementation.

## 9. Preferred Technology Baseline

```text
Language: Python
API: FastAPI
Runtime: custom deterministic workflow; LangGraph adapter deferred follow-up
Architecture: modular monolith + ports/adapters
Storage: PostgreSQL; pgvector-compatible retrieval adapter path
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
13. basic tests cover storage, model-router, context assembly and runtime workflow;
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


## 13. Event log и historical truth

Phase 1 использует append-only event log как immutable historical truth о действиях системы.

Это не full event sourcing: domain tables остаются operational read/write state, а event log обеспечивает audit, traceability и future reconstruction.

Каждое событие использует стабильный `EventEnvelope`. `request_id` связывает один user turn; `correlation_id` зарезервирован для long-running workflows; `causation_id` связывает прямую причинную цепочку.

Raw full prompts не хранятся по умолчанию. Для context assembly хранится `ContextManifest`. Raw message content хранится в `messages`, а events хранят refs/hashes/redacted snapshots.

## 14. Data sensitivity and privacy baseline

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

## 15. Hardening additions

Final reviewed package adds:

```text
28_mvp_acceptance_checklist.md
29_hardening_review_notes.md
AGENTS.md
ADR-016_context_representation_provider_neutral_post_mvp.md
```

`AGENTS.md` provides implementation rules for coding agents.
