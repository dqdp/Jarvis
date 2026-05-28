# Personal Assistant Runtime — Phase 1 MVP Documentation Package

Версия: reviewed baseline v16  
Дата: 2026-05-28  
Статус: **Accepted MVP documentation baseline, subject to revision after coding-agent analysis**.

## Назначение

Этот пакет фиксирует согласованную проектную документацию для Phase 1 локального личного ассистента.

Главный принцип Phase 1:

> Быстрый MVP, но с правильной архитектурой для последующего расширения.

Phase 1 строит не полноценного автономного агента, а надежный `Core Daemon`:

- local-first;
- durable;
- auditable;
- modular;
- TDD-first;
- replaceable by ports/adapters;
- готовый к будущим tools, MCP, voice, sleep/reflection, RAG/content retrieval и external LLM fallback.

## Как использовать пакет

1. Начать с `docs/00_project_charter.md`.
2. Затем прочитать `docs/01_target_architecture_overview.md`.
3. Для реализации использовать `docs/27_tdd_implementation_slices_plan.md`.
4. Для кодовых агентов использовать `AGENTS.md` и `docs/26_testing_strategy.md`.
5. При изменении архитектурных решений обновлять соответствующий ADR.

## Состав документов

```text
docs/
  00_project_charter.md
  01_target_architecture_overview.md
  02_storage_baseline_postgresql.md
  03_modularity_and_ports.md
  04_memory_subsystem_api.md
  05_component_ports_overview.md
  06_agent_runtime_and_loop_architecture.md
  07_core_daemon_design.md
  08_data_model_and_storage.md
  09_model_router_and_inference.md
  10_api_and_streaming.md
  11_observability_and_audit.md
  12_deployment_and_runtime.md
  13_phase_2_extension_points.md
  14_context_assembly.md
  15_post_mvp_context_management.md
  16_event_log_schema_and_correlation.md
  17_data_sensitivity_and_privacy_policy.md
  18_model_profiles_and_model_router.md
  19_embedding_and_retrieval_baseline.md
  20_post_mvp_rag_content_retrieval.md
  21_conversation_store_and_windowing.md
  22_api_shape_and_request_lifecycle.md
  23_error_handling_and_runtime_budgets.md
  24_post_mvp_agent_loop_followups.md
  25_configuration_model.md
  26_testing_strategy.md
  27_tdd_implementation_slices_plan.md
```

## ADR Index

```text
docs/adr/
  ADR-001_phase_1_uses_modular_monolith_with_ports_adapters_boundaries.md
  ADR-002_postgresql_is_the_primary_system_of_record.md
  ADR-003_pgvector_is_initial_retrieval_adapter,_not_memory_contract.md
  ADR-004_memory_subsystem_is_accessed_only_through_memory_ports.md
  ADR-005_replaceable_subsystems_require_contract_tests.md
  ADR-006_langgraph_as_phase_1_runtime_substrate.md
  ADR-007_model-router_boundary.md
  ADR-008_local-first_cloud-optional_policy.md
  ADR-009_streaming_events_as_first-class_runtime_api.md
  ADR-010_sleep_reflection_as_bounded_workflow.md
  ADR-011_defer_message_bus_until_background_workflows.md
  ADR-012_agent_loop_architecture.md
  ADR-013_memory_namespace_model.md
  ADR-014_memory_lifecycle_semantics.md
  ADR-015_context_assembler_as_core_phase_1_component.md
  ADR-016_context_representation_provider_neutral_post_mvp.md
  ADR-017_event_envelope_and_correlation_model.md
  ADR-018_minimal_data_sensitivity_model.md
  ADR-019_model_profiles_and_model_router_baseline.md
  ADR-020_embedding_and_retrieval_baseline.md
  ADR-021_full_rag_deferred_to_content_retrieval_subsystem.md
  ADR-022_conversation_store_and_windowing_baseline.md
  ADR-023_api_shape_and_request_lifecycle.md
  ADR-024_error_handling_and_runtime_budgets.md
  ADR-025_post_mvp_agent_loop_budget_and_capability_model.md
  ADR-026_configuration_model.md
  ADR-027_testing_strategy_for_agent_driven_development.md
  ADR-028_provisional_tdd_implementation_slices_plan.md
```

## Ключевые принятые решения

### Scope

- Phase 1 = Core Daemon MVP, не полноценный autonomous agent.
- Tools, MCP, RAG, voice, Telegram/Spotify integrations, ReAct, planner-executor и sleep/reflection implementation — out of MVP.
- Post-MVP направления явно зафиксированы в отдельных follow-up документах.

### Architecture style

- Modular monolith, not early microservices.
- Ports/adapters boundaries обязательны.
- Replaceable subsystems require contract tests.
- Runtime не зависит от конкретных adapters/providers/storage details.

### Storage

- PostgreSQL = primary system of record.
- pgvector = initial retrieval adapter, not Memory contract.
- LangGraph checkpoints хранятся в PostgreSQL как runtime state, not memory/audit truth.
- Phase 1 uses append-only event log as audit/reconstruction substrate, not full event sourcing.

### Event model

- Event log = immutable historical truth about system actions.
- All events use stable `EventEnvelope`.
- `request_id` links one user turn.
- `correlation_id` reserved for long-running workflows.
- `causation_id` links direct causal chain.
- Raw full prompts and token-by-token stream events are not persisted by default.

### Memory

- Minimal namespaces: `user.preferences`, `user.working_style`, `project.personal_assistant`, `system.runtime_rules`, `environment.inference_node`.
- Core memory types: `fact`, `preference`, `procedure`, `summary`.
- Specialized concepts such as `architecture_decision` are metadata, not core types.
- Lifecycle: `active`, `archived`, `superseded`.
- `secret` cannot be stored as long-term memory.

### Context

- Current context is ephemeral for one model call.
- ContextAssembler is a core Phase 1 component behind `ContextAssemblerPort`.
- AgentRuntime does not manually assemble prompts.
- Phase 1 ContextAssembler is deterministic-first.
- Full raw prompt logging is disabled by default; ContextManifest is stored instead.
- Advanced context techniques are post-MVP behind ContextAssemblerPort.

### Agent runtime

- LangGraph is execution substrate, not predefined agent architecture.
- Phase 1 loop = deterministic `memory_augmented_answer`.
- ReAct is deferred until `ToolGatewayPort` exists.
- Future loop strategies must declare budgets, capabilities, policy hooks, failure semantics and emitted events.

### Model routing

- ModelRouter is internal module/package in Phase 1.
- Local inference node is external OpenAI-compatible process.
- Required profiles: `local_main`, `local_structured`, `local_embedding`.
- `cloud_reasoning` may exist in config but is disabled by default.
- No automatic fallback, especially no cloud fallback.
- Embeddings go through `EmbeddingPort`, defaulting to `ModelRouter.embed(local_embedding)`.

### Retrieval and RAG

- Phase 1 indexes only explicit long-term `MemoryRecord` content/summary.
- No document/event/raw conversation/log/code vector indexing in MVP.
- Retrieval is active-only and namespace-aware.
- Full RAG is deferred to a separate future Content Retrieval subsystem.
- First post-MVP RAG target: project documentation / ADR corpus.

### API

- Message submission and streaming are separate.
- `POST /v1/conversations/{conversation_id}/messages` returns `request_id`.
- SSE endpoint: `GET /v1/requests/{request_id}/stream`.
- `client_message_id` provides idempotency per conversation.
- No token replay guarantee after SSE reconnect; final message is recoverable.

### Privacy and policy

- Sensitivity classes: `public`, `project`, `personal`, `infra`, `secret`.
- Cloud denied by default for all classes in Phase 1.
- `secret` never enters long-term memory, prompt context, raw logs or cloud.
- PolicyPort uses ConfigPolicyEngine in MVP.

### Configuration

- YAML config + environment overrides.
- No secrets in YAML.
- Strict startup validation.
- No hot reload in MVP.
- Model profiles, memory namespaces, budgets and windowing are config-driven.

### Testing and delivery

- Development is TDD-first.
- Tests are executable architecture policy for coding agents.
- Required layers: unit, contract, integration, golden, architecture, e2e.
- Fake model/embedding providers are mandatory.
- Real LLM calls are not required for CI.
- TDD implementation slice plan is accepted as provisional baseline.

## Current package status

This package is ready to be given to coding agents for initial repository analysis and implementation planning.

Allowed post-analysis changes:

- implementation-order-only changes update `docs/27_tdd_implementation_slices_plan.md`;
- architecture-changing changes require ADR update.

## Final hardening additions in v16

- README was rewritten as a stable index instead of chronological changelog.
- Missing ADR-016 was added for provider-neutral context representation.
- MVP acceptance checklist was added.
- Agent implementation instructions were added.
- Data model document was normalized to remove accumulated numbering drift.
- Hardening review notes were added.
