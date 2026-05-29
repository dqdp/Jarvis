# Personal Assistant Runtime — Phase 1 MVP Documentation Package

Версия: `mvp-0.1`
Дата: 2026-05-29
Статус: **Phase 1 Core Daemon MVP complete. Post-MVP Alpha work is now active.**

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
  28_mvp_acceptance_checklist.md
  29_hardening_review_notes.md
  30_mvp_implementation_archive.md
  31_mvp_release_notes.md
  32_known_limitations.md
  33_alpha_model_behavior_smoke.md
  34_post_mvp_roadmap.md
  35_post_mvp_adr_backlog.md
  36_post_mvp_plan_review.md
  37_post_mvp_tdd_slices_plan.md
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
  ADR-029_capability_and_permission_model.md
  ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
  ADR-031_agent_loop_strategy_architecture.md
  ADR-033_shell_sandbox_and_local_command_policy.md
  ADR-034_content_retrieval_subsystem_and_project_docs_rag.md
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
- Memory retrieval is behind `MemoryReadPort`; pgvector remains an adapter path,
  not the Memory contract.
- Graph checkpoints are deferred runtime state, not memory/audit truth.
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

- MVP runtime uses a custom deterministic workflow behind `AgentRuntime`.
- Phase 1 loop = deterministic `memory_augmented_answer`; LangGraph is deferred.
- ReAct is a future loop strategy, not a tool, and is deferred until
  `ToolGatewayPort` and the loop-strategy boundary exist.
- Future loop strategies must declare budgets, capabilities, policy hooks, failure semantics and emitted events.

### Model routing

- ModelRouter is internal module/package in Phase 1.
- Local inference is external and accessed through `local_openai_compatible`,
  `local_embedding` or native Ollama provider adapters behind ModelRouter.
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

The Phase 1 Core Daemon MVP is complete as `mvp-0.1`. The repository contains
the MVP contracts, adapters, runtime workflow, API surface, SSE stream, local
Ollama dogfood profile, CLI and acceptance tests. New work should be treated as
post-MVP Alpha scope unless it fixes the MVP baseline.

Implementation notes:

- The API surface is implemented with FastAPI while preserving the documented
  route contract.
- Memory embeddings are stored in PostgreSQL behind `MemoryReadPort`. The current
  MVP adapter uses portable PostgreSQL arrays and deterministic ranking; pgvector
  remains an adapter-level optimization path, not a runtime/domain dependency.
- Fake model and embedding providers are used for CI; real LLM calls are not
  required for acceptance.
- Runtime dogfood wiring is available through
  `assistant_core.app_factory:create_asgi_app --factory`, `make migrate` and
  `make run`.
- Local Ollama dogfood is available through `config/ollama.yaml`. The current
  local profile uses `qwen3.5:9b` for chat/structured calls and
  `embeddinggemma:latest` for embeddings.
- A thin local CLI is available through `make cli ARGS='...'` or the `jarvis`
  console entrypoint. Running `make cli` without `ARGS` opens an interactive
  chat shell with slash commands.

Future architecture-changing changes still require the relevant ADR update.

Post-MVP planning is tracked in:

- `docs/34_post_mvp_roadmap.md`;
- `docs/35_post_mvp_adr_backlog.md`;
- `docs/36_post_mvp_plan_review.md`.

## Final hardening additions in v16

- README was rewritten as a stable index instead of chronological changelog.
- Missing ADR-016 was added for provider-neutral context representation.
- MVP acceptance checklist was added.
- Agent implementation instructions were added.
- Data model document was normalized to remove accumulated numbering drift.
- Hardening review notes were added.

## Implementation additions in v18

- Phase 1 TDD slices 00-19 were implemented.
- MVP acceptance checklist was completed.
- Documentation acceptance tests were added.
- MVP implementation archive was added.

## Runtime verification additions in v21

- Production app factory was added for config-driven daemon assembly.
- Profile-specific local provider wiring was added for chat, structured and
  embedding profiles.
- OpenAI-compatible embedding calls are supported by the local provider adapter.
- Standard local runtime targets `make migrate` and `make run` were added.

## Local Ollama and CLI additions in v22

- `config/ollama.yaml` wires local Ollama profiles without cloud fallback.
- `qwen3.5:9b` is the current chat/structured model for local dogfood
  verification on the target machine.
- `embeddinggemma:latest` is the initial local embedding model.
- The Ollama adapter passes anti-repeat generation options and cuts off
  repeated-line loops before they run to the full token cap.
- `make models-list`, `make models-pull` and `make cli ARGS='...'` were added
  for local operation.
- `make cli` now opens an interactive chat shell; typing `/` in a TTY shows the
  available slash commands before submission.
- Interactive CLI input supports Up/Down in-session history without persisting
  raw prompts to disk.
