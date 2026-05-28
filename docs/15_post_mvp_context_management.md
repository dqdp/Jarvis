# 15 — Post-MVP Context Management Follow-up

## Purpose

This document records advanced context-management capabilities that are intentionally deferred beyond the Phase 1 MVP.

Phase 1 must remain deterministic and small, but its architecture must not block these future techniques.

## Accepted Phase 1 boundary

Phase 1 includes:

- ContextAssemblerPort as facade;
- deterministic context assembly;
- fixed prompt sections;
- namespace-aware active memory retrieval;
- bounded recent-message window;
- explicit token budget;
- ContextManifest audit;
- no raw full prompt logging by default.

Phase 1 does not include:

- query rewriting by LLM;
- learned reranking;
- rolling summaries;
- compression;
- multimodal context;
- tool-aware context;
- planner-aware context.

## Post-MVP extension themes

### Provider-neutral context representation

The internal message/context representation should remain provider-neutral.

ContextAssembler should produce internal domain messages. ModelRouter should convert them to provider-specific request formats.

Future typed content parts:

```text
text
image_ref
file_ref
audio_ref
tool_result_ref
structured_data_ref
```

### Advanced retrieval

Post-MVP retrieval may add:

- retrieval query rewriting;
- multi-query retrieval;
- hybrid vector/BM25 retrieval;
- namespace-aware reranking;
- importance/recency scoring;
- cross-encoder or LLM reranker.

### Rolling summaries and compression

Post-MVP context assembly may add:

- working conversation summary;
- rolling summary plus recent tail;
- extractive compression;
- model-based compression;
- memory-hit compression;
- tool-observation compression.

Working summaries are current-context artifacts. Long-term summaries are memory records.

### Tool-aware context

When ToolGatewayPort is introduced, ContextAssembler may add:

- available tools;
- tool policies;
- recent tool observations;
- approval state;
- risk constraints.

### Planner-aware context

When planner-executor loops are introduced, ContextAssembler may add:

- goal;
- plan;
- constraints;
- previous attempts;
- open blockers;
- budgets;
- approval constraints.

### Multimodal context

Future context sources may include:

- screenshots;
- PDFs;
- images;
- audio transcripts;
- files;
- structured documents.

This must be introduced through typed content parts and provider-specific conversion in ModelRouter.

## Rule

Advanced context-management capabilities must be implemented behind ContextAssemblerPort and must not require AgentRuntime to manually assemble prompts or provider-specific message payloads.


## ADR link

Provider-neutral context representation is captured in:

```text
ADR-016_context_representation_provider_neutral_post_mvp.md
```
