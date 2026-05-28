# ADR-016 — Context Representation Is Provider-Neutral

## Status

Accepted as post-MVP direction.

## Context

Phase 1 ContextAssembler emits text-only model context for a deterministic MVP. Future versions must support richer context management: typed content parts, multimodal references, tool observations, planner-specific context and provider portability.

If `AssembledContext` or `ChatMessage` is modeled as a provider-specific request dictionary, the runtime will become coupled to OpenAI/vLLM/Ollama request formats and will be harder to evolve.

## Decision

Context assembly must produce provider-neutral internal message/context structures.

Phase 1 may use text-only content, but internal schemas must not prevent future typed content parts such as:

```text
text
image_ref
file_ref
audio_ref
tool_result_ref
structured_data_ref
```

Provider-specific conversion belongs inside `ModelRouter` provider adapters.

`AgentRuntime` and `ContextAssembler` must not construct provider-specific request dictionaries.

Advanced context techniques remain behind `ContextAssemblerPort`.

## Rationale

This keeps the agent runtime and context subsystem portable across local OpenAI-compatible providers, future OpenAI API use, multimodal models and voice/tool workflows.

It also preserves the architectural rule that provider details belong to `ModelRouter`, not runtime or context assembly.

## Consequences

Positive:

- provider lock-in avoided;
- multimodal and tool-aware context remain possible;
- ContextAssembler can evolve without changing AgentRuntime;
- provider adapters own provider-specific request formatting.

Trade-offs:

- introduces a canonical internal message model;
- provider adapters must implement conversion;
- Phase 1 text-only implementation must still respect the future schema direction.

## Deferred

- typed content parts implementation;
- multimodal context;
- tool-aware context;
- planner-aware context;
- provider-specific optimization layers.
