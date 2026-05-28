# ADR-015 — Context Assembler as core Phase 1 component

## Status

Accepted.

## Context

Each model call needs a prompt context assembled from recent conversation, active memories, runtime rules, policy constraints, user message and output contract. If Agent Runtime assembles this ad hoc, context management becomes tightly coupled to memory, storage and prompt implementation details.

## Decision

Phase 1 introduces `ContextAssemblerPort` as a first-class core component.

Agent Runtime must call ContextAssembler and must not manually construct prompts from lower-level stores.

Phase 1 ContextAssembler is deterministic-first:

- recent conversation window;
- namespace-aware retrieval of active memories;
- stable prompt sections;
- explicit token budget;
- audit metadata for used memories/messages.

Advanced context planning, LLM-generated retrieval queries, learned reranking, compression, rolling summaries and multimodal context are deferred.

Accepted Phase 1 constraints:

- do not persist raw full prompt by default; persist ContextManifest instead;
- use current user message plus minimal recent-turn context as retrieval text;
- do not require reranker in MVP;
- do not require automatic rolling summaries in MVP;
- keep context assembly deterministic-first and testable.

## Consequences

- Current context is separated from long-term memory.
- Context assembly implementation can evolve without changing agent loop contract.
- ContextAssembler requires contract tests like other replaceable subsystems.
- Prompt construction becomes auditable and deterministic enough for MVP.

## Implementation notes

The ContextAssembler facade must remain stable even if later implementations add reranking, query rewriting, compression, rolling summaries, tool-aware context, or multimodal context. These capabilities are internal strategy changes, not AgentRuntime contract changes.
