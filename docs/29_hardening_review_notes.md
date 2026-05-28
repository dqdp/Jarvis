# 29 — Hardening Review Notes

## Status

Reviewed baseline v16.

## Review summary

The Phase 1 MVP documentation package is internally coherent and ready for coding-agent analysis.

The package now has:

- stable README/index;
- complete ADR numbering including ADR-016;
- explicit MVP acceptance checklist;
- AGENTS.md implementation rules;
- normalized Data Model document;
- clear MVP/post-MVP boundaries.

## Hardening changes made in v16

1. Rewrote README from chronological changelog into stable package index and decision summary.
2. Added ADR-016 for provider-neutral context representation.
3. Added MVP Acceptance Checklist.
4. Added AGENTS.md for coding agents.
5. Normalized `08_data_model_and_storage.md` and removed accumulated numbering drift.
6. Added this hardening review note.

## Remaining expected revisions

These are not blockers for MVP documentation handoff:

- slice plan may be revised after coding-agent repository/dependency analysis;
- implementation details may refine database technology choices;
- config file split may evolve from single YAML to multiple YAML files;
- local model names/endpoints are placeholders and deployment-specific;
- future post-MVP documents will be needed for tools, RAG, voice, scheduler and real agent loops.

## Handoff status

Ready for initial coding-agent analysis and TDD implementation planning.


## v17 TDD clarification

`AGENTS.md` was strengthened with an explicit red-green-refactor workflow, red-phase requirements, green-phase scope limits, test modification rules and slice definition-of-done.

## v18 implementation alignment

After the TDD implementation pass, the documentation package was updated from
pre-implementation handoff status to implemented MVP status.

One implementation-level storage adjustment is now explicit:

- the MVP memory retrieval adapter stores embeddings in PostgreSQL numeric
  arrays and uses deterministic ranking behind `MemoryReadPort`; pgvector
  remains a replaceable adapter path, not a runtime/domain dependency.

This adjustment preserves the accepted architecture boundaries and does not add
post-MVP scope.

## v19 FastAPI adapter alignment

The temporary minimal ASGI adapter was replaced with the intended FastAPI API
adapter. API/SSE/e2e contract tests now use `httpx.ASGITransport` against the
FastAPI app.

## v20 MVP hardening closure

The post-review hardening items were implemented in order:

- request execution starts after message submission, not when SSE is opened;
- SSE reconnect subscribes/replays without re-running the provider and emits
  heartbeat events while waiting;
- explicit request cancellation is implemented;
- `client_message_id` replay handles concurrent insert races;
- `memory.retrieved`, context causation and policy-decision audit are recorded;
- ContextAssembler uses `PolicyPort` for context inclusion;
- API bodies use strict FastAPI/Pydantic schemas with sanitized validation
  errors;
- destructive test DB helpers assert an explicit local test database;
- runtime context/model timeouts, readiness health and serialized golden
  ContextAssembler fixtures are covered by tests.

## v21 final review closure

The final consistency review follow-ups were closed without expanding MVP
scope:

- migration 0006 now rejects assistant messages whose `request_id` is missing
  or different from the owning assistant request;
- SSE live/replay output is projected through explicit public event DTOs;
- memory retrieval applies configured sensitivity exclusions and `min_score`;
- architecture tests guard against accidental MVP scope-creep packages for
  tools, MCP, RAG, ReAct, planner and voice;
- cancellation docs now state that already terminal requests remain unchanged.
