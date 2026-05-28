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
