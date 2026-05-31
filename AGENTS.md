# AGENTS.md — Phase 1 Implementation Instructions

## Scope

You are implementing Phase 1 Core Daemon for a local-first personal assistant runtime.

Read first:

```text
docs/00_project_charter.md
docs/01_target_architecture_overview.md
docs/26_testing_strategy.md
docs/27_tdd_implementation_slices_plan.md
```

## Non-negotiable rules

- Use TDD as the default implementation method.
- Do not write production code for a slice before writing or updating the relevant tests.
- Do not bypass ports/adapters boundaries to make tests pass.
- Do not implement tools, MCP, RAG, ReAct, planner-executor or voice in MVP.
- Do not enable cloud fallback.
- Do not store secrets in memory, raw logs or prompts.
- Do not log raw full prompts by default.
- Do not make real LLM calls required for CI.
- Use fake model/embedding providers for tests.
- Do not let facade modules become implementation dumps.
- Do not ignore obviously growing files, classes or methods when a slice touches them.
- If implementation order must change, update `docs/27_tdd_implementation_slices_plan.md`.
- If architecture must change, update the relevant ADR.

## TDD workflow

Every implementation slice must follow the same TDD loop:

```text
1. Read the relevant docs and ADRs.
2. Identify the smallest behavior or boundary to implement.
3. Write or update tests first.
4. Run the tests and confirm they fail for the expected reason.
5. Implement the smallest production change that makes the tests pass.
6. Run the relevant test group again.
7. Refactor only after the tests are green.
8. Run architecture/contract tests before declaring the slice complete.
```

A slice is not complete just because the visible feature works. It is complete only when the required tests for that slice are green.

## What “tests first” means

Before production code, create or update the appropriate tests:

```text
unit tests:
  pure logic, validation, state transitions, policy decisions

contract tests:
  required behavior of replaceable ports/adapters

integration tests:
  real PostgreSQL / migrations / storage adapters

golden tests:
  deterministic ContextAssembler output and ContextManifest

architecture tests:
  forbidden imports and boundary violations

e2e tests:
  full user-turn lifecycle with fake model providers
```

For each slice, use `docs/27_tdd_implementation_slices_plan.md` as the source of required tests.

## Red phase requirements

When adding tests, the first run must fail for the expected reason.

Good failure:

```text
ImportError because the port does not exist yet.
AssertionError because behavior is not implemented yet.
Contract test fails because adapter does not satisfy the port yet.
```

Bad failure:

```text
test typo
broken fixture
wrong import path
unclear assertion
test depends on real LLM/network
```

Do not proceed to production code until test failures are meaningful.

## Green phase requirements

Implement only the minimal code needed to satisfy the failing tests.

Do not use the green phase to add:

```text
extra abstractions
new framework layers
unsupported endpoints
new model providers
tools/MCP/RAG/ReAct
cloud fallback
unrequested background workers
```

If extra work seems necessary, stop and update the relevant document or propose a slice-plan change.

## Refactor phase requirements

Refactoring is allowed only after tests are green.

During refactor:

```text
keep public contracts stable;
do not weaken tests;
do not remove architecture guardrails;
do not replace explicit ports with direct adapter calls;
do not broaden MVP scope.
```

After refactor, rerun the same tests.

## Test modification rules

Tests are implementation contracts.

Do not delete or weaken tests to make production code pass.

Changing tests is allowed only when:

```text
the accepted design changed;
the test was incorrect;
the slice plan changed;
the test encoded implementation detail instead of contract.
```

If a test change reflects an architecture change, update the relevant ADR.

If a test change only reflects implementation order, update `docs/27_tdd_implementation_slices_plan.md`.

## Required test discipline

For every slice, run the smallest relevant test group first, then broader tests.

Typical order:

```text
unit / contract / golden for the slice
integration if storage is touched
architecture tests if imports/modules are touched
e2e smoke test before declaring runtime/API slices complete
```

Do not rely on manual testing or real LLM behavior as proof of correctness.

## Review agent workflow

Review agents are for independent code and architecture review, not for
duplicating the verification run.

Start review agents only after the relevant tests and verification for the
slice have already passed. The review-agent prompt must explicitly say:

```text
Tests are already green.
Do not run tests.
Do not edit files.
Perform read-only review only.
Focus on correctness, architecture, contracts, regressions, security/privacy,
operability and missing coverage.
Report findings with severity P0/P1/P2/P3 and concrete file/line references.
```

If a review agent believes an additional test is required, it should report the
gap and the exact recommended test command or case. It must not run the test
unless the user explicitly asks for a verification agent rather than a review
agent.

Relevant P0/P1 findings block the slice. P2/P3 findings may be fixed in the
current slice or recorded as follow-up work, depending on scope and risk.

## Milestone workflow

For long-running goals split into sequential milestones or implementation
slices, use this gate after every stage:

```text
1. Complete the stage using the normal TDD workflow.
2. Run the required tests and verification for that stage.
3. Fix test or verification failures until the stage is green.
4. Start two independent review agents from scratch with the review-agent
   prompt above.
5. If there are relevant P0/P1 findings, fix them and repeat verification plus
   review for the affected stage.
6. If there are no relevant P0/P1 findings, commit the completed stage.
7. Move to the next stage only after the commit.
```

Do not rerun review agents just because they found P2/P3 issues. Rerun review
agents only after fixing relevant P0/P1 findings or when the user explicitly
asks for another review pass.

## Architecture guardrails

AgentRuntime may depend on:

```text
ContextAssemblerPort
ModelRouterPort
EventLogPort
ConversationStorePort
PolicyPort
domain schemas
```

AgentRuntime must not import:

```text
PostgreSQL adapters
SQLAlchemy models
pgvector
vLLM/Ollama/OpenAI clients
provider-specific request dictionaries
```

ContextAssembler must not call provider-specific model clients.

Memory subsystem must not store document chunks.

ModelRouter must consult PolicyPort before provider calls.

## Facade and complexity discipline

Every implementation slice must keep architectural responsibilities explicit.

Watch for these design smells while editing code:

```text
large files that keep accumulating unrelated behavior;
classes that act as facade, service, adapter and serializer at the same time;
methods that combine validation, orchestration, persistence, policy and formatting;
API/CLI entrypoints that contain runtime or business logic;
storage adapters that own application workflows;
tool adapters that own approval, policy or audit lifecycle;
provider adapters that import router/facade implementation details;
tests that become mini-frameworks because production boundaries are unclear.
```

Facade modules are allowed to coordinate dependencies, but they must stay thin.
They should delegate real work to explicit services, ports, adapters, presenters
or lifecycle components.

When a touched file, class or method is already large or gains a second
responsibility, choose one of these actions before finishing the slice:

```text
extract the responsibility behind a clear boundary;
add or update an architecture test that protects the intended boundary;
record a follow-up in the slice plan if extraction would exceed the current slice;
update the relevant ADR if the boundary decision changed.
```

Do not add new feature behavior to a known god-module without first checking
whether the change belongs in a new service/module.

## Implementation method per slice

For each slice:

1. Read relevant docs and ADR.
2. Write failing tests first.
3. Implement the minimal production code.
4. Run unit/contract/integration/golden/architecture tests as appropriate.
5. Do not expand scope unless docs/ADR are updated.

## Slice plan

Follow:

```text
docs/27_tdd_implementation_slices_plan.md
```

The plan is provisional and may be revised after repository/dependency analysis.

## Reporting required for proposed changes

If you propose a slice plan change, include:

```text
Proposed change
Reason
Affected docs/ADR
Risk
Changed tests
Whether architecture decision changes
```

## Definition of done for a slice

A slice is done only when:

```text
required tests were written first;
tests failed for the expected reason before implementation;
production code is minimal and scoped;
all relevant tests are green;
architecture guardrails still pass;
no MVP scope expansion occurred;
docs/ADR were updated if the design changed.
```
