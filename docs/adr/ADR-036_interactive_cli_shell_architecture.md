# ADR-036 — Interactive CLI Shell Architecture

## Status

Accepted.

Date: 2026-05-31

## Context

PM-09 voice must build on a text surface that already feels usable for normal
assistant work. The current CLI has a thin interactive chat loop with slash
commands, in-session history, approvals and cancellation, but the terminal UX is
still closer to a line-oriented smoke client than to the intended dogfood
interface.

Before voice starts, the interactive shell needs Codex-like behavior:

- a live status line;
- visible activity while requests are running;
- slash command discovery that updates while the user types;
- clean rendering for chat, tools, approvals, cancellation and errors;
- predictable keyboard behavior after interruption;
- deterministic non-TTY behavior for tests and scripts.

This is PM-08i: terminal UX hardening after PM-08h and before PM-09.

## Decision

Use `prompt_toolkit` as the interactive TTY input/rendering substrate, with a
thin Jarvis-owned renderer for assistant-specific transcript and stream events.

The CLI remains an inline shell. PM-08i does not introduce a full-screen TUI,
alternate screen layout, panes, session dashboards or Textual app.

The CLI remains a client-only boundary:

- it may call existing Jarvis HTTP client methods;
- it must not import runtime selectors, ToolGateway, storage adapters, model
  providers or concrete tool adapters;
- it must not duplicate backend loop-selection or policy decisions;
- it renders public stream events and local shell state only.

## Rationale

`prompt_toolkit` directly provides the interactive primitives PM-08i needs:

- `PromptSession` and in-memory history;
- completers and completion menus;
- bottom toolbar/status-line rendering;
- async prompt support;
- key bindings for Ctrl-C, Ctrl-D, Tab, arrows and Esc.

Continuing to build this behavior with manual ANSI/raw terminal control would
make completion menus, cursor behavior and status-line invalidation fragile.
Textual would provide richer widgets, but would turn PM-08i into a full TUI
product rather than the narrow pre-voice shell hardening slice.

Rich remains useful as a future rendering option, but PM-08i does not add it.
The status/activity behavior should be implemented through `prompt_toolkit`
toolbar and prompt invalidation.

## Shell Behavior

TTY mode uses a `prompt_toolkit`-backed reader. Non-TTY mode keeps the existing
deterministic line-oriented reader. A global `--plain` flag forces the
line-oriented reader even on a TTY.

The shell status line shows:

- user-facing mode: `auto`, `chat` or `tools`;
- daemon readiness summary when known;
- current conversation summary when known;
- active request phase;
- model/profile summary when known;
- working-directory scope in a redacted, bounded form.

Activity indication is phase-based, not percentage-based. The shell may show
submitting, selecting, assembling context, retrieving context, running a tool,
waiting for approval, streaming, cancelled, failed or done. It must not invent
fake progress percentages.

Slash command discovery updates while the user types `/...`, filters commands,
shows descriptions and argument hints, supports keyboard selection/completion
and closes without submitting on Esc.

Transcript rendering stays line-oriented and append-only enough for terminal
scrollback. Tool, approval and cancellation output should be readable first and
raw JSON never becomes the normal user-facing format.

## Privacy

Interactive history is in-memory only.

The shell must not persist:

- raw user prompts;
- `/memory add` content;
- secret-sensitivity input;
- raw approval payloads;
- raw full prompts or provider payloads.

Status/toolbars must not leak full secret-like paths. Working-directory display
must be bounded and redacted enough for normal terminal sharing.

## Consequences

Add `prompt_toolkit>=3.0` as a runtime dependency.

PM-08i adds CLI-internal abstractions such as:

- `SlashCommandRegistry`;
- `SlashCommandDefinition`;
- `ShellActivityState`;
- `PromptToolkitLineReader`;
- status-line renderer helpers.

These abstractions are owned by the CLI package and are not domain/runtime
contracts.

PM-09 voice depends on PM-08i completion. Voice remains a separate client/channel
over the existing runtime and must not inherit separate routing logic from the
terminal shell.

## Test Requirements

PM-08i must be implemented TDD-first.

Required tests:

- slash command filtering, descriptions, argument hints and completion;
- status-line rendering, truncation and path redaction;
- activity state transitions from fake stream events;
- TTY `prompt_toolkit` reader selection;
- `--plain` and non-TTY fallback;
- Ctrl-C, `/cancel`, approval prompt and post-cancel prompt recovery;
- secret/history filtering;
- architecture guards proving CLI does not import runtime selector, ToolGateway,
  storage adapters or model provider clients, and `prompt_toolkit` imports stay
  inside CLI shell modules.

No PM-08i test may require a real LLM, real shell command, host diagnostics
command, microphone, speaker or cloud service.
