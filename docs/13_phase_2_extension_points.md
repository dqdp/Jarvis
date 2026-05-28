# 13 — Phase 2 Extension Points

## 1. Tool Gateway

Future ToolGatewayPort will handle:

- MCP;
- Linux shell sandbox;
- search;
- Spotify;
- Telegram;
- filesystem;
- home automation.

All tool calls require:

- policy check;
- audit;
- bounded execution;
- optional approval.

## 2. ReAct Tool Loop

Introduced only after ToolGatewayPort.

Must define:

- max_steps;
- max_tool_calls;
- max_wall_time;
- allowed_tools;
- approval rules;
- audit format.

## 3. Planner-Executor

For multi-step tasks requiring planning, approvals or longer execution.

## 4. Scheduler/Event Bus

Likely Phase 2 introduces NATS JetStream or equivalent.

Uses:

- sleep/reflection;
- reminders;
- proactive checks;
- long-running tasks.

## 5. Voice

Voice Gateway:

- wake word;
- VAD;
- STT;
- TTS;
- optional realtime cloud model;
- local-first audio policy.

## 6. Advanced Memory

- automatic extraction into memory_candidates;
- approval/rejection UI;
- consolidation jobs;
- hybrid retrieval;
- reranking;
- separate memory service if needed.


## Post-MVP RAG / Content Retrieval

Full RAG is deferred beyond Phase 1.

It must be introduced as a separate Content Retrieval subsystem, not as an extension of Memory records.

Future port:

```text
ContentRetrievalPort
```

First target:

```text
Project Documentation RAG
```

Initial corpus:

```text
docs/*.md
ADR documents
README
architecture docs
```

Boundary rules:

```text
Memory subsystem must not store document chunks.
Event log must not be default vector corpus.
Conversation history must not be vector-indexed in MVP.
RAG requires source registry, chunk model and citation model.
```


## Agent loop follow-ups

Post-MVP real agent loops must be introduced through explicit loop strategies.

Candidate loop strategies:

```text
tool_react_loop
planner_executor_loop
approval_gated_loop
sleep_consolidation_loop
maintenance_workflow
```

Each strategy must declare budgets, allowed capabilities, policy hooks, stopping conditions and emitted events.

Unbounded free-running agent loops are not allowed.
