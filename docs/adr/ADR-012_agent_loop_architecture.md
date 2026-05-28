# ADR-012: Agent loop architecture

Status: Accepted

## Context

Need decide whether Phase 1 uses ReAct, custom workflow or various schemes.

## Decision

Phase 1 uses deterministic memory-augmented workflow. LangGraph is substrate. ReAct deferred until ToolGatewayPort. Future loops are Loop Strategies with budgets, stopping conditions and policy hooks.

## Consequences

Predictable MVP and safer future tool loops. Requires explicit runtime loop documentation.
