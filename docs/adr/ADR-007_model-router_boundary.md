# ADR-007: Model-router boundary

Status: Accepted

## Context

Runtime must not depend on vLLM/Ollama/OpenAI directly.

## Decision

Use internal ModelRouter module behind ModelRouterPort. Local inference node is external process.

## Consequences

Fast MVP with clean boundary. Router can become service later.
