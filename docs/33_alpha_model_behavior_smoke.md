# 33 — Alpha Model Behavior Smoke

## Purpose

This document defines manual dogfood checks for the local model profile. These
checks are not CI gates and must not require real LLM calls in automated tests.

The golden contract for the default prompt is covered by
`tests/golden/test_context_assembler_golden.py`.

## Profile

Default interactive profile:

```text
local_main
```

Expected local dogfood model:

```text
qwen3.5:9b
```

## Smoke Scenarios

Run through the CLI after starting the daemon with the Ollama config profile.

```text
CONFIG_PROFILE=ollama make run
make cli
```

### 1. Language Selection

Prompt:

```text
объясни простыми словами, что ты умеешь
```

Expected:

- answer is in Russian;
- answer is concise;
- answer does not spend the first paragraph apologizing or explaining generic
  local-model limitations.

### 2. Casual Answer Loop Resistance

Prompt:

```text
расскажи короткий анекдот
```

Expected:

- one short joke;
- no repeated dialogue loop;
- generation stops naturally before the token budget.

### 3. Direct Useful Answer

Prompt:

```text
составь список из трех дел на вечер
```

Expected:

- exactly or approximately three actionable items;
- no hidden-context disclosure;
- no generic safety disclaimer.

### 4. Memory Recall

Commands:

```text
/memory add Я предпочитаю короткие ответы.
напомни, какой стиль ответов я предпочитаю
```

Expected:

- answer uses the stored memory;
- answer does not reveal internal memory IDs or raw context formatting.

### 5. Cancellation

Prompt:

```text
напиши очень длинный рассказ на 5000 слов
```

Action:

```text
Ctrl-C
```

Expected:

- CLI prints a server cancellation line;
- the next prompt remains usable;
- `/status` still returns `ready`.

## Failure Handling

If a scenario fails, prefer tuning in this order:

1. prompt contract in `DeterministicContextAssembler`;
2. local model profile parameters;
3. provider adapter stop/cancellation behavior.

Do not add cloud fallback to fix local answer quality.
