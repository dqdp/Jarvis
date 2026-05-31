from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import LoopSelectionMode, LoopSelectionRequest
from assistant_core.domain.models import StructuredModelRequest, StructuredModelResponse
from assistant_core.domain.policy import Capability, PermissionMode
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.runtime.model_intent_classifier import ModelBackedIntentClassifier
from assistant_core.runtime.request_metadata import available_tools_summary


pytestmark = pytest.mark.evaluation

CORPUS_PATH = Path("tests/fixtures/intent_routing/tool_intent_corpus.json")


def test_local_model_tool_intent_routing_corpus_eval() -> None:
    if os.environ.get("JARVIS_RUN_INTENT_ROUTING_CORPUS_EVAL") != "1":
        pytest.skip("set JARVIS_RUN_INTENT_ROUTING_CORPUS_EVAL=1 to run local model eval")

    cases = _limited_cases(_cases())
    router = OllamaStructuredRouter(
        endpoint=os.environ.get("JARVIS_INTENT_ROUTING_EVAL_OLLAMA_URL", "http://127.0.0.1:11434"),
        model=os.environ.get("JARVIS_INTENT_ROUTING_EVAL_MODEL", "qwen3.5:9b"),
        timeout_seconds=int(os.environ.get("JARVIS_INTENT_ROUTING_EVAL_TIMEOUT_SECONDS", "60")),
    )
    failures: list[str] = []
    for case in cases:
        classification = asyncio.run(
            ModelBackedIntentClassifier(router=router).classify(_request(case["text"]))
        )
        expected = case["expected"]
        actual_capabilities = {
            candidate.capability.value for candidate in classification.candidate_capabilities
        }
        actual_tool_names = {
            tool_name
            for candidate in classification.candidate_capabilities
            for tool_name in candidate.tool_names
        }
        if classification.intent_family.value != expected["intent_family"]:
            failures.append(
                f"{case['id']}: intent {classification.intent_family.value} "
                f"!= {expected['intent_family']}"
            )
            continue
        missing = set(expected["capabilities"]) - actual_capabilities
        if missing:
            failures.append(f"{case['id']}: missing capabilities {sorted(missing)}")
        missing_tools = set(expected["tool_names"]) - actual_tool_names
        if missing_tools:
            failures.append(f"{case['id']}: missing tool_names {sorted(missing_tools)}")

    assert failures == []


class OllamaStructuredRouter:
    def __init__(self, *, endpoint: str, model: str, timeout_seconds: int) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": message.role.value,
                    "content": "\n".join(part.text for part in message.content),
                }
                for message in request.messages
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 1024,
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(f"{self._endpoint}/api/chat", json=payload)
            response.raise_for_status()
        content = response.json()["message"]["content"]
        return StructuredModelResponse(value=_json_object(content))


def _json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("model did not return a JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model did not return a JSON object")
    return value


def _request(text: str) -> LoopSelectionRequest:
    return LoopSelectionRequest(
        request_id="request-1",
        conversation_id="conversation-1",
        user_id="user-1",
        requested_mode=LoopSelectionMode.AUTO,
        user_input=text,
        current_message_sensitivity=Sensitivity.PROJECT,
        active_project_namespace="project.personal_assistant",
        working_directory="/tmp/project",
        permission_mode=PermissionMode.DEVELOPER_LOCAL,
        available_capabilities=frozenset(Capability),
        available_tools_summary=available_tools_summary(ConfigLoader(Path("config")).load("test")),
        runtime_budget_summary={},
        metadata={"source": "intent_routing_corpus_eval"},
    )


def _limited_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limit = int(os.environ.get("JARVIS_INTENT_ROUTING_EVAL_LIMIT", "0"))
    return cases[:limit] if limit > 0 else cases


def _cases() -> list[dict[str, Any]]:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return payload["cases"]
