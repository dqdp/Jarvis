from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import (
    CapabilityCandidate,
    IntentClassification,
    IntentFamily,
    LoopSelectionMode,
    LoopSelectionRequest,
    SelectionFallbackPreference,
)
from assistant_core.domain.models import StructuredModelRequest, StructuredModelResponse
from assistant_core.domain.policy import Capability, PermissionMode, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.runtime.loop_selection import DeterministicIntentClassifier
from assistant_core.runtime.model_intent_classifier import ModelBackedIntentClassifier
from assistant_core.runtime.request_resolver import RequestResolverIntentClassifier
from assistant_core.runtime.routing import CapabilityRoutingRegistry


pytestmark = pytest.mark.unit


def _default_available_tools_summary() -> tuple[dict, ...]:
    settings = ConfigLoader(Path("config")).load("test")
    return CapabilityRoutingRegistry.from_settings(settings).available_tools_summary()


def test_model_backed_classifier_maps_structured_payload_to_intent_classification() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "system_diagnostics",
            "confidence": 0.91,
            "candidate_capabilities": [
                {
                    "capability": "tool.system.read.hardware",
                    "intent_family": "system_diagnostics",
                    "confidence": 0.91,
                    "requires_live_state": True,
                    "requires_execution": True,
                    "requires_write": False,
                    "tool_names": ["tool.system.read.hardware"],
                    "risk_classes": ["read_only"],
                    "scope_hint": "os_version",
                    "evidence_codes": ["system_os_version_request"],
                }
            ],
            "requires_live_state": True,
            "requires_execution": True,
            "answer_without_tools_would_be_misleading": True,
            "reason_code": "model_system_diagnostics",
            "fallback_preference": "fail_unavailable",
        }
    )

    classification = asyncio.run(ModelBackedIntentClassifier(router=router).classify(_request()))

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.confidence == 0.91
    assert classification.requires_live_state is True
    assert classification.answer_without_tools_would_be_misleading is True
    assert classification.reason_code == "model_system_diagnostics"
    assert classification.classification_source == "model"
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SYSTEM_READ_HARDWARE
    assert candidate.tool_names == ("tool.system.read.hardware",)
    assert candidate.risk_classes == frozenset({RiskClass.READ_ONLY})
    assert candidate.scope_hint == "os_version"


def test_model_backed_classifier_sends_constrained_schema_to_model_router() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "ordinary_chat",
            "confidence": 0.82,
            "candidate_capabilities": [],
            "requires_live_state": False,
            "requires_execution": False,
            "answer_without_tools_would_be_misleading": False,
            "reason_code": "model_ordinary_chat",
            "fallback_preference": "chat",
        }
    )

    asyncio.run(
        ModelBackedIntentClassifier(router=router).classify(
            _request(
                user_input="hello",
                available_tools_summary=(
                    {
                        "tool_name": "tool.system.read.hardware",
                        "capability": "tool.system.read.hardware",
                        "description": "read hardware and operating system metadata",
                    },
                ),
            )
        )
    )

    request = router.requests[0]
    assert request.profile == "local_structured"
    assert request.request_id == "request-1"
    assert request.conversation_id == "conversation-1"
    assert request.schema["required"] == [
        "intent_family",
        "confidence",
        "candidate_capabilities",
        "requires_live_state",
        "requires_execution",
        "answer_without_tools_would_be_misleading",
        "reason_code",
        "fallback_preference",
    ]
    assert "intent_family" in request.schema["properties"]
    assert "candidate_capabilities" in request.schema["properties"]
    assert request.messages[0].role.value == "system"
    assert request.messages[-1].role.value == "user"
    system_text = request.messages[0].content[0].text
    assert "intent_family" in system_text
    assert "candidate_capabilities" in system_text
    assert "current local machine/system state" in system_text
    assert "operating system version/build" in system_text
    assert "tool.system.read.hardware" in system_text
    assert "read hardware and operating system metadata" in system_text


def test_model_backed_classifier_rejects_raw_command_tool_names() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "system_diagnostics",
            "confidence": 0.91,
            "candidate_capabilities": [
                {
                    "capability": "tool.system.read.hardware",
                    "intent_family": "system_diagnostics",
                    "confidence": 0.91,
                    "requires_live_state": True,
                    "requires_execution": True,
                    "requires_write": False,
                    "tool_names": ["sysctl -n hw.logicalcpu"],
                    "risk_classes": ["read_only"],
                    "evidence_codes": ["raw_command"],
                }
            ],
            "requires_live_state": True,
            "requires_execution": True,
            "answer_without_tools_would_be_misleading": True,
            "reason_code": "model_system_diagnostics",
            "fallback_preference": "fail_unavailable",
        }
    )

    classification = asyncio.run(ModelBackedIntentClassifier(router=router).classify(_request()))

    assert classification.intent_family is IntentFamily.UNKNOWN
    assert classification.confidence == 0.0
    assert classification.candidate_capabilities == ()
    assert classification.reason_code == "classifier_unavailable"
    assert classification.fallback_preference is SelectionFallbackPreference.FAIL_UNAVAILABLE


def test_model_backed_classifier_falls_back_to_deterministic_classifier_on_router_error() -> None:
    router = FakeStructuredRouter(RuntimeError("model unavailable"))

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(user_input="Какая версия операционной системы?"))
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities[0].capability is Capability.TOOL_SYSTEM_READ_HARDWARE
    assert classification.candidate_capabilities[0].scope_hint == "os_version"


def test_model_backed_classifier_guardrail_applies_after_invalid_model_payload_fallback() -> None:
    router = FakeStructuredRouter({"intent_family": "ordinary_chat"})

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(user_input="Какой у меня macOS билд?"))
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.reason_code == "local_system_state_guardrail"
    assert classification.candidate_capabilities[0].scope_hint == "os_version"


def test_model_backed_classifier_guardrail_corrects_local_os_build_false_negative() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "ordinary_chat",
            "confidence": 0.76,
            "candidate_capabilities": [],
            "requires_live_state": False,
            "requires_execution": False,
            "answer_without_tools_would_be_misleading": False,
            "reason_code": "ordinary_chat",
            "fallback_preference": "chat",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(router=router).classify(
            _request(user_input="Какой у меня macOS билд?")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.reason_code == "local_system_state_guardrail"
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SYSTEM_READ_HARDWARE
    assert candidate.tool_names == ("tool.system.read.hardware",)
    assert candidate.scope_hint == "os_version"


@pytest.mark.parametrize(
    ("user_input", "capability", "tool_name", "scope_hint"),
    [
        (
            "Текущая температура процессора.",
            Capability.TOOL_SYSTEM_READ_SENSORS,
            "tool.system.read.sensors",
            None,
        ),
        (
            "Сколько памяти сейчас свободно в системе?",
            Capability.TOOL_SYSTEM_READ_RESOURCES,
            "tool.system.read.resources",
            None,
        ),
        (
            "Сколько свободного места на диске?",
            Capability.TOOL_SYSTEM_READ_RESOURCES,
            "tool.system.read.resources",
            "disk_free",
        ),
        (
            "Сколько процентов заряда аккумулятора осталось на макбуке?",
            Capability.TOOL_SYSTEM_READ_HARDWARE,
            "tool.system.read.hardware",
            "battery_charge",
        ),
        (
            "Включен ли VPN сейчас?",
            Capability.TOOL_SYSTEM_READ_NETWORK,
            "tool.system.read.network",
            "vpn_status",
        ),
        (
            'Запущен ли сейчас процесс, в имени которого есть "HFT"?',
            Capability.TOOL_SYSTEM_READ_PROCESS,
            "tool.system.read.process",
            "process_name_search",
        ),
    ],
)
def test_model_backed_classifier_fallback_corrects_live_state_false_negative(
    user_input: str,
    capability: Capability,
    tool_name: str,
    scope_hint: str | None,
) -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "ordinary_chat",
            "confidence": 0.92,
            "candidate_capabilities": [],
            "requires_live_state": False,
            "requires_execution": False,
            "answer_without_tools_would_be_misleading": False,
            "reason_code": "ordinary_chat",
            "fallback_preference": "chat",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(user_input=user_input))
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is capability
    assert candidate.tool_names == (tool_name,)
    assert candidate.scope_hint == scope_hint


@pytest.mark.parametrize(
    ("user_input", "tool_name"),
    [
        ("Посчитай 128 * 64", "calculator.evaluate"),
        ("Check the assistant daemon status", "daemon.status"),
    ],
)
def test_model_backed_classifier_fallback_corrects_safe_builtin_false_negative(
    user_input: str,
    tool_name: str,
) -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "ordinary_chat",
            "confidence": 0.92,
            "candidate_capabilities": [],
            "requires_live_state": False,
            "requires_execution": False,
            "answer_without_tools_would_be_misleading": False,
            "reason_code": "ordinary_chat",
            "fallback_preference": "chat",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(user_input=user_input))
    )

    assert classification.intent_family is IntentFamily.SAFE_BUILTIN_TOOL
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SAFE
    assert candidate.tool_names == (tool_name,)


def test_model_backed_classifier_prefers_fallback_for_allowlisted_direct_tool_intent() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "system_diagnostics",
            "confidence": 0.93,
            "candidate_capabilities": [
                {
                    "capability": "tool.system.read.hardware",
                    "intent_family": "system_diagnostics",
                    "confidence": 0.93,
                    "requires_live_state": True,
                    "requires_execution": True,
                    "requires_write": False,
                    "tool_names": [],
                    "risk_classes": ["read_only"],
                    "scope_hint": None,
                    "evidence_codes": ["model_system_request"],
                }
            ],
            "requires_live_state": True,
            "requires_execution": True,
            "answer_without_tools_would_be_misleading": True,
            "reason_code": "system_diagnostics",
            "fallback_preference": "fail_unavailable",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(user_input="Какая версия операционной системы?"))
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.reason_code == "system_diagnostics_hint"
    candidate = classification.candidate_capabilities[0]
    assert candidate.tool_names == ("tool.system.read.hardware",)
    assert candidate.scope_hint == "os_version"


def test_model_backed_classifier_calls_model_for_medium_confidence_direct_intent() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "ordinary_chat",
            "confidence": 0.92,
            "candidate_capabilities": [],
            "requires_live_state": False,
            "requires_execution": False,
            "answer_without_tools_would_be_misleading": False,
            "reason_code": "ordinary_chat",
            "fallback_preference": "chat",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(user_input="Какая версия операционной системы?"))
    )

    assert router.requests
    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.reason_code == "system_diagnostics_hint"
    assert classification.candidate_capabilities[0].scope_hint == "os_version"


def test_model_backed_classifier_short_circuits_high_confidence_ordinary_chat_fallback() -> None:
    router = FakeStructuredRouter(RuntimeError("model should not be called"))

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(user_input="Расскажи, как решаются кубические уравнения."))
    )

    assert router.requests == []
    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.reason_code == "ordinary_chat_explicit_hint"


def test_model_backed_classifier_short_circuits_direct_answer_prompt() -> None:
    router = FakeStructuredRouter(RuntimeError("model should not be called"))

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(user_input="Ответь ровно одним словом: OK"))
    )

    assert router.requests == []
    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.reason_code == "ordinary_chat_explicit_hint"


def test_model_backed_classifier_does_not_short_circuit_default_ordinary_fallback() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "project_inspection",
            "confidence": 0.88,
            "candidate_capabilities": [
                {
                    "capability": "tool.shell.read",
                    "intent_family": "project_inspection",
                    "confidence": 0.88,
                    "requires_live_state": True,
                    "requires_execution": True,
                    "requires_write": False,
                    "tool_names": ["tool.shell.read.project"],
                    "risk_classes": ["read_only"],
                    "scope_hint": None,
                    "evidence_codes": ["model_project_lookup"],
                }
            ],
            "requires_live_state": True,
            "requires_execution": True,
            "answer_without_tools_would_be_misleading": True,
            "reason_code": "model_project_inspection",
            "fallback_preference": "fail_unavailable",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(user_input="Please examine the repository layout"))
    )

    assert router.requests
    assert classification.intent_family is IntentFamily.PROJECT_INSPECTION
    assert classification.reason_code == "model_project_inspection"


def test_model_backed_classifier_does_not_short_circuit_at_fast_path_threshold() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "project_inspection",
            "confidence": 0.88,
            "candidate_capabilities": [
                {
                    "capability": "tool.shell.read",
                    "intent_family": "project_inspection",
                    "confidence": 0.88,
                    "requires_live_state": True,
                    "requires_execution": True,
                    "requires_write": False,
                    "tool_names": ["tool.shell.read.project"],
                    "risk_classes": ["read_only"],
                    "scope_hint": None,
                    "evidence_codes": ["model_project_lookup"],
                }
            ],
            "requires_live_state": True,
            "requires_execution": True,
            "answer_without_tools_would_be_misleading": True,
            "reason_code": "model_project_inspection",
            "fallback_preference": "fail_unavailable",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=StaticOrdinaryChatClassifier(confidence=0.9),
        ).classify(_request(user_input="Ответь ровно одним словом: OK"))
    )

    assert router.requests
    assert classification.intent_family is IntentFamily.PROJECT_INSPECTION


def test_app_factory_uses_request_resolver_as_runtime_default(monkeypatch) -> None:
    monkeypatch.setenv(
        "JARVIS_LOOP_SELECTION__DETERMINISTIC_FAST_PATH_THRESHOLD",
        "0.91",
    )
    from assistant_core.app_factory import build_intent_classifier

    settings = ConfigLoader(Path("config")).load("test")
    classifier = build_intent_classifier(
        settings=settings,
        router=FakeStructuredRouter(RuntimeError("model should not be called")),
    )

    classification = asyncio.run(
        classifier.classify(_request(user_input="Расскажи, как решаются кубические уравнения."))
    )

    assert isinstance(classifier, RequestResolverIntentClassifier)
    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.classification_source == "request_resolver"


def test_model_backed_classifier_does_not_short_circuit_generic_fallback_tool_hint() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "ordinary_chat",
            "confidence": 0.9,
            "candidate_capabilities": [],
            "requires_live_state": False,
            "requires_execution": False,
            "answer_without_tools_would_be_misleading": False,
            "reason_code": "ordinary_chat",
            "fallback_preference": "chat",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=router,
            fallback=StaticClassifier(
                intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
                capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
                tool_name="tool.system.read.hardware",
                scope_hint=None,
            ),
        ).classify(_request(user_input="Tell me about the system"))
    )

    assert router.requests
    assert classification.intent_family is IntentFamily.ORDINARY_CHAT


def test_model_backed_classifier_rejects_non_boolean_top_level_flags() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "ordinary_chat",
            "confidence": 0.76,
            "candidate_capabilities": [],
            "requires_live_state": "false",
            "requires_execution": False,
            "answer_without_tools_would_be_misleading": False,
            "reason_code": "ordinary_chat",
            "fallback_preference": "chat",
        }
    )

    classification = asyncio.run(ModelBackedIntentClassifier(router=router).classify(_request()))

    assert classification.intent_family is IntentFamily.UNKNOWN
    assert classification.reason_code == "classifier_unavailable"


def test_model_backed_classifier_rejects_non_boolean_candidate_flags() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "system_diagnostics",
            "confidence": 0.91,
            "candidate_capabilities": [
                {
                    "capability": "tool.system.read.hardware",
                    "intent_family": "system_diagnostics",
                    "confidence": 0.91,
                    "requires_live_state": "false",
                    "requires_execution": True,
                    "requires_write": False,
                    "tool_names": ["tool.system.read.hardware"],
                    "risk_classes": ["read_only"],
                    "scope_hint": "os_version",
                    "evidence_codes": ["system_os_version_request"],
                }
            ],
            "requires_live_state": True,
            "requires_execution": True,
            "answer_without_tools_would_be_misleading": True,
            "reason_code": "model_system_diagnostics",
            "fallback_preference": "fail_unavailable",
        }
    )

    classification = asyncio.run(ModelBackedIntentClassifier(router=router).classify(_request()))

    assert classification.intent_family is IntentFamily.UNKNOWN
    assert classification.reason_code == "classifier_unavailable"


def test_model_backed_classifier_guardrail_keeps_conceptual_os_question_as_chat() -> None:
    router = FakeStructuredRouter(
        {
            "intent_family": "ordinary_chat",
            "confidence": 0.8,
            "candidate_capabilities": [],
            "requires_live_state": False,
            "requires_execution": False,
            "answer_without_tools_would_be_misleading": False,
            "reason_code": "ordinary_chat",
            "fallback_preference": "chat",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(router=router).classify(
            _request(user_input="Объясни, что такое macOS build number")
        )
    )

    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.candidate_capabilities == ()


class FakeStructuredRouter:
    def __init__(self, response: dict | Exception) -> None:
        self.response = response
        self.requests: list[StructuredModelRequest] = []

    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return StructuredModelResponse(value=self.response)


class StaticClassifier:
    def __init__(
        self,
        *,
        intent_family: IntentFamily,
        capability: Capability,
        tool_name: str,
        scope_hint: str | None,
    ) -> None:
        self.intent_family = intent_family
        self.capability = capability
        self.tool_name = tool_name
        self.scope_hint = scope_hint

    async def classify(self, request: LoopSelectionRequest) -> IntentClassification:
        return IntentClassification(
            intent_family=self.intent_family,
            confidence=0.82,
            candidate_capabilities=(
                CapabilityCandidate(
                    capability=self.capability,
                    intent_family=self.intent_family,
                    confidence=0.82,
                    requires_live_state=True,
                    requires_execution=True,
                    requires_write=False,
                    tool_names=(self.tool_name,),
                    risk_classes=frozenset({RiskClass.READ_ONLY}),
                    scope_hint=self.scope_hint,
                    evidence_codes=("generic_fallback_hint",),
                ),
            ),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
            reason_code="generic_fallback_hint",
            fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        )


class StaticOrdinaryChatClassifier:
    def __init__(self, *, confidence: float) -> None:
        self.confidence = confidence

    async def classify(self, request: LoopSelectionRequest) -> IntentClassification:
        return IntentClassification(
            intent_family=IntentFamily.ORDINARY_CHAT,
            confidence=self.confidence,
            candidate_capabilities=(),
            requires_live_state=False,
            requires_execution=False,
            answer_without_tools_would_be_misleading=False,
            reason_code="ordinary_chat_explicit_hint",
            fallback_preference=SelectionFallbackPreference.CHAT,
        )


def _request(
    *,
    user_input: str = "Какая версия операционной системы?",
    available_tools_summary: tuple[dict, ...] | None = None,
) -> LoopSelectionRequest:
    return LoopSelectionRequest(
        request_id="request-1",
        conversation_id="conversation-1",
        user_id="user-1",
        requested_mode=LoopSelectionMode.AUTO,
        user_input=user_input,
        current_message_sensitivity=Sensitivity.PROJECT,
        active_project_namespace="project.personal_assistant",
        working_directory="/tmp/project",
        permission_mode=PermissionMode.DEVELOPER_LOCAL,
        available_capabilities=frozenset(Capability),
        available_tools_summary=(
            _default_available_tools_summary()
            if available_tools_summary is None
            else available_tools_summary
        ),
        runtime_budget_summary={},
        metadata={"source": "test"},
    )
