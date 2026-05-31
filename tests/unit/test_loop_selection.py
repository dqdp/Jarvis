from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import (
    CapabilityCandidate,
    IntentClassification,
    IntentFamily,
    LoopSelectionDecision,
    LoopSelectionMode,
    LoopSelectionRequest,
    SelectionDecisionStatus,
    SelectionFallbackPreference,
)
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.policy import (
    Capability,
    PermissionMode,
    PolicyDecision,
    PolicyDecisionOutcome,
    RiskClass,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.runtime.loop_selection import (
    DeterministicIntentClassifier,
    FakeIntentClassifier,
    LoopStrategySelector,
)
from assistant_core.runtime.request_metadata import resolve_loop_selection_mode


pytestmark = pytest.mark.unit


def test_loop_selection_mode_accepts_auto_chat_tools() -> None:
    assert [
        LoopSelectionMode.AUTO.value,
        LoopSelectionMode.CHAT.value,
        LoopSelectionMode.TOOLS.value,
    ] == ["auto", "chat", "tools"]
    assert LoopSelectionMode("auto") is LoopSelectionMode.AUTO
    assert LoopSelectionMode("chat") is LoopSelectionMode.CHAT
    assert LoopSelectionMode("tools") is LoopSelectionMode.TOOLS
    assert LoopSelectionMode("invalid_override") is LoopSelectionMode.INVALID_OVERRIDE


def test_internal_invalid_override_mode_is_not_user_selectable() -> None:
    with pytest.raises(ValueError, match="loop strategy is not configured"):
        resolve_loop_selection_mode("invalid_override")


def test_loop_selection_request_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="request_id is required"):
        _request(request_id="")


def test_intent_classification_requires_confidence_between_zero_and_one() -> None:
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        _classification(IntentFamily.ORDINARY_CHAT, confidence=1.5)

    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        _classification(IntentFamily.ORDINARY_CHAT, confidence=-0.01)


def test_intent_classification_rejects_non_finite_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be finite"):
        _classification(IntentFamily.ORDINARY_CHAT, confidence=float("nan"))

    with pytest.raises(ValueError, match="confidence must be finite"):
        _candidate(confidence=float("inf"))


def test_capability_candidate_does_not_store_raw_prompt_evidence() -> None:
    with pytest.raises(ValueError, match="evidence_codes must be stable labels"):
        _candidate(evidence_codes=("please check cpu temperature",))


def test_capability_candidate_does_not_store_raw_prompt_scope_hint() -> None:
    with pytest.raises(ValueError, match="scope_hint must be a stable label"):
        _candidate(scope_hint="please check cpu temperature on my laptop")


def test_capability_candidate_payload_does_not_include_scope_hint() -> None:
    payload = _candidate(scope_hint="project_file_lookup").redacted_payload()

    assert "scope_hint" not in payload


def test_capability_candidate_rejects_unknown_capability_strings() -> None:
    with pytest.raises(ValueError, match="capability must be a known capability"):
        _candidate(capability="please check cpu temperature on my laptop")


def test_loop_selection_decision_distinguishes_requested_mode_from_selected_loop() -> None:
    decision = LoopSelectionDecision(
        requested_mode=LoopSelectionMode.AUTO,
        selected_loop_strategy=LoopStrategyName.TOOL_REACT_LOOP,
        selected_model_profile=None,
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        reason_code="tool_intent_system_diagnostics",
        confidence=0.92,
        candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_SENSORS),),
        requires_tools=True,
        requires_live_state=True,
        policy_outcome=PolicyDecisionOutcome.ALLOW,
        approval_possible=False,
        fallback_behavior=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        decision_status=SelectionDecisionStatus.SELECTED,
    )

    assert decision.requested_mode is LoopSelectionMode.AUTO
    assert decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP
    assert decision.selected_model_profile is None


def test_medium_confidence_tool_intent_falls_back_conservatively() -> None:
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.62,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_RESOURCES),),
            requires_live_state=True,
            requires_execution=True,
        )
    )

    assert decision.decision_status is SelectionDecisionStatus.FALLBACK_CHAT
    assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER
    assert decision.reason_code == "tool_intent_medium_confidence_fallback_chat"


def test_medium_confidence_misleading_tool_intent_with_tools_disabled_does_not_fallback() -> None:
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.62,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_RESOURCES),),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        tools_enabled=False,
    )

    assert decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE
    assert decision.selected_loop_strategy is None
    assert decision.reason_code == "tool_intent_medium_confidence_requires_tools"


def test_medium_confidence_non_misleading_tool_intent_with_tools_disabled_falls_back() -> None:
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.62,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_RESOURCES),),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=False,
        ),
        tools_enabled=False,
    )

    assert decision.decision_status is SelectionDecisionStatus.FALLBACK_CHAT
    assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER
    assert decision.reason_code == "tool_intent_medium_confidence_fallback_chat"


def test_misleading_without_tools_does_not_fallback_to_fake_chat() -> None:
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.91,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_SENSORS),),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        tools_enabled=False,
    )

    assert decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE
    assert decision.selected_loop_strategy is None
    assert decision.reason_code == "tools_disabled_for_tool_intent"


def test_selector_uses_intent_classifier_for_auto_mode() -> None:
    classifier = FakeIntentClassifier(
        _classification(IntentFamily.ORDINARY_CHAT, confidence=0.93)
    )

    decision = asyncio.run(LoopStrategySelector(intent_classifier=classifier).select(_request()))

    assert classifier.requests == (_request(),)
    assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER


def test_fake_intent_classifier_drives_selector_decision() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_INSPECTION,
            confidence=0.93,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SHELL_READ),),
            requires_live_state=True,
            requires_execution=True,
        ),
        policy=FakeCapabilityPolicy(),
    )

    assert decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP
    assert decision.reason_code == "tool_intent_project_read"


def test_auto_selects_memory_loop_for_ordinary_chat() -> None:
    decision = _select(_classification(IntentFamily.ORDINARY_CHAT, confidence=0.95))

    assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER
    assert decision.reason_code == "ordinary_chat"


def test_auto_selects_memory_loop_for_project_docs_question() -> None:
    decision = _select(_classification(IntentFamily.PROJECT_DOCS_QUESTION, confidence=0.96))

    assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER
    assert decision.requires_tools is False
    assert decision.reason_code == "project_docs_question"


def test_auto_selects_tool_loop_for_project_shell_read_intent() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_INSPECTION,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SHELL_READ),),
            requires_live_state=True,
            requires_execution=True,
        ),
        policy=FakeCapabilityPolicy(),
    )

    assert decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP
    assert decision.requires_tools is True


def test_auto_selects_tool_loop_for_system_diagnostics_intent() -> None:
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_SENSORS),),
            requires_live_state=True,
            requires_execution=True,
        ),
        policy=FakeCapabilityPolicy(),
    )

    assert decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP
    assert decision.reason_code == "tool_intent_system_diagnostics"


def test_selector_passes_working_directory_to_real_policy_for_system_diagnostics() -> None:
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_SENSORS),),
            requires_live_state=True,
            requires_execution=True,
        ),
        request=_request(
            current_message_sensitivity=Sensitivity.INFRA,
            working_directory=str(Path.cwd()),
        ),
        policy=ConfigPolicyEngine(ConfigLoader(Path("config")).load("test")),
    )

    assert decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP
    assert decision.policy_outcome is PolicyDecisionOutcome.ALLOW


def test_auto_reports_tools_disabled_for_tool_intent() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_INSPECTION,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SHELL_READ),),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        tools_enabled=False,
    )

    assert decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE
    assert decision.selected_loop_strategy is None


def test_tools_disabled_reason_takes_precedence_over_unavailable_capabilities() -> None:
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_SENSORS),),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        request=_request(available_capabilities=frozenset()),
        tools_enabled=False,
        policy=FakeCapabilityPolicy(),
    )

    assert decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE
    assert decision.reason_code == "tools_disabled_for_tool_intent"


def test_classifier_low_confidence_falls_back_to_chat() -> None:
    decision = _select(_classification(IntentFamily.UNKNOWN, confidence=0.1))

    assert decision.decision_status is SelectionDecisionStatus.FALLBACK_CHAT
    assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER
    assert decision.reason_code == "classifier_low_confidence"


def test_non_tool_intent_drops_tool_candidate_metadata_before_chat_fallback() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_DOCS_QUESTION,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SHELL_READ),),
        )
    )

    assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER
    assert decision.candidate_capabilities == ()
    assert decision.policy_outcome is None


def test_classifier_unavailable_does_not_return_executable_loop() -> None:
    decision = asyncio.run(LoopStrategySelector(intent_classifier=FailingIntentClassifier()).select(_request()))

    assert decision.decision_status is SelectionDecisionStatus.CLASSIFIER_UNAVAILABLE
    assert decision.selected_loop_strategy is None
    assert decision.reason_code == "classifier_unavailable"


def test_tool_intent_without_candidate_capability_is_unavailable() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_INSPECTION,
            confidence=0.9,
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        policy=FakeCapabilityPolicy(),
    )

    assert decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE
    assert decision.selected_loop_strategy is None
    assert decision.reason_code == "missing_capability_candidate_for_tool_intent"


def test_tool_intent_without_policy_is_rejected() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_INSPECTION,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SHELL_READ),),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        policy=None,
    )

    assert decision.decision_status is SelectionDecisionStatus.REJECTED_BY_POLICY
    assert decision.selected_loop_strategy is None
    assert decision.reason_code == "policy_unavailable_for_tool_intent"


def test_classifier_tool_intent_is_clamped_by_policy() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_INSPECTION,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SHELL_READ),),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        policy=FakeCapabilityPolicy(allowed=False, code="shell_read_denied"),
    )

    assert decision.decision_status is SelectionDecisionStatus.REJECTED_BY_POLICY
    assert decision.selected_loop_strategy is None
    assert decision.policy_outcome is PolicyDecisionOutcome.DENY
    assert decision.reason_code == "shell_read_denied"


def test_classifier_tool_intent_checks_all_candidate_capabilities() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_INSPECTION,
            confidence=0.9,
            candidate_capabilities=(
                _candidate(capability=Capability.TOOL_SHELL_READ),
                _candidate(capability=Capability.TOOL_SHELL_WRITE),
            ),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        policy=FakeCapabilityPolicy(
            outcomes_by_capability={
                Capability.TOOL_SHELL_WRITE: PolicyDecisionOutcome.DENY,
            },
            codes_by_capability={
                Capability.TOOL_SHELL_WRITE: "shell_write_denied",
            },
        ),
    )

    assert decision.decision_status is SelectionDecisionStatus.REJECTED_BY_POLICY
    assert decision.selected_loop_strategy is None
    assert decision.policy_outcome is PolicyDecisionOutcome.DENY
    assert decision.reason_code == "shell_write_denied"


def test_classifier_tool_intent_requires_available_capability() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_INSPECTION,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SHELL_READ),),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        request=_request(available_capabilities=frozenset({Capability.MEMORY_READ})),
        policy=FakeCapabilityPolicy(),
    )

    assert decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE
    assert decision.selected_loop_strategy is None
    assert decision.reason_code == "capability_unavailable_for_tool_intent"


def test_classifier_tool_intent_preserves_approval_required_policy_outcome() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_INSPECTION,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SHELL_READ),),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
        ),
        policy=FakeCapabilityPolicy(
            allowed=False,
            code="shell_read_needs_approval",
            outcome=PolicyDecisionOutcome.APPROVAL_REQUIRED,
        ),
    )

    assert decision.decision_status is SelectionDecisionStatus.SELECTED
    assert decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP
    assert decision.policy_outcome is PolicyDecisionOutcome.APPROVAL_REQUIRED
    assert decision.approval_possible is True


def test_explicit_chat_override_selects_memory_loop() -> None:
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.9,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_RESOURCES),),
        ),
        request=_request(requested_mode=LoopSelectionMode.CHAT),
    )

    assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER
    assert decision.intent_family is IntentFamily.ORDINARY_CHAT
    assert decision.reason_code == "explicit_memory_loop"


def test_explicit_tools_override_selects_tool_loop() -> None:
    decision = _select(
        _classification(IntentFamily.ORDINARY_CHAT, confidence=0.9),
        request=_request(requested_mode=LoopSelectionMode.TOOLS),
        policy=FakeCapabilityPolicy(),
    )

    assert decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP
    assert decision.reason_code == "explicit_tool_loop"


def test_explicit_tools_override_requires_available_capability() -> None:
    decision = _select(
        _classification(IntentFamily.ORDINARY_CHAT, confidence=0.9),
        request=_request(
            requested_mode=LoopSelectionMode.TOOLS,
            available_capabilities=frozenset(),
        ),
        policy=FakeCapabilityPolicy(),
    )

    assert decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE
    assert decision.selected_loop_strategy is None
    assert decision.reason_code == "capability_unavailable_for_tool_intent"


def test_deterministic_classifier_keeps_general_where_question_as_chat() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="where are you from?")
        )
    )

    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.candidate_capabilities == ()


def test_deterministic_classifier_routes_current_time_question_to_builtin_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Сколько время?")
        )
    )

    assert classification.intent_family is IntentFamily.SAFE_BUILTIN_TOOL
    assert classification.candidate_capabilities
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SAFE
    assert candidate.tool_names == ("datetime.now",)
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_routes_common_russian_current_time_question_to_builtin_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Сколько времени?")
        )
    )

    assert classification.intent_family is IntentFamily.SAFE_BUILTIN_TOOL
    assert classification.candidate_capabilities[0].tool_names == ("datetime.now",)


def test_deterministic_classifier_routes_russian_christmas_countdown_to_datetime_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="через сколько дней Рождество?")
        )
    )

    assert classification.intent_family is IntentFamily.SAFE_BUILTIN_TOOL
    assert classification.candidate_capabilities
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SAFE
    assert candidate.tool_names == ("datetime.now",)
    assert candidate.scope_hint == "christmas_countdown"
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_keeps_coding_current_time_question_as_chat() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="How do I print current time in Python?")
        )
    )

    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.candidate_capabilities == ()


def test_deterministic_classifier_routes_russian_cpu_temperature_to_sensors_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Текущая температура процессора.")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SYSTEM_READ_SENSORS
    assert candidate.tool_names == ("tool.system.read.sensors",)
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_routes_russian_free_memory_to_resources_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Сколько памяти сейчас свободно в системе?")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SYSTEM_READ_RESOURCES
    assert candidate.tool_names == ("tool.system.read.resources",)
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_routes_russian_free_disk_to_resources_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Сколько свободного места на диске?")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SYSTEM_READ_RESOURCES
    assert candidate.tool_names == ("tool.system.read.resources",)
    assert candidate.scope_hint == "disk_free"
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_routes_russian_battery_charge_to_hardware_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Сколько процентов заряда аккумулятора осталось на макбуке?")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SYSTEM_READ_HARDWARE
    assert candidate.tool_names == ("tool.system.read.hardware",)
    assert candidate.scope_hint == "battery_charge"
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_routes_russian_cpu_cores_and_load_to_cpu_overview_tools() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Сколько ядер у центрального процессора и на сколько они загружены?")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert [candidate.capability for candidate in classification.candidate_capabilities] == [
        Capability.TOOL_SYSTEM_READ_HARDWARE,
        Capability.TOOL_SYSTEM_READ_RESOURCES,
    ]
    assert [
        candidate.tool_names for candidate in classification.candidate_capabilities
    ] == [
        ("tool.system.read.hardware",),
        ("tool.system.read.resources",),
    ]
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_routes_russian_os_version_to_hardware_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Какая версия операционной системы?")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SYSTEM_READ_HARDWARE
    assert candidate.tool_names == ("tool.system.read.hardware",)
    assert candidate.scope_hint == "os_version"
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_keeps_weather_temperature_question_as_chat() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Какая температура на улице?")
        )
    )

    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.candidate_capabilities == ()


def test_deterministic_classifier_does_not_match_temp_inside_template() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Explain the system template.")
        )
    )

    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.candidate_capabilities == ()


def test_auto_selects_tool_loop_for_current_time_question() -> None:
    decision = asyncio.run(
        LoopStrategySelector(
            intent_classifier=DeterministicIntentClassifier(),
            policy=FakeCapabilityPolicy(),
        ).select(_request(user_input="what time is it?"))
    )

    assert decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP
    assert decision.intent_family is IntentFamily.SAFE_BUILTIN_TOOL
    assert decision.reason_code == "tool_intent_safe_builtin"


def test_selector_does_not_treat_rag_as_tool_loop() -> None:
    decision = _select(
        _classification(
            IntentFamily.PROJECT_DOCS_QUESTION,
            confidence=0.92,
            candidate_capabilities=(_candidate(capability=Capability.CONTENT_RETRIEVE),),
            requires_live_state=False,
            requires_execution=False,
        )
    )

    assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER
    assert decision.requires_tools is False


def test_selector_outputs_reason_code_and_candidate_capabilities() -> None:
    candidate = _candidate(
        capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
        evidence_codes=("system_diagnostics",),
    )
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.94,
            candidate_capabilities=(candidate,),
            requires_live_state=True,
            requires_execution=True,
        ),
        policy=FakeCapabilityPolicy(),
    )

    assert decision.reason_code == "tool_intent_system_diagnostics"
    assert decision.candidate_capabilities == (candidate,)


def test_selector_does_not_log_raw_prompt_in_decision_payload() -> None:
    request = _request(user_input="please check cpu temperature on my laptop")
    decision = _select(
        _classification(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.94,
            candidate_capabilities=(_candidate(capability=Capability.TOOL_SYSTEM_READ_SENSORS),),
            requires_live_state=True,
            requires_execution=True,
        ),
        request=request,
        policy=FakeCapabilityPolicy(),
    )

    payload = decision.redacted_event_payload(
        request_id=request.request_id,
        conversation_id=request.conversation_id,
    )

    assert request.user_input not in repr(payload)
    assert payload["selected_loop_strategy"] == "tool_react_loop"
    assert payload["reason_code"] == "tool_intent_system_diagnostics"


def test_deterministic_classifier_prefers_project_docs_over_where_hint() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="where do docs describe shell sandbox rules?")
        )
    )

    assert classification.intent_family is IntentFamily.PROJECT_DOCS_QUESTION
    assert classification.candidate_capabilities == ()


def test_deterministic_classifier_does_not_treat_general_memory_question_as_diagnostics() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="explain memory architecture")
        )
    )

    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.requires_execution is False


def test_deterministic_classifier_returns_capability_candidates_for_tool_intent() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="check cpu temperature")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    assert classification.candidate_capabilities[0].capability is Capability.TOOL_SYSTEM_READ_SENSORS


def test_deterministic_classifier_routes_listening_on_port_to_network_diagnostics() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="what is listening on port 8080?")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    assert classification.candidate_capabilities[0].capability is Capability.TOOL_SYSTEM_READ_NETWORK


def test_deterministic_classifier_routes_process_listening_on_port_to_network_diagnostics() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="show what process is listening on this port")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    assert classification.candidate_capabilities[0].capability is Capability.TOOL_SYSTEM_READ_NETWORK


def test_deterministic_classifier_routes_russian_vpn_status_to_network_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="Включен ли VPN сейчас?")
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SYSTEM_READ_NETWORK
    assert candidate.tool_names == ("tool.system.read.network",)
    assert candidate.scope_hint == "vpn_status"
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_does_not_route_generic_process_word_to_tools() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="same process replay")
        )
    )

    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.candidate_capabilities == ()


def test_deterministic_classifier_routes_russian_process_name_search_to_process_tool() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input='Запущен ли сейчас процесс, в имени которого есть "HFT"?')
        )
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities
    candidate = classification.candidate_capabilities[0]
    assert candidate.capability is Capability.TOOL_SYSTEM_READ_PROCESS
    assert candidate.tool_names == ("tool.system.read.process",)
    assert candidate.scope_hint == "process_name_search"
    assert classification.answer_without_tools_would_be_misleading is True


def test_deterministic_classifier_keeps_general_network_question_as_chat() -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(
            _request(user_input="explain network architecture")
        )
    )

    assert classification.intent_family is IntentFamily.ORDINARY_CHAT
    assert classification.candidate_capabilities == ()


class FakeCapabilityPolicy:
    def __init__(
        self,
        *,
        allowed: bool = True,
        code: str = "allowed",
        outcome: PolicyDecisionOutcome | None = None,
        outcomes_by_capability: dict[Capability, PolicyDecisionOutcome] | None = None,
        codes_by_capability: dict[Capability, str] | None = None,
    ) -> None:
        self.allowed = allowed
        self.code = code
        self.outcome = outcome or (
            PolicyDecisionOutcome.ALLOW if allowed else PolicyDecisionOutcome.DENY
        )
        self.outcomes_by_capability = outcomes_by_capability or {}
        self.codes_by_capability = codes_by_capability or {}
        self.requests = []

    async def evaluate_capability_request(self, request):
        self.requests.append(request)
        outcome = self.outcomes_by_capability.get(request.capability, self.outcome)
        code = self.codes_by_capability.get(request.capability, self.code)
        return PolicyDecision(
            allowed=outcome is PolicyDecisionOutcome.ALLOW,
            code=code,
            reason=code,
            outcome=outcome,
            capability=request.capability,
            permission_mode=request.permission_mode,
            risk_classes=request.risk_classes,
            sensitivity=request.sensitivity,
        )


class FailingIntentClassifier:
    async def classify(self, request):
        raise RuntimeError("classifier failed")


def _select(
    classification: IntentClassification,
    *,
    request: LoopSelectionRequest | None = None,
    tools_enabled: bool = True,
    policy: object | None = None,
) -> LoopSelectionDecision:
    return asyncio.run(
        LoopStrategySelector(
            intent_classifier=FakeIntentClassifier(classification),
            tools_enabled=tools_enabled,
            policy=policy,
        ).select(request or _request())
    )


def _request(
    *,
    request_id: str = "request-1",
    requested_mode: LoopSelectionMode = LoopSelectionMode.AUTO,
    user_input: str = "hello",
    current_message_sensitivity: Sensitivity = Sensitivity.PERSONAL,
    available_capabilities: frozenset[Capability | str] = frozenset(Capability),
    working_directory: str | None = None,
) -> LoopSelectionRequest:
    return LoopSelectionRequest(
        request_id=request_id,
        conversation_id="conversation-1",
        user_id="user-1",
        requested_mode=requested_mode,
        user_input=user_input,
        current_message_sensitivity=current_message_sensitivity,
        active_project_namespace="project.personal_assistant",
        permission_mode=PermissionMode.DEVELOPER_LOCAL,
        available_capabilities=available_capabilities,
        available_tools_summary=(),
        runtime_budget_summary={},
        working_directory=working_directory,
        metadata={"source": "test"},
    )


def _classification(
    intent_family: IntentFamily,
    *,
    confidence: float,
    candidate_capabilities: tuple[CapabilityCandidate, ...] = (),
    requires_live_state: bool = False,
    requires_execution: bool = False,
    answer_without_tools_would_be_misleading: bool = False,
) -> IntentClassification:
    return IntentClassification(
        intent_family=intent_family,
        confidence=confidence,
        candidate_capabilities=candidate_capabilities,
        requires_live_state=requires_live_state,
        requires_execution=requires_execution,
        answer_without_tools_would_be_misleading=answer_without_tools_would_be_misleading,
        reason_code=intent_family.value,
        fallback_preference=SelectionFallbackPreference.CHAT,
    )


def _candidate(
    *,
    capability: Capability = Capability.TOOL_SAFE,
    intent_family: IntentFamily = IntentFamily.SYSTEM_DIAGNOSTICS,
    confidence: float = 0.9,
    requires_live_state: bool = True,
    requires_execution: bool = True,
    requires_write: bool = False,
    tool_names: tuple[str, ...] = ("system_diagnostics",),
    risk_classes: frozenset[RiskClass] = frozenset({RiskClass.READ_ONLY}),
    scope_hint: str | None = None,
    evidence_codes: tuple[str, ...] = ("system_diagnostics",),
) -> CapabilityCandidate:
    return CapabilityCandidate(
        capability=capability,
        intent_family=intent_family,
        confidence=confidence,
        requires_live_state=requires_live_state,
        requires_execution=requires_execution,
        requires_write=requires_write,
        tool_names=tool_names,
        risk_classes=risk_classes,
        scope_hint=scope_hint,
        evidence_codes=evidence_codes,
    )
