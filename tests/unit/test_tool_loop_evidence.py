from __future__ import annotations

import pytest

from assistant_core.domain.loops import (
    LoopBudget,
    LoopExecutionRequest,
    LoopStrategyName,
    ToolObservationRef,
    ToolRequestPlan,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolObservationStatus, ToolParseStatus
from assistant_core.runtime.loops.tool_loop_evidence import (
    LiveStateEvidenceKind,
    LiveStateEvidenceFamily,
    LiveStateEvidencePlan,
    contains_live_state_intent,
    detect_live_state_family,
    failed_observation_exhausts_missing_evidence,
    final_answer_missing_evidence_plan,
    is_live_state_tool_name,
    live_state_evidence_plan,
    request_requires_initial_tool_evidence,
    request_needs_live_state_math_evidence,
    should_defer_final_answer_for_calculator_evidence,
)
from assistant_core.runtime.loops.tool_loop_contracts import tool_proposal_output_contract


pytestmark = pytest.mark.unit


def _budget() -> LoopBudget:
    return LoopBudget(
        max_steps=4,
        max_model_calls=4,
        max_tool_calls=2,
        max_wall_time_seconds=60,
        max_context_assembly_seconds=10,
        max_model_call_seconds=60,
        max_consecutive_failures=1,
    )


def _request(user_input: str) -> LoopExecutionRequest:
    return LoopExecutionRequest(
        request_id="request-evidence",
        conversation_id="conversation-evidence",
        user_message_id="message-user",
        user_id="user-1",
        user_input=user_input,
        active_project_namespace="project.personal_assistant",
        current_message_sensitivity=Sensitivity.PROJECT,
        model_profile="local_main",
        strategy_name=LoopStrategyName.TOOL_REACT_LOOP,
        budget=_budget(),
    )


def _plan(
    *allowed_tool_names: str,
    live_state_tool_names: tuple[str, ...] = (),
) -> ToolRequestPlan:
    return ToolRequestPlan(
        policy="available",
        allowed_tool_names=frozenset(allowed_tool_names),
        live_state_tool_names=frozenset(live_state_tool_names),
    )


def _completed_ref(
    tool_name: str,
    *,
    arguments: dict | None = None,
    content: str = "{}",
    structured_content: dict | None = None,
    structured_schema: str | None = None,
    parse_status: ToolParseStatus | None = None,
    metadata: dict | None = None,
    truncated: bool = False,
) -> ToolObservationRef:
    if metadata is None and arguments and isinstance(arguments.get("argv"), list):
        metadata = {"exit_code": 0}
    return ToolObservationRef(
        tool_call_id=f"tool-call-{tool_name}",
        tool_name=tool_name,
        status=ToolObservationStatus.COMPLETED,
        content=content,
        content_type="application/json",
        sensitivity=Sensitivity.PROJECT,
        truncated=truncated,
        structured_content=structured_content,
        structured_schema=structured_schema,
        parse_status=parse_status,
        metadata=metadata or {},
        arguments=arguments or {},
    )


def _completed_resource_ref() -> ToolObservationRef:
    return _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.resource_overview",
        structured_content={
            "cpu": {"used_percent": 10.2},
            "memory": {"used_percent": 42.0},
            "load_average": [1.0, 1.2, 1.4],
        },
        parse_status=ToolParseStatus.PARSED,
    )


def _failed_ref(tool_name: str, *, error_code: str = "tool_failed") -> ToolObservationRef:
    return ToolObservationRef(
        tool_call_id=f"tool-call-{tool_name}",
        tool_name=tool_name,
        status=ToolObservationStatus.FAILED,
        content="",
        content_type="text/plain",
        sensitivity=Sensitivity.PROJECT,
        error_code=error_code,
    )


def test_live_state_evidence_plan_detects_current_time_modifier_and_candidates() -> None:
    plan = live_state_evidence_plan(
        _request("сколько времени в данный момент?"),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})


def test_live_state_evidence_plan_requires_datetime_for_bare_current_time_question() -> None:
    plan = live_state_evidence_plan(
        _request("какое время?"),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})
    assert plan.missing_families == frozenset({LiveStateEvidenceFamily.CURRENT_TIME})


@pytest.mark.parametrize(
    "user_input",
    [
        "сколько времени в Париже?",
        "какое время в Париже?",
        "what time is it in Paris?",
        "current local time in Berlin?",
    ],
)
def test_live_state_evidence_plan_requires_datetime_for_location_scoped_time(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})
    assert plan.missing_families == frozenset({LiveStateEvidenceFamily.CURRENT_TIME})


def test_live_state_evidence_plan_requires_datetime_for_current_date() -> None:
    plan = live_state_evidence_plan(
        _request("какая дата в данный момент?"),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_DATE
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})
    assert plan.missing_families == frozenset({LiveStateEvidenceFamily.CURRENT_DATE})


def test_live_state_evidence_plan_accepts_completed_datetime_until_for_countdown() -> None:
    payload = {
        "from_iso": "2026-06-05T20:59:07+03:00",
        "target": "next_new_year",
        "unit": "seconds",
        "value": 18337521,
    }
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.until",
                structured_schema="datetime.until",
                structured_content=payload,
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason is None


def test_live_state_evidence_plan_keeps_unsupported_countdown_target_unavailable_after_datetime_now() -> None:
    request = _request("how many seconds until Christmas?")
    request_plan = _plan(
        "datetime.now",
        "datetime.until",
        live_state_tool_names=("datetime.now", "datetime.until"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T15:13:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )

    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref,),
    )
    final_answer_plan = final_answer_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref,),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset()
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"
    assert final_answer_plan is not None
    assert final_answer_plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_evidence_plan_rejects_datetime_until_arguments_without_typed_payload() -> None:
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.until",
                arguments={"target": "next_new_year", "unit": "seconds"},
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_rejects_datetime_until_raw_json_payload() -> None:
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.until",
                content='{"target": "next_new_year", "unit": "seconds", "value": 18337521}',
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == plan.candidate_tool_names


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"target": "next_new_year", "unit": "days"},
        {"target": "next_christmas", "unit": "seconds"},
    ],
)
def test_live_state_evidence_plan_requires_relevant_datetime_until_arguments(
    arguments: dict,
) -> None:
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(_completed_ref("datetime.until", arguments=arguments),),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_rejects_datetime_until_with_unmatched_from_iso() -> None:
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.until",
                arguments={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": "2000-01-01T00:00:00+00:00",
                },
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_rejects_datetime_until_content_with_unmatched_from_iso() -> None:
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.until",
                content=(
                    '{"target": "next_new_year", '
                    '"unit": "seconds", '
                    '"from_iso": "2000-01-01T00:00:00+00:00"}'
                ),
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_rejects_datetime_until_argument_match_with_unmatched_content_from_iso() -> None:
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.until",
                arguments={
                    "target": "next_new_year",
                    "unit": "seconds",
                },
                content=(
                    '{"target": "next_new_year", '
                    '"unit": "seconds", '
                    '"from_iso": "2000-01-01T00:00:00+00:00"}'
                ),
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_rejects_datetime_until_unmatched_argument_from_iso_even_when_content_matches() -> None:
    now_iso = "2026-06-05T20:59:07+03:00"
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                structured_schema="datetime.now",
                structured_content={"iso": now_iso},
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref(
                "datetime.until",
                arguments={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": "2000-01-01T00:00:00+00:00",
                },
                structured_schema="datetime.until",
                structured_content={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": "2000-01-01T00:00:00+00:00",
                },
                parse_status=ToolParseStatus.PARSED,
                content=(
                    '{"target": "next_new_year", '
                    '"unit": "seconds", '
                    f'"from_iso": "{now_iso}"}}'
                ),
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_accepts_datetime_until_structured_source_when_raw_content_disagrees() -> None:
    now_iso = "2026-06-05T20:59:07+03:00"
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                structured_schema="datetime.now",
                structured_content={"iso": now_iso},
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref(
                "datetime.until",
                arguments={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": now_iso,
                },
                structured_schema="datetime.until",
                structured_content={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": now_iso,
                },
                parse_status=ToolParseStatus.PARSED,
                content=(
                    '{"target": "next_new_year", '
                    '"unit": "seconds", '
                    '"from_iso": "2000-01-01T00:00:00+00:00"}'
                ),
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_accepts_datetime_until_with_matching_datetime_now_source() -> None:
    now_iso = "2026-06-05T20:59:07+03:00"
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                structured_schema="datetime.now",
                structured_content={"iso": now_iso},
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref(
                "datetime.until",
                arguments={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": now_iso,
                },
                structured_schema="datetime.until",
                structured_content={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": now_iso,
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_datetime_until_with_raw_datetime_now_content_source() -> None:
    now_iso = "2026-06-05T20:59:07+03:00"
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                content='{"iso": "2026-06-05T20:59:07+03:00"}',
            ),
            _completed_ref(
                "datetime.until",
                arguments={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": now_iso,
                },
                structured_schema="datetime.until",
                structured_content={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": now_iso,
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_requires_datetime_until_after_datetime_now_for_supported_countdown() -> None:
    plan = live_state_evidence_plan(
        _request("сколько секунд до нового года?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                structured_schema="datetime.now",
                structured_content={"iso": "2026-06-07T15:13:00+03:00"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == frozenset({"datetime.until"})


def test_live_state_evidence_plan_supported_countdown_stays_unavailable_when_datetime_until_is_not_allowed_after_now() -> None:
    request = _request("сколько секунд до нового года?")
    request_plan = _plan("datetime.now", live_state_tool_names=("datetime.now",))

    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                structured_schema="datetime.now",
                structured_content={"iso": "2026-06-07T15:13:00+03:00"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    final_answer_plan = final_answer_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                structured_schema="datetime.now",
                structured_content={"iso": "2026-06-07T15:13:00+03:00"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"
    assert final_answer_plan is not None
    assert final_answer_plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_evidence_plan_does_not_clear_unsupported_countdown_with_datetime_now() -> None:
    plan = live_state_evidence_plan(
        _request("how many seconds until Christmas?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                structured_schema="datetime.now",
                structured_content={"iso": "2026-06-07T15:13:00+03:00"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset()
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_evidence_plan_rejects_untyped_datetime_now_observation() -> None:
    plan = live_state_evidence_plan(
        _request("what time is it now?"),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(_completed_ref("datetime.now"),),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})


def test_live_state_evidence_plan_rejects_schema_less_datetime_now_payload() -> None:
    plan = live_state_evidence_plan(
        _request("what time is it now?"),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                structured_content={"iso": "2026-06-07T15:13:00+03:00"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})


def test_live_state_evidence_plan_rejects_content_embedded_datetime_schema() -> None:
    plan = live_state_evidence_plan(
        _request("what time is it now?"),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                structured_content={
                    "schema": "datetime.now",
                    "iso": "2026-06-07T15:13:00+03:00",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})


def test_live_state_evidence_plan_rejects_raw_json_datetime_now_payload() -> None:
    plan = live_state_evidence_plan(
        _request("what time is it now?"),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(
            _completed_ref(
                "datetime.now",
                content='{"iso": "2026-06-07T15:13:00+03:00"}',
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})


def test_live_state_evidence_plan_does_not_accept_datetime_until_for_unsupported_countdown_target() -> None:
    plan = live_state_evidence_plan(
        _request("how many seconds until Christmas?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(_completed_ref("datetime.until"),),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset()
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"


@pytest.mark.parametrize(
    ("user_input", "expected_family"),
    [
        ("какое сейчас время?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("какая сейчас дата?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("какая дата в данный момент?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("какое в данный момент время?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("какая в данный момент дата?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("назови текущее время", LiveStateEvidenceFamily.CURRENT_TIME),
        ("скажи время", LiveStateEvidenceFamily.CURRENT_TIME),
        ("скажи, который час", LiveStateEvidenceFamily.CURRENT_TIME),
        ("подскажи, пожалуйста, который час", LiveStateEvidenceFamily.CURRENT_TIME),
        ("что там по времени?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("what is the time right now?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("what's the time?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("time please", LiveStateEvidenceFamily.CURRENT_TIME),
        ("the time please", LiveStateEvidenceFamily.CURRENT_TIME),
        ("please tell me what time it is", LiveStateEvidenceFamily.CURRENT_TIME),
        ("can you tell me the time?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("can you give me the current time?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("got the time?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("do you know what time it is?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("what does the clock say?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("clock time now?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("¿Qué hora es?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("dime la hora actual", LiveStateEvidenceFamily.CURRENT_TIME),
        ("Quelle heure est-il ?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("donne-moi l'heure actuelle", LiveStateEvidenceFamily.CURRENT_TIME),
        ("Wie spät ist es?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("aktuelle Uhrzeit bitte", LiveStateEvidenceFamily.CURRENT_TIME),
        ("Che ore sono?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("dimmi l'ora attuale", LiveStateEvidenceFamily.CURRENT_TIME),
        ("Que horas são?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("qual é a hora atual?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("котра година?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("скільки зараз часу?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("która jest godzina?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("jaki jest aktualny czas?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("Saat kaç?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("şu an saat kaç?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("كم الساعة؟", LiveStateEvidenceFamily.CURRENT_TIME),
        ("今何時ですか", LiveStateEvidenceFamily.CURRENT_TIME),
        ("现在几点？", LiveStateEvidenceFamily.CURRENT_TIME),
        ("через сколько дней Рождество?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("Сколько дней до Рождества?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("сколько секунд до нового года?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("сколько секунд прошло с последнего Рождества?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("посчитай секунды с последнего Рождества", LiveStateEvidenceFamily.CURRENT_TIME),
        ("how many seconds since Christmas?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("seconds since last Christmas?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("how many seconds until Christmas?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("what is the date right now?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("what does time.time return right now?", LiveStateEvidenceFamily.CURRENT_TIME),
        ("какое сегодня число?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("какое сейчас число?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("какой сегодня день?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("какой сейчас день?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("какой в данный момент день?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("какой день сейчас?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("какой день сегодня?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("какой день в данный момент?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("what day is it?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("what day is it today?", LiveStateEvidenceFamily.CURRENT_DATE),
        ("what is the local date?", LiveStateEvidenceFamily.CURRENT_DATE),
    ],
)
def test_live_state_evidence_plan_accepts_now_modifier_with_time_or_date_noun(
    user_input: str,
    expected_family: LiveStateEvidenceFamily,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(),
    )

    assert plan.family is expected_family
    if "рождеств" in user_input.lower() or "christmas" in user_input.lower():
        assert plan.candidate_tool_names == frozenset()
        assert plan.unavailable_reason == "live_state_tool_unavailable"
    else:
        assert plan.candidate_tool_names == frozenset({"datetime.now"})


@pytest.mark.parametrize("user_input", ["сейчас?", "в данный момент?", "right now?"])
def test_live_state_evidence_plan_does_not_treat_now_markers_as_standalone_intent(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(),
    )

    assert plan.evidence_required is False
    assert plan.family is None
    assert plan.candidate_tool_names == frozenset()


def test_direct_live_state_family_detection_normalizes_like_plan_helper() -> None:
    assert detect_live_state_family("what is a network interface?") is None


@pytest.mark.parametrize(
    "user_input",
    [
        "сколько времени занимает merge sort?",
        "how long does checking CPU usage take now?",
        "how long to check current CPU usage now?",
        "how long to check CPU and memory usage now?",
        "how long to check CPU and current memory usage now?",
        "how much time does checking CPU usage take now?",
        "что значит который час?",
        "show Python datetime.now examples",
        "show examples of what time is it right now",
        "write Python code for what time is it right now",
        "write a Python script to show current CPU usage",
        "write Python code to show CPU and memory usage",
        "write Python code to show CPU and current memory usage",
        "write Python code: is Ollama running?",
        "write Python code to check CPU and whether memory usage is above 80%",
        "write Python code to check CPU and check memory usage",
        "write Python code: is memory usage above 80%",
        "write Python code to check CPU and if memory usage is above 80%",
        "write Python code to check CPU and when memory usage is above 80%",
        "show me code to get current CPU usage",
        "show code to get current CPU usage",
        "show code to get CPU and memory usage",
        "show code for what time is it right now",
        "give me code for what time is it right now",
        "write Python code to calculate hours between 2025-09-01T00:00:00+03:00 and 2026-06-07T20:17:00+03:00",
        "give me a Python snippet to show current CPU usage",
        "show me a snippet to get current CPU usage",
        "give me a shell script to show current CPU usage",
        "how can I check current CPU usage in Python",
        "how to check current CPU usage in Python",
        "read the system time from the logs",
        "show Python CPU usage examples",
        "show current CPU usage examples",
        "remind me to check CPU usage later",
        "remind me to check CPU and memory usage later",
        "what cron should I use to log memory usage?",
        "set a timer to check battery status",
        "what is time complexity?",
        "какое время поставить в cron для запуска?",
        "поставь таймер на 5 минут",
        "what is CPU?",
        "explain network basics",
        "what is a daemon?",
        "what is a process?",
        "what is a service?",
        "what is a PID?",
        "what is a PID controller?",
        "what is a process manager?",
        "what is pgrep?",
        "explain ps command",
        "what is a temperature sensor?",
        "what is VPN?",
        "что такое нагрузка?",
        "what is an IP address?",
        "what is a network interface?",
        "what is a network interface right now?",
        "show CPU usage from logs",
        "read current CPU usage from logs",
        "show CPU usage from logs right now",
        "what was memory usage yesterday right now",
        "what was memory usage yesterday?",
        "what does CPU usage mean?",
        "what does CPU usage mean right now?",
        "what is CPU usage?",
        "what is CPU utilization?",
        "what is CPU and memory usage?",
        "what is memory usage?",
        "what's CPU usage?",
        "what does CPU and memory usage mean?",
        "what does disk usage mean?",
        "how does CPU usage work right now?",
        "how does a battery work?",
        "как работает аккумулятор макбука?",
    ],
)
def test_live_state_evidence_plan_excludes_non_live_near_misses(user_input: str) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "datetime.now",
            "tool.system.read.resources",
            live_state_tool_names=("datetime.now", "tool.system.read.resources"),
        ),
        tool_observation_refs=(),
    )

    assert plan.evidence_required is False
    assert plan.family is None
    assert plan.candidate_tool_names == frozenset()


@pytest.mark.parametrize(
    ("user_input", "expected_family", "expected_tool"),
    [
        (
            "what is current CPU usage?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "what is the CPU usage?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "сколько сейчас памяти занято?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "сколько сейчас CPU загружено?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "мой VPN подключен?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "какой внешний IP?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "am I online right now?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "is Wi-Fi connected?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "is the internet connected?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "интернет подключен?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "как работает VPN сейчас?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "джарвис включен ли vpn сейчас",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "как работает интернет в данный момент?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "is VPN working right now?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "what is VPN status?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "what is network status?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "is the internet working?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "am I connected to the internet?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "is my network down?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "is wifi on?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "What is listening on port 8080?",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "Run netstat style network diagnostics",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "Покажи, кто слушает порт 8080",
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            "tool.system.read.network",
        ),
        (
            "какая температура CPU?",
            LiveStateEvidenceFamily.SYSTEM_SENSORS,
            "tool.system.read.sensors",
        ),
        (
            "what is the CPU temperature?",
            LiveStateEvidenceFamily.SYSTEM_SENSORS,
            "tool.system.read.sensors",
        ),
        (
            "Show thermal sensor state on this laptop",
            LiveStateEvidenceFamily.SYSTEM_SENSORS,
            "tool.system.read.sensors",
        ),
        (
            "what processor do I have?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "какой у меня процессор?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "какой процессор у меня?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "how much RAM do I have?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "какой процессор на этом Mac?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "What is the OS version?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "Which macOS build am I on?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "Какая версия операционной системы?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "Какой у меня macOS билд?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "Покажи сборку macOS на этом компьютере",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "How many CPU cores are there?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "what is current system load?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "what is load average?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "what is current disk usage?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "how much disk space is available?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "Сколько свободного места на диске?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "Покажи свободное место на накопителе",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "Текущее свободное место на диске?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            "tool.system.read.resources",
        ),
        (
            "статус демона?",
            LiveStateEvidenceFamily.DAEMON_STATUS,
            "daemon.status",
        ),
        (
            "what is the daemon status?",
            LiveStateEvidenceFamily.DAEMON_STATUS,
            "daemon.status",
        ),
        (
            "is the daemon up?",
            LiveStateEvidenceFamily.DAEMON_STATUS,
            "daemon.status",
        ),
        (
            "is the daemon down?",
            LiveStateEvidenceFamily.DAEMON_STATUS,
            "daemon.status",
        ),
        (
            "daemon health?",
            LiveStateEvidenceFamily.DAEMON_STATUS,
            "daemon.status",
        ),
        (
            "what is current runtime status?",
            LiveStateEvidenceFamily.DAEMON_STATUS,
            "daemon.status",
        ),
        (
            "How much battery is left on this MacBook?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "what is the battery status?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "what is battery status?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "сколько battery осталось на macbook",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
        (
            "Сколько заряда батареи сейчас на макбуке?",
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            "tool.system.read.hardware",
        ),
    ],
)
def test_live_state_evidence_plan_detects_system_families(
    user_input: str,
    expected_family: LiveStateEvidenceFamily,
    expected_tool: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            expected_tool,
            live_state_tool_names=(expected_tool,),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is expected_family
    assert plan.families == frozenset({expected_family})
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset({expected_tool})
    assert plan.missing_tool_names == frozenset({expected_tool})
    assert plan.unavailable_reason is None


def test_live_state_evidence_plan_tracks_multiple_live_state_families() -> None:
    plan = live_state_evidence_plan(
        _request("what time is it and what is current CPU usage?"),
        _plan(
            "datetime.now",
            "tool.system.read.resources",
            live_state_tool_names=("datetime.now", "tool.system.read.resources"),
        ),
        tool_observation_refs=(),
    )

    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.CURRENT_TIME,
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
        }
    )
    assert plan.candidate_tool_names == frozenset(
        {"datetime.now", "tool.system.read.resources"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_keeps_script_process_clause_in_mixed_live_request() -> None:
    plan = live_state_evidence_plan(
        _request("is server.py running and what is current CPU usage?"),
        _plan(
            "tool.system.read.process",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.process", "tool.system.read.resources"),
        ),
        tool_observation_refs=(),
    )

    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_PROCESS,
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
        }
    )
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.process", "tool.system.read.resources"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


@pytest.mark.parametrize(
    "user_input",
    [
        "is the wifi up?",
        "is the internet up?",
        "is VPN running?",
    ],
)
def test_live_state_evidence_plan_network_status_does_not_require_process_evidence(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.network",
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.network", "tool.system.read.process"),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_NETWORK
    assert plan.families == frozenset({LiveStateEvidenceFamily.SYSTEM_NETWORK})
    assert plan.candidate_tool_names == frozenset({"tool.system.read.network"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.network"})


def test_live_state_evidence_plan_requires_process_observation_for_process_status() -> None:
    request = _request("is Ollama running?")
    request_plan = _plan(
        "tool.system.read.process",
        live_state_tool_names=("tool.system.read.process",),
    )
    missing_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    observed_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "matches": [{"pid": 1234, "name": "Ollama"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert missing_plan.family is not None
    assert missing_plan.family.value == "system_process"
    assert missing_plan.families == frozenset({missing_plan.family})
    assert missing_plan.candidate_tool_names == frozenset({"tool.system.read.process"})
    assert missing_plan.missing_tool_names == frozenset({"tool.system.read.process"})
    assert missing_plan.missing_families == frozenset({missing_plan.family})
    assert observed_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_named_third_party_daemon_requires_process_observation() -> None:
    plan = live_state_evidence_plan(
        _request("is the Ollama daemon running?"),
        _plan(
            "daemon.status",
            "tool.system.read.process",
            live_state_tool_names=("daemon.status", "tool.system.read.process"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "daemon.status",
                structured_schema="daemon.status",
                structured_content={"status": "running"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.families == frozenset({LiveStateEvidenceFamily.SYSTEM_PROCESS})
    assert plan.candidate_tool_names == frozenset({"tool.system.read.process"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_generic_named_daemon_requires_process_observation() -> None:
    plan = live_state_evidence_plan(
        _request("is the ClickHouse daemon running?"),
        _plan(
            "daemon.status",
            "tool.system.read.process",
            live_state_tool_names=("daemon.status", "tool.system.read.process"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "daemon.status",
                structured_schema="daemon.status",
                structured_content={"status": "running"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.candidate_tool_names == frozenset({"tool.system.read.process"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_named_daemon_subject_first_keeps_identity() -> None:
    unrelated_plan = live_state_evidence_plan(
        _request("ClickHouse daemon is running?"),
        _plan(
            "daemon.status",
            "tool.system.read.process",
            live_state_tool_names=("daemon.status", "tool.system.read.process"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [{"pid": 123, "name": "MongoDB"}],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    matching_plan = live_state_evidence_plan(
        _request("ClickHouse daemon is running?"),
        _plan(
            "daemon.status",
            "tool.system.read.process",
            live_state_tool_names=("daemon.status", "tool.system.read.process"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [{"pid": 123, "name": "clickhouse-server"}],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert unrelated_plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert unrelated_plan.missing_tool_names == frozenset({"tool.system.read.process"})
    assert matching_plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert matching_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_named_daemon_status_rejects_unrelated_process_snapshot() -> None:
    plan = live_state_evidence_plan(
        _request("Ollama daemon status"),
        _plan(
            "daemon.status",
            "tool.system.read.process",
            live_state_tool_names=("daemon.status", "tool.system.read.process"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [{"pid": 123, "name": "MongoDB"}],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_named_daemon_definition_is_near_miss() -> None:
    plan = live_state_evidence_plan(
        _request("what is the Docker daemon?"),
        _plan(
            "daemon.status",
            "tool.system.read.process",
            live_state_tool_names=("daemon.status", "tool.system.read.process"),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is None
    assert plan.evidence_required is False


def test_live_state_evidence_plan_rejects_mismatched_process_observation() -> None:
    plan = live_state_evidence_plan(
        _request("is Ollama running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Redis"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Redis",
                    "matches": [{"pid": 2345, "name": "redis-server"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


@pytest.mark.parametrize(
    ("user_input", "expected_families", "expected_tools"),
    [
        (
            "VPN and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_NETWORK, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.network", "daemon.status"},
        ),
        (
            "daemon and VPN status",
            {LiveStateEvidenceFamily.SYSTEM_NETWORK, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.network", "daemon.status"},
        ),
        (
            "Wi-Fi and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_NETWORK, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.network", "daemon.status"},
        ),
        (
            "internet and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_NETWORK, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.network", "daemon.status"},
        ),
        (
            "network and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_NETWORK, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.network", "daemon.status"},
        ),
        (
            "my VPN and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_NETWORK, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.network", "daemon.status"},
        ),
        (
            "battery and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_HARDWARE, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.hardware", "daemon.status"},
        ),
        (
            "disk and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_RESOURCES, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.resources", "daemon.status"},
        ),
        (
            "hardware and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_HARDWARE, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.hardware", "daemon.status"},
        ),
        (
            "sensors and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_SENSORS, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.sensors", "daemon.status"},
        ),
        (
            "temperature and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_SENSORS, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.sensors", "daemon.status"},
        ),
        (
            "thermal and daemon status",
            {LiveStateEvidenceFamily.SYSTEM_SENSORS, LiveStateEvidenceFamily.DAEMON_STATUS},
            {"tool.system.read.sensors", "daemon.status"},
        ),
    ],
)
def test_live_state_evidence_plan_tracks_shared_status_suffix_families(
    user_input: str,
    expected_families: set[LiveStateEvidenceFamily],
    expected_tools: set[str],
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.network",
            "tool.system.read.resources",
            "tool.system.read.hardware",
            "tool.system.read.sensors",
            "daemon.status",
            live_state_tool_names=(
                "tool.system.read.network",
                "tool.system.read.resources",
                "tool.system.read.hardware",
                "tool.system.read.sensors",
                "daemon.status",
            ),
        ),
        tool_observation_refs=(),
    )

    assert plan.families == frozenset(expected_families)
    assert plan.candidate_tool_names == frozenset(expected_tools)
    assert plan.missing_tool_names == plan.candidate_tool_names


@pytest.mark.parametrize(
    "user_input",
    [
        "what is current CPU usage now compared to yesterday?",
        "is CPU usage higher now than yesterday?",
        "is CPU usage higher than yesterday?",
        "compare current memory usage to yesterday",
    ],
)
def test_live_state_evidence_plan_keeps_current_observation_in_history_comparison(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.families == frozenset({LiveStateEvidenceFamily.SYSTEM_RESOURCES})
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


def test_live_state_evidence_plan_keeps_missing_tools_for_unobserved_family() -> None:
    plan = live_state_evidence_plan(
        _request("what time is it and what is current CPU usage?"),
        _plan(
            "datetime.now",
            "tool.system.read.resources",
            live_state_tool_names=("datetime.now", "tool.system.read.resources"),
        ),
        tool_observation_refs=(_completed_resource_ref(),),
    )

    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.CURRENT_TIME,
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
        }
    )
    assert plan.missing_tool_names == frozenset({"datetime.now"})
    assert plan.missing_families == frozenset({LiveStateEvidenceFamily.CURRENT_TIME})


@pytest.mark.parametrize(
    "user_input",
    [
        "what does CPU usage mean, and what is current memory usage?",
        "what does CPU usage mean but what is current memory usage?",
        "what is current CPU usage, and what does CPU usage mean?",
        "Explain CPU usage. What is current memory usage?",
        "Explain CPU usage\nWhat is current memory usage?",
        "what does CPU usage mean, what is current memory usage?",
        "show Python CPU usage examples and what is current memory usage?",
        "read CPU usage from logs and show current memory usage",
        "what does CPU usage mean, and what is the CPU usage?",
    ],
)
def test_live_state_evidence_plan_keeps_live_clause_in_mixed_near_miss_requests(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.families == frozenset({LiveStateEvidenceFamily.SYSTEM_RESOURCES})
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


@pytest.mark.parametrize(
    "user_input",
    [
        "what does CPU usage mean, and what is current VPN status?",
        "what does CPU usage mean, what is current VPN status?",
    ],
)
def test_live_state_evidence_plan_does_not_leak_near_miss_family_into_live_clause(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.network",
            "tool.system.read.resources",
            live_state_tool_names=(
                "tool.system.read.network",
                "tool.system.read.resources",
            ),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_NETWORK
    assert plan.families == frozenset({LiveStateEvidenceFamily.SYSTEM_NETWORK})
    assert plan.candidate_tool_names == frozenset({"tool.system.read.network"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.network"})


def test_live_state_evidence_plan_does_not_leak_near_miss_threshold_into_live_math() -> None:
    plan = live_state_evidence_plan(
        _request("what does CPU usage above 80% mean, and what is current VPN status?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.network",
            "tool.system.read.resources",
            live_state_tool_names=(
                "tool.system.read.network",
                "tool.system.read.resources",
            ),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_NETWORK
    assert plan.families == frozenset({LiveStateEvidenceFamily.SYSTEM_NETWORK})
    assert plan.candidate_tool_names == frozenset({"tool.system.read.network"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.network"})


@pytest.mark.parametrize(
    "user_input",
    [
        "what is a network interface, and what is current CPU usage?",
        "what is a network interface\nwhat is current CPU usage?",
        "what is VPN, and what is current CPU usage?",
    ],
)
def test_live_state_evidence_plan_does_not_leak_concept_clause_family(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.network",
            "tool.system.read.resources",
            live_state_tool_names=(
                "tool.system.read.network",
                "tool.system.read.resources",
            ),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.families == frozenset({LiveStateEvidenceFamily.SYSTEM_RESOURCES})
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


def test_live_state_evidence_plan_does_not_leak_unrelated_profile_clause_family() -> None:
    plan = live_state_evidence_plan(
        _request("VPN profile, what is current CPU usage?"),
        _plan(
            "tool.system.read.network",
            "tool.system.read.resources",
            live_state_tool_names=(
                "tool.system.read.network",
                "tool.system.read.resources",
            ),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.families == frozenset({LiveStateEvidenceFamily.SYSTEM_RESOURCES})
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


def test_live_state_evidence_plan_keeps_threshold_clause_in_mixed_near_miss_request() -> None:
    plan = live_state_evidence_plan(
        _request("what does CPU usage mean, and is memory usage above 80%?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert LiveStateEvidenceFamily.SYSTEM_RESOURCES in plan.families
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_keeps_newline_threshold_clause_in_mixed_near_miss_request() -> None:
    plan = live_state_evidence_plan(
        _request("Explain CPU usage\nis memory usage above 80%?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert LiveStateEvidenceFamily.SYSTEM_RESOURCES in plan.families
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_keeps_colon_threshold_clause_in_mixed_near_miss_request() -> None:
    plan = live_state_evidence_plan(
        _request("Explain CPU usage: is memory usage above 80%?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert LiveStateEvidenceFamily.SYSTEM_RESOURCES in plan.families
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_keeps_semicolon_threshold_clause_in_mixed_near_miss_request() -> None:
    plan = live_state_evidence_plan(
        _request("Explain CPU usage; is memory usage above 80%?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert LiveStateEvidenceFamily.SYSTEM_RESOURCES in plan.families
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_links_followup_threshold_clause_to_prior_live_family() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU load? Is it greater than 10*e?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert LiveStateEvidenceFamily.SYSTEM_RESOURCES in plan.families
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_accepts_decimal_calculator_observation() -> None:
    plan = live_state_evidence_plan(
        _request("is CPU load greater than 1.5*e?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_resource_ref(),
            _completed_ref("calculator.evaluate", arguments={"expression": "1.5*e"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_accepts_decimal_comma_calculator_observation() -> None:
    plan = live_state_evidence_plan(
        _request("is CPU load greater than 1,5*e?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_resource_ref(),
            _completed_ref("calculator.evaluate", arguments={"expression": "1,5*e"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_does_not_leak_new_year_target_from_near_miss_clause() -> None:
    plan = live_state_evidence_plan(
        _request("what does New Year mean, and how many seconds until Christmas?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.families == frozenset({LiveStateEvidenceFamily.CURRENT_TIME})
    assert plan.candidate_tool_names == frozenset()
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_evidence_plan_scopes_countdown_candidates_per_live_family_clause() -> None:
    plan = live_state_evidence_plan(
        _request(
            "what is current VPN status for New Year profile, "
            "and how many seconds until Christmas?"
        ),
        _plan(
            "datetime.now",
            "datetime.until",
            "tool.system.read.network",
            live_state_tool_names=(
                "datetime.now",
                "datetime.until",
                "tool.system.read.network",
            ),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_NETWORK
    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
            LiveStateEvidenceFamily.CURRENT_TIME,
        }
    )
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.network"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names
    assert plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_evidence_plan_does_not_accept_until_for_different_countdown_clause() -> None:
    plan = live_state_evidence_plan(
        _request("how many seconds until Christmas, and how many seconds until New Year?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.until",
                structured_schema="datetime.until",
                structured_content={
                    "from_iso": "2026-06-05T20:59:07+03:00",
                    "target": "next_new_year",
                    "unit": "seconds",
                    "value": 18337521,
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_evidence_plan_does_not_accept_until_from_unrelated_live_clause() -> None:
    plan = live_state_evidence_plan(
        _request(
            "what is current VPN status for New Year profile, "
            "and how many seconds until Christmas?"
        ),
        _plan(
            "datetime.now",
            "datetime.until",
            "tool.system.read.network",
            live_state_tool_names=(
                "datetime.now",
                "datetime.until",
                "tool.system.read.network",
            ),
        ),
        tool_observation_refs=(
            _completed_ref(
                "datetime.until",
                arguments={"target": "next_new_year", "unit": "seconds"},
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_NETWORK
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.network"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names
    assert plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_evidence_plan_keeps_russian_but_live_clause_in_mixed_near_miss_request() -> None:
    plan = live_state_evidence_plan(
        _request("что такое нагрузка CPU, но какая сейчас нагрузка CPU?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.families == frozenset({LiveStateEvidenceFamily.SYSTEM_RESOURCES})
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


@pytest.mark.parametrize(
    "user_input",
    [
        "через сколько дней Рождество?",
        "сколько секунд до нового года?",
        "сколько секунд прошло с последнего Рождества?",
        "how many seconds until Christmas?",
        "how many seconds since Christmas?",
    ],
)
def test_live_state_evidence_plan_detects_countdown_and_elapsed_time_requests(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.evidence_required is True
    if "нового года" in user_input.lower():
        assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
        assert plan.missing_tool_names == plan.candidate_tool_names
    else:
        assert plan.candidate_tool_names == frozenset()
        assert plan.missing_tool_names == frozenset()
        assert plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_evidence_plan_tracks_cpu_overview_hardware_and_resources() -> None:
    plan = _multi_family_plan("Show CPU core count and current utilization")

    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
        }
    )
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.hardware", "tool.system.read.resources"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_tracks_busy_cpu_overview_hardware_and_resources() -> None:
    plan = _multi_family_plan("How many CPU cores are there and how busy are they?")

    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
        }
    )
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.hardware", "tool.system.read.resources"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


@pytest.mark.parametrize(
    "user_input",
    [
        "Сколько ядер у центрального процессора и на сколько они загружены?",
        "Покажи загрузку CPU и количество ядер",
    ],
)
def test_live_state_evidence_plan_tracks_ru_cpu_overview_hardware_and_resources(
    user_input: str,
) -> None:
    plan = _multi_family_plan(user_input)

    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_HARDWARE,
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
        }
    )
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.hardware", "tool.system.read.resources"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


def _multi_family_plan(user_input: str) -> LiveStateEvidencePlan:
    return live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.hardware",
            "tool.system.read.resources",
            live_state_tool_names=(
                "tool.system.read.hardware",
                "tool.system.read.resources",
            ),
        ),
        tool_observation_refs=(),
    )


@pytest.mark.parametrize(
    "user_input",
    [
        "what does datetime.now return right now?",
        "сколько сейчас времени для таймера?",
    ],
)
def test_live_state_evidence_plan_near_miss_exclusions_keep_current_observation(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})


def test_live_state_evidence_plan_requires_live_state_and_calculator_for_math() -> None:
    plan = live_state_evidence_plan(
        _request("is CPU load greater than 10*e"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


@pytest.mark.parametrize(
    "user_input",
    [
        "is current CPU usage over 80%?",
        "is current memory usage greater than 8GB?",
        "CPU load above 90%?",
        "is current CPU usage > 80%?",
        "memory usage <= 8GB?",
        "is current CPU usage 80% or higher?",
        "is memory usage 8GB or less?",
        "CPU usage 80%+?",
        "is CPU over 80%?",
        "is battery over 80%?",
        "is system load over 2?",
        "is current memory usage at most 8GB?",
        "is current memory usage not more than 8GB?",
        "is current memory usage between 4GB and 8GB?",
        "is current memory usage equal to 8GB?",
        "нагрузка CPU больше 80%?",
        "использование памяти не больше 8GB?",
        "использование памяти не более 8GB?",
        "использование памяти как максимум 8GB?",
    ],
)
def test_live_state_evidence_plan_treats_threshold_comparison_as_live_state_only(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is not LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert "calculator.evaluate" not in plan.candidate_tool_names
    assert "calculator.evaluate" not in plan.missing_tool_names
    assert plan.candidate_tool_names


def test_live_state_evidence_plan_clears_threshold_comparison_with_live_observation_only() -> None:
    plan = live_state_evidence_plan(
        _request("is current CPU usage over 80%?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_resource_ref(),
            _completed_ref("calculator.evaluate", arguments={"expression": "72 > 80"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_ignores_threshold_calculator_mismatch_after_live_observation() -> None:
    plan = live_state_evidence_plan(
        _request("is current CPU usage over 80%?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_resource_ref(),
            _completed_ref("calculator.evaluate", arguments={"expression": "72 > 70"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.missing_tool_names == frozenset()


@pytest.mark.parametrize("expression", ["80", "80+1", "180-100", "1 < 80", "80 == 80", "80 > 1"])
def test_live_state_evidence_plan_ignores_threshold_calculator_non_comparison_after_live_observation(
    expression: str,
) -> None:
    plan = live_state_evidence_plan(
        _request("is current CPU usage over 80%?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_resource_ref(),
            _completed_ref("calculator.evaluate", arguments={"expression": expression}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.missing_tool_names == frozenset()


@pytest.mark.parametrize(
    ("user_input", "expected_family"),
    [
        (
            "is the Python process memory usage greater than 10*e?",
            LiveStateEvidenceFamily.LIVE_STATE_MATH,
        ),
        (
            "what is current CPU usage of the Python process?",
            LiveStateEvidenceFamily.SYSTEM_PROCESS,
        ),
        (
            "is daemon runtime CPU load greater than 10*e?",
            LiveStateEvidenceFamily.LIVE_STATE_MATH,
        ),
    ],
)
def test_live_state_evidence_plan_near_miss_exclusions_do_not_hide_system_math(
    user_input: str,
    expected_family: LiveStateEvidenceFamily,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is expected_family
    if "process" in user_input.lower():
        assert plan.candidate_tool_names == frozenset()
        assert plan.unavailable_reason == "live_state_tool_unavailable"
    else:
        assert "tool.system.read.resources" in plan.candidate_tool_names
    if (
        expected_family is LiveStateEvidenceFamily.LIVE_STATE_MATH
        and "tool.system.read.resources" in plan.candidate_tool_names
    ):
        assert "calculator.evaluate" in plan.candidate_tool_names


def test_live_state_evidence_plan_clears_missing_tools_after_matching_observations() -> None:
    request = _request("is CPU load greater than 10*e")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            _completed_resource_ref(),
            _completed_ref("calculator.evaluate", arguments={"expression": "10*e"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.evidence_required is True
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_disk_request_requires_disk_resource_observation() -> None:
    default_resource_plan = live_state_evidence_plan(
        _request("what is current disk space available?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"metric": "cpu_and_memory"},
                structured_schema="system.resource_overview",
                structured_content={"cpu": {"used_percent": 10.2}},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    disk_resource_plan = live_state_evidence_plan(
        _request("what is current disk space available?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["df", "-h"]},
                structured_schema="system.disk_free",
                structured_content={"filesystems": []},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert default_resource_plan.missing_tool_names == frozenset({"tool.system.read.resources"})
    assert disk_resource_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_cpu_and_disk_request_requires_relevant_resource_observations() -> None:
    disk_only_plan = live_state_evidence_plan(
        _request("what is current CPU usage and disk space available?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["df", "-h"]},
                structured_schema="system.disk_free",
                structured_content={"filesystems": []},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    complete_plan = live_state_evidence_plan(
        _request("what is current CPU usage and disk space available?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"metric": "cpu_and_memory"},
                structured_schema="system.resource_overview",
                structured_content={"cpu": {"used_percent": 10.2}},
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["df", "-h"]},
                structured_schema="system.disk_free",
                structured_content={"filesystems": []},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert disk_only_plan.missing_tool_names == frozenset({"tool.system.read.resources"})
    assert complete_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_ru_system_summary_requires_storage_and_network_evidence() -> None:
    request = _request(
        "Предоставь сводку данных о текущем состоянии системы, включая нагрузку "
        "процессора, занятую память, активность сети и процент свободного хранилища."
    )
    request_plan = _plan(
        "tool.system.read.resources",
        "tool.system.read.network",
        live_state_tool_names=(
            "tool.system.read.resources",
            "tool.system.read.network",
        ),
    )
    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    partial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"metric": "cpu_and_memory"},
                structured_schema="system.resource_overview",
                structured_content={
                    "cpu": {"used_percent": 10.2},
                    "memory": {"used_percent": 62.5},
                },
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ifconfig"]},
                structured_schema="system.network_interfaces",
                structured_content={"interfaces": []},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    complete_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"metric": "cpu_and_memory"},
                structured_schema="system.resource_overview",
                structured_content={
                    "cpu": {"used_percent": 10.2},
                    "memory": {"used_percent": 62.5},
                },
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["df", "-h"]},
                structured_schema="system.disk_free",
                structured_content={"filesystems": []},
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ifconfig"]},
                structured_schema="system.network_interfaces",
                structured_content={"interfaces": []},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert initial_plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            LiveStateEvidenceFamily.SYSTEM_NETWORK,
        }
    )
    assert initial_plan.candidate_tool_names == frozenset(
        {"tool.system.read.resources", "tool.system.read.network"}
    )
    assert partial_plan.missing_tool_names == frozenset({"tool.system.read.resources"})
    assert partial_plan.missing_families == frozenset({LiveStateEvidenceFamily.SYSTEM_RESOURCES})
    assert complete_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_cpu_and_memory_request_requires_both_resource_subtypes() -> None:
    memory_only_plan = live_state_evidence_plan(
        _request("what is current CPU usage and memory usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["free", "-m"]},
                structured_schema="system.memory_overview",
                structured_content={"available": "18000 MiB"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    cpu_only_plan = live_state_evidence_plan(
        _request("what is current CPU usage and memory usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["top", "-l", "1", "-n", "0"]},
                structured_schema="system.cpu_overview",
                structured_content={"used_percent": 20.0},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    complete_plan = live_state_evidence_plan(
        _request("what is current CPU usage and memory usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["top", "-l", "1", "-n", "0"]},
                structured_schema="system.cpu_overview",
                structured_content={"used_percent": 20.0},
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["free", "-m"]},
                structured_schema="system.memory_overview",
                structured_content={"available": "18000 MiB"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert memory_only_plan.missing_tool_names == frozenset({"tool.system.read.resources"})
    assert cpu_only_plan.missing_tool_names == frozenset({"tool.system.read.resources"})
    assert complete_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_process_resource_request_rejects_global_resource_observation() -> None:
    global_memory_plan = live_state_evidence_plan(
        _request("is the Python process memory usage greater than 10*e?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.process",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.process", "tool.system.read.resources"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["free", "-m"]},
                structured_schema="system.memory_overview",
                structured_content={"available": "18000 MiB"},
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref("calculator.evaluate", arguments={"expression": "10*e"}),
        ),
    )
    global_cpu_plan = live_state_evidence_plan(
        _request("what is current CPU usage of the Python process?"),
        _plan(
            "tool.system.read.process",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.process", "tool.system.read.resources"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["top", "-l", "1", "-n", "0"]},
                structured_schema="system.cpu_overview",
                structured_content={"used_percent": 20.0},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert global_memory_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert global_memory_plan.missing_tool_names == frozenset({"tool.system.read.process"})
    assert global_cpu_plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert global_cpu_plan.missing_tool_names == frozenset({"tool.system.read.process"})


@pytest.mark.parametrize(
    "user_input",
    [
        "what is current CPU usage of Python?",
        "what is current CPU usage of Chrome?",
        "what is current CPU usage of Chrome right now?",
        "what is current memory usage for qwen-server?",
        "How much CPU is Chrome using right now?",
        "How much memory is Ollama using?",
        "CPU usage of Google Chrome",
        "Google Chrome.app CPU usage",
        "Chrome CPU usage",
        "qwen-server memory usage",
    ],
)
def test_live_state_evidence_plan_named_process_resource_request_rejects_global_resources(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.process",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.process", "tool.system.read.resources"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["top", "-l", "1", "-n", "0"]},
                structured_schema="system.cpu_overview",
                structured_content={"used_percent": 20.0},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.candidate_tool_names == frozenset({"tool.system.read.process"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_named_process_using_threshold_requires_process_evidence_only() -> None:
    plan = live_state_evidence_plan(
        _request("Is Chrome using more than 10% CPU?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert "tool.system.read.process" in plan.candidate_tool_names
    assert "tool.system.read.process" in plan.missing_tool_names
    assert "calculator.evaluate" not in plan.candidate_tool_names
    assert "calculator.evaluate" not in plan.missing_tool_names


def test_live_state_evidence_plan_named_process_resource_accepts_ps_aux_snapshot() -> None:
    plans = [
        live_state_evidence_plan(
            _request(user_input),
            _plan(
                "tool.system.read.process",
                live_state_tool_names=("tool.system.read.process",),
            ),
            tool_observation_refs=(
                _completed_ref(
                    "tool.system.read.process",
                    arguments={"argv": ["ps", "aux"]},
                    structured_schema="system.process_resource_snapshot",
                    structured_content={
                        "processes": [
                            {
                                "pid": 123,
                                "name": "Google Chrome",
                                "cpu_percent": 12.5,
                                "memory_percent": 3.2,
                            },
                        ],
                        "source": "ps",
                    },
                    parse_status=ToolParseStatus.PARSED,
                ),
            ),
        )
        for user_input in (
            "Chrome CPU usage",
            "what is current CPU usage of Chrome right now?",
            "How much CPU is Chrome using right now?",
        )
    ]

    assert [plan.family for plan in plans] == [LiveStateEvidenceFamily.SYSTEM_PROCESS] * 3
    assert [plan.missing_tool_names for plan in plans] == [frozenset()] * 3


def test_live_state_evidence_plan_named_script_resource_accepts_ps_aux_command_name() -> None:
    plan = live_state_evidence_plan(
        _request("server.py CPU usage"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [
                        {
                            "pid": 123,
                            "name": "python",
                            "command_name": "server.py",
                            "cpu_percent": 12.5,
                            "memory_percent": 3.2,
                        },
                    ],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_named_process_resource_rejects_mismatched_ps_aux_snapshot() -> None:
    plan = live_state_evidence_plan(
        _request("Chrome CPU usage"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [
                        {
                            "pid": 123,
                            "name": "MongoDB",
                            "cpu_percent": 12.5,
                            "memory_percent": 3.2,
                        },
                    ],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_named_process_resource_accepts_versioned_executable_name() -> None:
    plan = live_state_evidence_plan(
        _request("Python process CPU usage"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [
                        {"pid": 321, "name": "python3.11", "cpu_percent": 7.5},
                    ],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.missing_tool_names == frozenset()


@pytest.mark.parametrize(
    "user_input",
    [
        "is CPU load greater than 10*e",
        "is memory usage 8GB or less?",
        "what processor do I have?",
    ],
)
def test_live_state_evidence_plan_named_process_resource_parser_keeps_system_requests_global(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "calculator.evaluate",
            "tool.system.read.hardware",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources", "tool.system.read.hardware"),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is not LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert "tool.system.read.process" not in plan.candidate_tool_names


def test_live_state_evidence_plan_process_resource_request_requires_process_tool() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage of the Python process?"),
        _plan(
            "tool.system.read.process",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.process", "tool.system.read.resources"),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["top", "-l", "1", "-n", "0"]},
                structured_schema="system.cpu_overview",
                structured_content={"used_percent": 20.0},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.candidate_tool_names == frozenset({"tool.system.read.process"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_resource_request_accepts_typed_process_resource_payload() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage of the Python process?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [
                        {
                            "pid": 3456,
                            "name": "Python",
                            "cpu_percent": 12.5,
                            "memory_percent": 1.2,
                        },
                    ],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_process_resource_request_rejects_mismatched_process_tool() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage of the Python process?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Redis"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Redis",
                    "matches": [{"pid": 2345, "name": "redis-server"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_resource_request_rejects_name_only_process_tool() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage of the Python process?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Python"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Python",
                    "matches": [{"pid": 3456, "name": "Python"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_status_rejects_query_with_unrelated_match() -> None:
    plan = live_state_evidence_plan(
        _request("is go running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "go"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "go",
                    "matches": [{"pid": 4567, "name": "mongo"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_status_rejects_partial_process_search() -> None:
    plan = live_state_evidence_plan(
        _request("is Ollama running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "matches": [],
                    "error": "pgrep failed",
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARTIAL,
                metadata={"exit_code": 2},
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_status_rejects_query_without_matches_field() -> None:
    plan = live_state_evidence_plan(
        _request("is Ollama running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_status_accepts_full_command_script_name() -> None:
    plan = live_state_evidence_plan(
        _request("is server.py running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-lf", "server.py"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "server.py",
                    "matches": [{"pid": 1234, "name": "python", "command_name": "server.py"}],
                    "source": "pgrep",
                    "match_mode": "full_command",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.missing_tool_names == frozenset()


@pytest.mark.parametrize(
    ("truncated", "metadata"),
    [
        (True, {}),
        (False, {"stdout_truncated": True}),
    ],
)
def test_live_state_evidence_plan_process_status_rejects_truncated_absence_search(
    truncated: bool,
    metadata: dict,
) -> None:
    plan = live_state_evidence_plan(
        _request("is Ollama running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "matches": [],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
                metadata=metadata,
                truncated=truncated,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_status_accepts_non_truncated_absence_search() -> None:
    plan = live_state_evidence_plan(
        _request("is Ollama running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "matches": [],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_process_status_rejects_structured_evidence_without_parsed_status() -> None:
    plan = live_state_evidence_plan(
        _request("is Ollama running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "matches": [{"pid": 1234, "name": "Ollama"}],
                    "source": "pgrep",
                },
                parse_status=None,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_status_rejects_generic_query_without_matches_field() -> None:
    plan = live_state_evidence_plan(
        _request("process status"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_generic_process_status_rejects_narrow_process_search() -> None:
    plan = live_state_evidence_plan(
        _request("process status"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "matches": [{"pid": 1234, "name": "Ollama"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_named_process_request_rejects_raw_query_without_structured_rows() -> None:
    plan = live_state_evidence_plan(
        _request("is go running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "go"]},
                content='{"exit_code": 0, "stdout": "4567 mongo\\n", "stderr": ""}',
                structured_content=None,
                structured_schema=None,
                parse_status=None,
                metadata={"exit_code": 0},
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_generic_process_request_rejects_raw_failed_content() -> None:
    plan = live_state_evidence_plan(
        _request("process status"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                content='{"exit_code": 2, "stdout": "", "stderr": "ps failed"}',
                structured_content=None,
                structured_schema=None,
                parse_status=None,
                metadata={},
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_generic_process_list_rejects_raw_ps_snapshot() -> None:
    plan = live_state_evidence_plan(
        _request("show process list"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                content='{"exit_code": 0, "stdout": "USER PID %CPU %MEM COMMAND\\n", "stderr": ""}',
                structured_content=None,
                structured_schema=None,
                parse_status=None,
                metadata={"exit_code": 0},
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_status_accepts_typed_process_resource_snapshot() -> None:
    chrome_plan = live_state_evidence_plan(
        _request("is Chrome running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [{"pid": 123, "name": "Google Chrome"}],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    pid_plan = live_state_evidence_plan(
        _request("is PID 123 running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [{"pid": 123, "name": "Google Chrome"}],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    process_pid_plan = live_state_evidence_plan(
        _request("is process 123 running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [{"pid": 123, "name": "Google Chrome"}],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert chrome_plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert chrome_plan.missing_tool_names == frozenset()
    assert pid_plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert pid_plan.missing_tool_names == frozenset()
    assert process_pid_plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert process_pid_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_process_status_accepts_multi_token_process_name() -> None:
    plan = live_state_evidence_plan(
        _request("is Google Chrome.app running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                structured_schema="system.process_resource_snapshot",
                structured_content={
                    "processes": [{"pid": 123, "name": "Google Chrome"}],
                    "source": "ps",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_PROCESS
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_generic_process_request_rejects_raw_ps_exit_code_one() -> None:
    plan = live_state_evidence_plan(
        _request("process status"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["ps", "aux"]},
                content='{"exit_code": 1, "stdout": "", "stderr": "ps failed"}',
                structured_content=None,
                structured_schema=None,
                parse_status=None,
                metadata={"exit_code": 1},
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_process_status_rejects_name_and_pid_from_different_rows() -> None:
    plan = live_state_evidence_plan(
        _request("is the Redis process PID 123 running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Redis"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Redis",
                    "matches": [{"pid": 123, "name": "Ollama"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_accepts_boundary_process_name_alias() -> None:
    plan = live_state_evidence_plan(
        _request("is service Redis running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Redis"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Redis",
                    "matches": [{"pid": 1234, "name": "redis-server"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset()


@pytest.mark.parametrize(
    "user_input",
    [
        "is service Redis running?",
        "is the process Redis running?",
    ],
)
def test_live_state_evidence_plan_rejects_mismatched_prefix_process_observation(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "matches": [{"pid": 1234, "name": "Ollama"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_rejects_mismatched_service_observation() -> None:
    plan = live_state_evidence_plan(
        _request("is the Redis service running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Ollama"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Ollama",
                    "matches": [{"pid": 1234, "name": "Ollama"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_rejects_substring_process_observation() -> None:
    plan = live_state_evidence_plan(
        _request("is go running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "mongo"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "mongo",
                    "matches": [{"pid": 4567, "name": "mongo"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_requires_process_name_and_pid_match() -> None:
    plan = live_state_evidence_plan(
        _request("is the Redis process PID 123 running?"),
        _plan(
            "tool.system.read.process",
            live_state_tool_names=("tool.system.read.process",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.process",
                arguments={"argv": ["pgrep", "-l", "Redis"]},
                structured_schema="system.process_name_search",
                structured_content={
                    "query": "Redis",
                    "matches": [{"pid": 999, "name": "Redis"}],
                    "source": "pgrep",
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is not None
    assert plan.family.value == "system_process"
    assert plan.missing_tool_names == frozenset({"tool.system.read.process"})


def test_live_state_evidence_plan_raw_diagnostics_requires_usable_observation_metadata() -> None:
    unavailable_plan = live_state_evidence_plan(
        _request("what is current CPU usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["top", "-l", "1", "-n", "0"]},
                structured_content=None,
                metadata={"unavailable": True, "source": "top"},
            ),
        ),
    )
    nonzero_plan = live_state_evidence_plan(
        _request("am I connected to the internet?"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ifconfig"]},
                structured_content={"stdout": ""},
                metadata={"exit_code": 1},
            ),
        ),
    )

    assert unavailable_plan.missing_tool_names == frozenset({"tool.system.read.resources"})
    assert nonzero_plan.missing_tool_names == frozenset({"tool.system.read.network"})


@pytest.mark.parametrize(
    ("user_input", "tool_name"),
    [
        ("покажи текущие ресурсы", "tool.system.read.resources"),
        ("network diagnostics", "tool.system.read.network"),
        ("hardware metadata", "tool.system.read.hardware"),
    ],
)
def test_live_state_evidence_plan_rejects_raw_success_without_retained_argv(
    user_input: str,
    tool_name: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(tool_name, live_state_tool_names=(tool_name,)),
        tool_observation_refs=(
            _completed_ref(
                tool_name,
                content='{"exit_code": 0, "stdout": "ok", "stderr": ""}',
                metadata={"exit_code": 0},
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({tool_name})


@pytest.mark.parametrize(
    ("user_input", "tool_name", "arguments"),
    [
        (
            "покажи текущие ресурсы",
            "tool.system.read.resources",
            {"argv": ["top", "-l", "1", "-n", "0"]},
        ),
        ("network diagnostics", "tool.system.read.network", {"argv": ["ifconfig"]}),
        ("hardware metadata", "tool.system.read.hardware", {"argv": ["sw_vers"]}),
    ],
)
def test_live_state_evidence_plan_rejects_raw_success_without_diagnostic_payload(
    user_input: str,
    tool_name: str,
    arguments: dict,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(tool_name, live_state_tool_names=(tool_name,)),
        tool_observation_refs=(
            _completed_ref(
                tool_name,
                arguments=arguments,
                content='{"exit_code": 0, "stdout": "", "stderr": ""}',
                metadata={"exit_code": 0},
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({tool_name})


def test_live_state_evidence_plan_rejects_raw_success_with_only_diagnostics_envelope_metadata() -> None:
    plan = live_state_evidence_plan(
        _request("network diagnostics"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ifconfig"]},
                content=(
                    '{"exit_code": 0, '
                    '"stdout": "", '
                    '"stderr": "", '
                    '"truncated": {"stdout": false, "stderr": false}}'
                ),
                metadata={"exit_code": 0},
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({"tool.system.read.network"})


@pytest.mark.parametrize(
    "ref_kwargs",
    [
        {"structured_content": {"foo": "bar"}},
        {"content": '{"exit_code": 0, "foo": "bar"}'},
    ],
)
def test_live_state_evidence_plan_rejects_raw_success_with_arbitrary_payload(
    ref_kwargs: dict,
) -> None:
    plan = live_state_evidence_plan(
        _request("network diagnostics"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ifconfig"]},
                metadata={"exit_code": 0},
                **ref_kwargs,
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({"tool.system.read.network"})


def test_live_state_evidence_plan_rejects_raw_success_with_structured_stdout_payload() -> None:
    plan = live_state_evidence_plan(
        _request("network diagnostics"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ifconfig"]},
                structured_content={"stdout": "en0: status: active"},
                metadata={"exit_code": 0},
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({"tool.system.read.network"})


@pytest.mark.parametrize(
    ("user_input", "tool_name", "wrong_schema"),
    [
        ("покажи текущие ресурсы", "tool.system.read.resources", "system.network_interfaces"),
        ("network diagnostics", "tool.system.read.network", "system.resource_overview"),
        ("hardware metadata", "tool.system.read.hardware", "system.resource_overview"),
    ],
)
def test_live_state_evidence_plan_broad_system_family_rejects_wrong_typed_schema(
    user_input: str,
    tool_name: str,
    wrong_schema: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(tool_name, live_state_tool_names=(tool_name,)),
        tool_observation_refs=(
            _completed_ref(
                tool_name,
                structured_schema=wrong_schema,
                structured_content={"status": "ok"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({tool_name})


def test_live_state_evidence_plan_unavailable_sensor_snapshot_does_not_clear_evidence() -> None:
    plan = live_state_evidence_plan(
        _request("what is the CPU temperature?"),
        _plan(
            "tool.system.read.sensors",
            live_state_tool_names=("tool.system.read.sensors",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.sensors",
                structured_schema="system.sensor_snapshot",
                structured_content={"available": False, "source": "powermetrics"},
                parse_status=ToolParseStatus.PARSED,
                metadata={"unavailable": True, "source": "powermetrics"},
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({"tool.system.read.sensors"})


def test_live_state_evidence_plan_generic_sensor_completion_does_not_clear_evidence() -> None:
    plan = live_state_evidence_plan(
        _request("what is the CPU temperature?"),
        _plan(
            "tool.system.read.sensors",
            live_state_tool_names=("tool.system.read.sensors",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.sensors",
                structured_schema=None,
                structured_content={"stdout": "CPU temperature unavailable"},
                metadata={"exit_code": 0},
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({"tool.system.read.sensors"})


def test_live_state_evidence_plan_schema_observation_requires_parse_status() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                structured_schema="system.cpu_overview",
                structured_content={"used_percent": 20.0},
                parse_status=None,
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


@pytest.mark.parametrize(
    ("user_input", "tool_name"),
    [
        ("покажи текущие ресурсы", "tool.system.read.resources"),
        ("network diagnostics", "tool.system.read.network"),
        ("hardware metadata", "tool.system.read.hardware"),
    ],
)
def test_live_state_evidence_plan_broad_system_request_rejects_unavailable_observation(
    user_input: str,
    tool_name: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(tool_name, live_state_tool_names=(tool_name,)),
        tool_observation_refs=(
            _completed_ref(
                tool_name,
                content='{"available": false, "reason": "backend_not_found"}',
                metadata={"unavailable": True, "source": "missing-backend"},
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({tool_name})


def test_live_state_evidence_plan_broad_network_request_rejects_nonzero_exit_without_retained_argv() -> None:
    plan = live_state_evidence_plan(
        _request("network diagnostics"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                content='{"exit_code": 1, "stdout": "", "stderr": "command failed"}',
                metadata={"exit_code": 1},
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset({"tool.system.read.network"})


@pytest.mark.parametrize(
    "user_input",
    [
        "what is my current local IP address?",
        "am I connected to the internet?",
    ],
)
def test_live_state_evidence_plan_ip_addr_schema_can_satisfy_local_network_evidence(
    user_input: str,
) -> None:
    plan = live_state_evidence_plan(
        _request(user_input),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ip", "addr"]},
                structured_schema="system.network_interfaces",
                structured_content={"interfaces": []},
                parse_status=ToolParseStatus.PARSED,
                metadata={"exit_code": 0},
            ),
        ),
    )

    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_disk_available_rejects_path_usage_observation() -> None:
    du_plan = live_state_evidence_plan(
        _request("what is current disk space available?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["du", "-sh", "."]},
                structured_content={"stdout": "12K ."},
            ),
        ),
    )
    df_plan = live_state_evidence_plan(
        _request("what is current disk space available?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["df", "-h"]},
                structured_schema="system.disk_free",
                structured_content={"filesystems": []},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert du_plan.missing_tool_names == frozenset({"tool.system.read.resources"})
    assert df_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_battery_request_requires_battery_hardware_observation() -> None:
    default_hardware_plan = live_state_evidence_plan(
        _request("what is current battery level?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["sysctl", "-n", "hw.logicalcpu"]},
                structured_content={"cores": 10},
            ),
        ),
    )
    battery_plan = live_state_evidence_plan(
        _request("what is current battery level?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["pmset", "-g", "batt"]},
                structured_schema="system.battery_charge",
                structured_content={"percentage": 77},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert default_hardware_plan.missing_tool_names == frozenset({"tool.system.read.hardware"})
    assert battery_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_hardware_request_requires_ram_observation() -> None:
    core_plan = live_state_evidence_plan(
        _request("how much RAM do I have?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["sysctl", "-n", "hw.logicalcpu"]},
                structured_content={"cores": 10},
            ),
        ),
    )
    memory_plan = live_state_evidence_plan(
        _request("how much RAM do I have?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["sysctl", "-n", "hw.memsize"]},
                structured_schema="system.memory_overview",
                structured_content={"bytes": 25_769_803_776},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert core_plan.missing_tool_names == frozenset({"tool.system.read.hardware"})
    assert memory_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_hardware_request_requires_cpu_brand_observation() -> None:
    memory_plan = live_state_evidence_plan(
        _request("what processor do I have?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["sysctl", "-n", "hw.memsize"]},
                structured_content={"bytes": 25_769_803_776},
            ),
        ),
    )
    core_schema_plan = live_state_evidence_plan(
        _request("what processor do I have?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["sysctl", "-n", "hw.logicalcpu"]},
                structured_schema="system.cpu_overview",
                structured_content={"logical_cores": 10, "source": "sysctl"},
                parse_status=ToolParseStatus.PARTIAL,
            ),
        ),
    )
    cpu_brand_plan = live_state_evidence_plan(
        _request("what processor do I have?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["sysctl", "-n", "machdep.cpu.brand_string"]},
                structured_schema="system.cpu_overview",
                structured_content={"brand": "Apple M4"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert memory_plan.missing_tool_names == frozenset({"tool.system.read.hardware"})
    assert core_schema_plan.missing_tool_names == frozenset({"tool.system.read.hardware"})
    assert cpu_brand_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_hardware_mixed_request_requires_every_subtype() -> None:
    battery_only_plan = live_state_evidence_plan(
        _request("How many CPU cores are there and what is current battery level?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["pmset", "-g", "batt"]},
                structured_schema="system.battery_charge",
                structured_content={"percent": 77},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    complete_plan = live_state_evidence_plan(
        _request("How many CPU cores are there and what is current battery level?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["pmset", "-g", "batt"]},
                structured_schema="system.battery_charge",
                structured_content={"percent": 77},
                parse_status=ToolParseStatus.PARSED,
            ),
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["sysctl", "-n", "hw.logicalcpu"]},
                structured_schema="system.cpu_overview",
                structured_content={"logical_cpus": 10},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert battery_only_plan.missing_tool_names == frozenset({"tool.system.read.hardware"})
    assert complete_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_os_request_requires_os_hardware_observation() -> None:
    default_hardware_plan = live_state_evidence_plan(
        _request("what macOS version is this machine running?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["sysctl", "-n", "hw.logicalcpu"]},
                structured_content={"cores": 10},
            ),
        ),
    )
    os_plan = live_state_evidence_plan(
        _request("what macOS version is this machine running?"),
        _plan(
            "tool.system.read.hardware",
            live_state_tool_names=("tool.system.read.hardware",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.hardware",
                arguments={"argv": ["sw_vers"]},
                structured_schema="system.os_version",
                structured_content={"product_name": "macOS"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert default_hardware_plan.missing_tool_names == frozenset({"tool.system.read.hardware"})
    assert os_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_vpn_request_requires_vpn_network_observation() -> None:
    netstat_plan = live_state_evidence_plan(
        _request("is VPN connected right now?"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["netstat", "-an"]},
                structured_content={"stdout": "tcp4 0 0 *.443 *.* LISTEN"},
            ),
        ),
    )
    unparsed_vpn_plan = live_state_evidence_plan(
        _request("is VPN connected right now?"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["scutil", "--nc", "list"]},
                structured_schema="system.vpn_status",
                structured_content=None,
                parse_status=ToolParseStatus.UNPARSED,
            ),
        ),
    )
    vpn_plan = live_state_evidence_plan(
        _request("is VPN connected right now?"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["scutil", "--nc", "list"]},
                structured_schema="system.vpn_status",
                structured_content={"connected": True},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert netstat_plan.missing_tool_names == frozenset({"tool.system.read.network"})
    assert unparsed_vpn_plan.missing_tool_names == frozenset({"tool.system.read.network"})
    assert vpn_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_ip_request_requires_ip_network_observation() -> None:
    netstat_plan = live_state_evidence_plan(
        _request("what is my current local IP address?"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["netstat", "-an"]},
                structured_content={"stdout": "tcp4 0 0 *.443 *.* LISTEN"},
            ),
        ),
    )
    local_ip_plan = live_state_evidence_plan(
        _request("what is my current local IP address?"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ifconfig"]},
                content=(
                    '{"exit_code": 0, '
                    '"stdout": "en0: inet 192.168.1.10 netmask 0xffffff00", '
                    '"stderr": ""}'
                ),
            ),
        ),
    )
    public_ip_plan = live_state_evidence_plan(
        _request("what is my public IP address?"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ifconfig"]},
                structured_content={"stdout": "en0: inet 192.168.1.10 netmask 0xffffff00"},
            ),
        ),
    )

    assert netstat_plan.missing_tool_names == frozenset({"tool.system.read.network"})
    assert local_ip_plan.missing_tool_names == frozenset()
    assert public_ip_plan.missing_tool_names == frozenset({"tool.system.read.network"})


def test_live_state_evidence_plan_internet_status_requires_connectivity_network_observation() -> None:
    netstat_plan = live_state_evidence_plan(
        _request("am I connected to the internet?"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["netstat", "-an"]},
                structured_content={"stdout": "tcp4 0 0 *.443 *.* LISTEN"},
            ),
        ),
    )
    interface_plan = live_state_evidence_plan(
        _request("am I connected to the internet?"),
        _plan(
            "tool.system.read.network",
            live_state_tool_names=("tool.system.read.network",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.network",
                arguments={"argv": ["ifconfig"]},
                content='{"exit_code": 0, "stdout": "en0: status: active", "stderr": ""}',
            ),
        ),
    )

    assert netstat_plan.missing_tool_names == frozenset({"tool.system.read.network"})
    assert interface_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_load_average_requires_load_average_observation() -> None:
    resource_overview_plan = live_state_evidence_plan(
        _request("what is load average?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                structured_schema="system.resource_overview",
                structured_content={
                    "cpu": {"used_percent": 20.0},
                    "memory": {"used_percent": 70.0},
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )
    uptime_plan = live_state_evidence_plan(
        _request("what is load average?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                arguments={"argv": ["uptime"]},
                content='{"exit_code": 0, "stdout": "load averages: 1.20 1.10 1.00", "stderr": ""}',
            ),
        ),
    )

    assert resource_overview_plan.missing_tool_names == frozenset(
        {"tool.system.read.resources"}
    )
    assert uptime_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_clears_parenthesized_calculator_expression() -> None:
    request = _request("is CPU load greater than 10*(e+1)")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            _completed_resource_ref(),
            _completed_ref("calculator.evaluate", arguments={"expression": "10*(e+1)"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_requires_every_calculator_expression_for_live_state_math() -> None:
    request = _request("is CPU greater than 10*e and memory greater than 2*pi?")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    partial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
                _completed_resource_ref(),
            _completed_ref("calculator.evaluate", arguments={"expression": "10*e"}),
        ),
    )
    complete_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
                _completed_resource_ref(),
            _completed_ref("calculator.evaluate", arguments={"expression": "10*e"}),
            _completed_ref("calculator.evaluate", arguments={"expression": "2*pi"}),
        ),
    )

    assert partial_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert partial_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert complete_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_requires_calculator_for_derived_countdown_value() -> None:
    request = _request(
        "посчитай с точностью до 4 знаков после запятой натуральный логарифм "
        "количества секунд, оставшихся до Нового года"
    )
    request_plan = _plan(
        "calculator.evaluate",
        "datetime.now",
        "datetime.until",
        live_state_tool_names=("datetime.now", "datetime.until"),
    )
    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    countdown_ref = _completed_ref(
        "datetime.until",
        structured_schema="datetime.until",
        structured_content={
            "from_iso": "2026-06-05T20:59:07+03:00",
            "target": "next_new_year",
            "unit": "seconds",
            "seconds": 18337521,
            "value": 18337521,
        },
        parse_status=ToolParseStatus.PARSED,
    )
    after_countdown_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(countdown_ref,),
    )
    wrong_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            countdown_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "ln(123)"}),
        ),
    )
    bare_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            countdown_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "18337521"}),
        ),
    )
    ungrounded_constant_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            countdown_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "ln(18337521 + 999)"}),
        ),
    )
    wrong_operation_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            countdown_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "sqrt(18337521)"}),
        ),
    )
    calculator_before_live_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            _completed_ref(
                "calculator.evaluate",
                arguments={"expression": "round(ln(18337521), 4)"},
            ),
            countdown_ref,
        ),
    )
    missing_round_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            countdown_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "ln(18337521)"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            countdown_ref,
            _completed_ref(
                "calculator.evaluate",
                arguments={"expression": "round(ln(18337521), 4)"},
            ),
        ),
    )

    assert initial_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert initial_plan.families == frozenset(
        {
            LiveStateEvidenceFamily.CURRENT_TIME,
            LiveStateEvidenceFamily.LIVE_STATE_MATH,
        }
    )
    assert initial_plan.candidate_tool_names == frozenset(
        {"datetime.now", "datetime.until", "calculator.evaluate"}
    )
    assert initial_plan.missing_tool_names == initial_plan.candidate_tool_names
    assert after_countdown_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert wrong_calculator_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert bare_calculator_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert ungrounded_constant_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert wrong_operation_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert calculator_before_live_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert missing_round_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_accepts_model_resolved_event_for_derived_elapsed_duration_value() -> None:
    request = _request(
        "корень кубический из количества минут, прошедших с дня благодарения последнего."
    )
    request_plan = _plan(
        "calculator.evaluate",
        "datetime.diff",
        "datetime.now",
        "datetime.until",
        live_state_tool_names=("datetime.diff", "datetime.now", "datetime.until"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T20:17:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    diff_ref = _completed_ref(
        "datetime.diff",
        structured_schema="datetime.diff",
        structured_content={
            "from_iso": "2025-11-27T00:00:00-05:00",
            "to_iso": "2026-06-07T20:17:00+03:00",
            "seconds": 16910220,
            "minutes": 281837,
            "hours": 4697.283333333334,
            "days": 195.7201388888889,
            "unit": "minutes",
            "value": 281837,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )

    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    after_now_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref,),
    )
    after_diff_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, diff_ref),
    )
    ungrounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            now_ref,
            diff_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "cbrt(282720)"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            now_ref,
            diff_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "cbrt(281837)"}),
        ),
    )

    assert initial_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert LiveStateEvidenceFamily.CURRENT_TIME in initial_plan.families
    assert initial_plan.candidate_tool_names == frozenset(
        {"datetime.now", "datetime.diff", "calculator.evaluate"}
    )
    assert initial_plan.missing_tool_names == frozenset(
        {"datetime.now", "datetime.diff", "calculator.evaluate"}
    )
    assert after_now_plan.missing_tool_names == frozenset(
        {"datetime.diff", "calculator.evaluate"}
    )
    assert after_diff_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert ungrounded_calculator_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


@pytest.mark.parametrize(
    ("prompt", "from_iso", "unit", "unit_value", "matching_expression", "wrong_base_expression"),
    [
        (
            "Десятичный логарифм количества часов, прошедших с 1 сентября прошлого года.",
            "2025-09-01T00:00:00+03:00",
            "hours",
            6716.283333333334,
            "log10(6716.283333333334)",
            "log2(6716.283333333334)",
        ),
        (
            "Десятичный логарифм количества часов, прошедших с "
            "2025-09-01T00:00:00+03:00.",
            "2025-09-01T00:00:00+03:00",
            "hours",
            6716.283333333334,
            "log10(6716.283333333334)",
            "log2(6716.283333333334)",
        ),
        (
            "двоичный логарифм количества часов, прошедших с 11 сентября прошлого года.",
            "2025-09-11T00:00:00+03:00",
            "hours",
            6476.283333333334,
            "log2(6476.283333333334)",
            "log10(6476.283333333334)",
        ),
        (
            "Натуральный логарифм количества миллисекунд, прошедших с 1 сентября прошлого года.",
            "2025-09-01T00:00:00+03:00",
            "milliseconds",
            24178620000.0,
            "ln(24178620000)",
            "log10(24178620000)",
        ),
        (
            "двоичный логарифм количества недель, прошедших с чемпионата мира по футболу в России 2018 года.",
            "2018-06-14T18:00:00+03:00",
            "weeks",
            416.4421626984127,
            "log2(416.4421626984127)",
            "log10(416.4421626984127)",
        ),
        (
            "Десятичный логарифм количества секунд, прошедших со дня рождения Билла Клинтона.",
            "1946-08-19T00:00:00-05:00",
            "seconds",
            2519999820.0,
            "log10(2519999820)",
            "log2(2519999820)",
        ),
        (
            "Натуральный логарифм количества дней, прошедших со дня смерти королевы Виктории.",
            "1901-01-22T00:00:00+00:00",
            "days",
            45794.84513888889,
            "ln(45794.84513888889)",
            "log10(45794.84513888889)",
        ),
        (
            "Двоичный логарифм количества недель, прошедших со дня отмены крепостного права в России.",
            "1861-03-03T00:00:00+03:00",
            "weeks",
            8626.120833333334,
            "log2(8626.120833333334)",
            "log10(8626.120833333334)",
        ),
    ],
)
def test_live_state_evidence_plan_requires_exact_log_base_for_elapsed_delta_value(
    prompt: str,
    from_iso: str,
    unit: str,
    unit_value: float,
    matching_expression: str,
    wrong_base_expression: str,
) -> None:
    request = _request(prompt)
    request_plan = _plan(
        "calculator.evaluate",
        "datetime.diff",
        "datetime.now",
        live_state_tool_names=("datetime.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T20:17:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    diff_ref = _completed_ref(
        "datetime.diff",
        structured_schema="datetime.diff",
        structured_content={
            "from_iso": from_iso,
            "to_iso": "2026-06-07T20:17:00+03:00",
            "seconds": 24178620,
            "milliseconds": 24178620000.0,
            "minutes": 402977.0,
            "hours": unit_value if unit == "hours" else 6716.283333333334,
            "days": 279.84513888888887,
            "weeks": unit_value if unit == "weeks" else 39.97787698412698,
            "unit": unit,
            "value": unit_value,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )

    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    after_now_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref,),
    )
    after_diff_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, diff_ref),
    )
    wrong_base_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            now_ref,
            diff_ref,
            _completed_ref(
                "calculator.evaluate",
                arguments={"expression": wrong_base_expression},
            ),
        ),
    )
    matching_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            now_ref,
            diff_ref,
            _completed_ref(
                "calculator.evaluate",
                arguments={"expression": matching_expression},
            ),
        ),
    )

    assert initial_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert initial_plan.candidate_tool_names == frozenset(
        {"datetime.now", "datetime.diff", "calculator.evaluate"}
    )
    assert initial_plan.missing_tool_names == frozenset(
        {"datetime.now", "datetime.diff", "calculator.evaluate"}
    )
    assert after_now_plan.missing_tool_names == frozenset(
        {"datetime.diff", "calculator.evaluate"}
    )
    assert after_diff_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert wrong_base_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert matching_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_datetime_diff_with_wrong_explicit_elapsed_endpoint() -> None:
    request = _request(
        "Десятичный логарифм количества часов, прошедших с "
        "2025-09-01T00:00:00+03:00."
    )
    request_plan = _plan(
        "calculator.evaluate",
        "datetime.diff",
        "datetime.now",
        live_state_tool_names=("datetime.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T20:17:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    wrong_diff_ref = _completed_ref(
        "datetime.diff",
        structured_schema="datetime.diff",
        structured_content={
            "from_iso": "2001-01-01T00:00:00+03:00",
            "to_iso": "2026-06-07T20:17:00+03:00",
            "hours": 222925.28333333333,
            "unit": "hours",
            "value": 222925.28333333333,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )
    matching_calculator_ref = _completed_ref(
        "calculator.evaluate",
        arguments={"expression": "log10(222925.28333333333)"},
    )

    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, wrong_diff_ref, matching_calculator_ref),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset(
        {"datetime.diff", "calculator.evaluate"}
    )


def test_live_state_evidence_plan_rejects_calendar_diff_for_datetime_diff_requirement() -> None:
    request = _request(
        "Десятичный логарифм количества часов, прошедших с "
        "2025-09-01T00:00:00+03:00."
    )
    request_plan = _plan(
        "calculator.evaluate",
        "datetime.diff",
        "datetime.now",
        live_state_tool_names=("datetime.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T20:17:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    calendar_diff_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={
            "from_iso": "2025-09-01T00:00:00+03:00",
            "to_iso": "2026-06-07T20:17:00+03:00",
            "hours": 6716.283333333334,
            "unit": "hours",
            "value": 6716.283333333334,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )
    calculator_ref = _completed_ref(
        "calculator.evaluate",
        arguments={"expression": "log10(6716.283333333334)"},
    )

    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, calendar_diff_ref, calculator_ref),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset({"datetime.diff"})


@pytest.mark.parametrize(
    ("prompt", "unit", "unit_value"),
    [
        ("Сколько с тех пор прошло минут?", "minutes", 32898294),
        ("сколько прошло с тех пор дней?", "days", 22846.0375),
    ],
)
def test_live_state_evidence_plan_requires_datetime_diff_for_anaphoric_elapsed_duration(
    prompt: str,
    unit: str,
    unit_value: float,
) -> None:
    request = _request(prompt)
    request_plan = _plan(
        "calendar.diff",
        "datetime.diff",
        "datetime.now",
        live_state_tool_names=("calendar.diff", "datetime.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T23:24:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    diff_ref = _completed_ref(
        "datetime.diff",
        structured_schema="datetime.diff",
        structured_content={
            "from_iso": "1963-11-22T12:30:00-06:00",
            "to_iso": "2026-06-07T23:24:00+03:00",
            unit: unit_value,
            "unit": unit,
            "value": unit_value,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )

    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    after_now_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref,),
    )
    after_diff_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, diff_ref),
    )

    assert initial_plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert initial_plan.candidate_tool_names == frozenset({"datetime.now", "datetime.diff"})
    assert initial_plan.missing_tool_names == frozenset({"datetime.now", "datetime.diff"})
    assert initial_plan.candidate_evidence_kinds == frozenset(
        {
            LiveStateEvidenceKind.CURRENT_TIMESTAMP,
            LiveStateEvidenceKind.FIXED_TIME_INTERVAL,
        }
    )
    assert initial_plan.missing_evidence_kinds == initial_plan.candidate_evidence_kinds
    assert after_now_plan.missing_tool_names == frozenset({"datetime.diff"})
    assert after_now_plan.missing_evidence_kinds == frozenset(
        {LiveStateEvidenceKind.FIXED_TIME_INTERVAL}
    )
    assert after_diff_plan.missing_tool_names == frozenset()
    assert after_diff_plan.missing_evidence_kinds == frozenset()
    assert "calculator.evaluate" not in initial_plan.candidate_tool_names


def test_live_state_evidence_plan_requires_datetime_diff_for_anaphoric_unit_correction() -> None:
    request = _request("Я же спросил про это количество дней, а не лет.")
    request_plan = _plan(
        "datetime.diff",
        "datetime.now",
        live_state_tool_names=("datetime.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-08T00:12:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )

    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    after_now_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref,),
    )

    assert initial_plan.candidate_tool_names == frozenset({"datetime.now", "datetime.diff"})
    assert initial_plan.missing_tool_names == frozenset({"datetime.now", "datetime.diff"})
    assert after_now_plan.missing_tool_names == frozenset({"datetime.diff"})


def test_live_state_evidence_plan_accepts_model_resolved_historical_endpoint_for_elapsed_duration() -> None:
    request = _request("Сколько дней прошло с Великой Октябрьской революции 1917 года?")
    request_plan = _plan(
        "datetime.diff",
        "datetime.now",
        live_state_tool_names=("datetime.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-08T00:12:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    diff_ref = _completed_ref(
        "datetime.diff",
        structured_schema="datetime.diff",
        structured_content={
            "from_iso": "1917-11-07T00:00:00+03:00",
            "to_iso": "2026-06-08T00:12:00+03:00",
            "days": 39659.00833333333,
            "unit": "days",
            "value": 39659.00833333333,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )

    after_diff_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, diff_ref),
    )

    assert after_diff_plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert after_diff_plan.missing_tool_names == frozenset()
    assert after_diff_plan.missing_evidence_kinds == frozenset()


def test_tool_proposal_contract_prefers_fixed_interval_tool_for_fixed_time_delta_evidence() -> None:
    request = _request(
        "Какова длительность промежутка времени с того момента до текущего в минутах?"
    )
    request_plan = _plan(
        "calendar.diff",
        "datetime.diff",
        "datetime.now",
        live_state_tool_names=("calendar.diff", "datetime.diff", "datetime.now"),
    )
    evidence_plan = final_answer_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )

    assert evidence_plan is not None
    contract = tool_proposal_output_contract(
        request_plan,
        completed_observations=0,
        missing_evidence_plan=evidence_plan,
    )

    assert "missing evidence kinds: current_timestamp, fixed_time_interval" in contract
    assert "candidate evidence tools: datetime.diff, datetime.now" in contract
    assert "use datetime.now and then datetime.diff" in contract
    assert "use datetime.now and then calendar.diff" not in contract


def test_tool_proposal_contract_uses_datetime_diff_after_current_timestamp_evidence() -> None:
    request = _request("Сколько прошло дней с момента Карибского кризиса?")
    request_plan = _plan(
        "calendar.diff",
        "datetime.diff",
        "datetime.now",
        live_state_tool_names=("calendar.diff", "datetime.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-08T00:12:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    evidence_plan = final_answer_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref,),
    )

    assert evidence_plan is not None
    contract = tool_proposal_output_contract(
        request_plan,
        completed_observations=1,
        missing_evidence_plan=evidence_plan,
    )

    assert "missing evidence kinds: fixed_time_interval" in contract
    assert "candidate evidence tools: datetime.diff, datetime.now" in contract
    assert 'Return exactly {"action":"tool_call","tool_name":"datetime.diff"' in contract
    assert "Do not call datetime.now again" in contract
    assert "use datetime.diff with explicit timezone-aware ISO timestamp arguments" in contract
    assert "named or historical event" in contract
    assert "Self-contained calendar, duration, or arithmetic questions" not in contract
    assert "use datetime.now and then datetime.diff" not in contract
    assert "use datetime.now and then calendar.diff" not in contract


def test_tool_proposal_contract_uses_explicit_endpoints_for_self_contained_calendar_interval() -> None:
    request = _request(
        "Количество месяцев между 2025-11-27T00:00:00+00:00 "
        "и 2026-04-05T00:00:00+00:00."
    )
    request_plan = _plan(
        "calendar.diff",
        "datetime.now",
        live_state_tool_names=("calendar.diff", "datetime.now"),
    )
    evidence_plan = final_answer_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )

    assert evidence_plan is not None
    assert evidence_plan.candidate_tool_names == frozenset({"calendar.diff"})
    assert evidence_plan.missing_tool_names == frozenset({"calendar.diff"})
    assert evidence_plan.missing_evidence_kinds == frozenset(
        {LiveStateEvidenceKind.CALENDAR_INTERVAL}
    )
    contract = tool_proposal_output_contract(
        request_plan,
        completed_observations=0,
        missing_evidence_plan=evidence_plan,
    )

    assert 'Return exactly {"action":"tool_call","tool_name":"calendar.diff"' in contract
    assert "Use the explicit ISO endpoints from the user request" in contract
    assert "completed datetime.now iso" not in contract
    assert "Do not call datetime.now again" not in contract


@pytest.mark.parametrize(
    ("prompt", "unit", "unit_value", "matching_payload"),
    [
        (
            "Количество микросекунд между последним днём Благодарения и Пасхой.",
            "microseconds",
            11145600000000,
            {
                "from_iso": "2025-11-27T00:00:00+00:00",
                "to_iso": "2026-04-05T00:00:00+00:00",
                "microseconds": 11145600000000,
                "milliseconds": 11145600000.0,
                "seconds": 11145600,
                "minutes": 185760.0,
                "hours": 3096.0,
                "days": 129.0,
                "weeks": 18.428571428571427,
                "months": 4,
                "quarters": 1,
                "decades": 0,
                "unit": "microseconds",
                "value": 11145600000000,
                "absolute": False,
            },
        ),
        (
            "Количество месяцев между последним днём Благодарения и Пасхой.",
            "months",
            4,
            {
                "from_iso": "2025-11-27T00:00:00+00:00",
                "to_iso": "2026-04-05T00:00:00+00:00",
                "microseconds": 11145600000000,
                "milliseconds": 11145600000.0,
                "seconds": 11145600,
                "minutes": 185760.0,
                "hours": 3096.0,
                "days": 129.0,
                "weeks": 18.428571428571427,
                "months": 4,
                "quarters": 1,
                "decades": 0,
                "unit": "months",
                "value": 4,
                "absolute": False,
            },
        ),
        (
            "Количество недель между последним днём Благодарения и Пасхой.",
            "weeks",
            18.428571428571427,
            {
                "from_iso": "2025-11-27T00:00:00+00:00",
                "to_iso": "2026-04-05T00:00:00+00:00",
                "microseconds": 11145600000000,
                "milliseconds": 11145600000.0,
                "seconds": 11145600,
                "minutes": 185760.0,
                "hours": 3096.0,
                "days": 129.0,
                "weeks": 18.428571428571427,
                "months": 4,
                "quarters": 1,
                "decades": 0,
                "unit": "weeks",
                "value": 18.428571428571427,
                "absolute": False,
            },
        ),
        (
            "Количество дней между последним днём Благодарения и Пасхой.",
            "days",
            129.0,
            {
                "from_iso": "2025-11-27T00:00:00+00:00",
                "to_iso": "2026-04-05T00:00:00+00:00",
                "microseconds": 11145600000000,
                "milliseconds": 11145600000.0,
                "seconds": 11145600,
                "minutes": 185760.0,
                "hours": 3096.0,
                "days": 129.0,
                "weeks": 18.428571428571427,
                "months": 4,
                "quarters": 1,
                "decades": 0,
                "unit": "days",
                "value": 129.0,
                "absolute": False,
            },
        ),
        (
            "Количество часов между последним днём Благодарения и Пасхой.",
            "hours",
            3096.0,
            {
                "from_iso": "2025-11-27T00:00:00+00:00",
                "to_iso": "2026-04-05T00:00:00+00:00",
                "microseconds": 11145600000000,
                "milliseconds": 11145600000.0,
                "seconds": 11145600,
                "minutes": 185760.0,
                "hours": 3096.0,
                "days": 129.0,
                "weeks": 18.428571428571427,
                "months": 4,
                "quarters": 1,
                "decades": 0,
                "unit": "hours",
                "value": 3096.0,
                "absolute": False,
            },
        ),
        (
            "Количество минут между последним днём Благодарения и Пасхой.",
            "minutes",
            185760.0,
            {
                "from_iso": "2025-11-27T00:00:00+00:00",
                "to_iso": "2026-04-05T00:00:00+00:00",
                "microseconds": 11145600000000,
                "milliseconds": 11145600000.0,
                "seconds": 11145600,
                "minutes": 185760.0,
                "hours": 3096.0,
                "days": 129.0,
                "weeks": 18.428571428571427,
                "months": 4,
                "quarters": 1,
                "decades": 0,
                "unit": "minutes",
                "value": 185760.0,
                "absolute": False,
            },
        ),
        (
            "Количество кварталов между последним днём Благодарения и Пасхой.",
            "quarters",
            1,
            {
                "from_iso": "2025-11-27T00:00:00+00:00",
                "to_iso": "2026-04-05T00:00:00+00:00",
                "microseconds": 11145600000000,
                "milliseconds": 11145600000.0,
                "seconds": 11145600,
                "minutes": 185760.0,
                "hours": 3096.0,
                "days": 129.0,
                "weeks": 18.428571428571427,
                "months": 4,
                "quarters": 1,
                "decades": 0,
                "unit": "quarters",
                "value": 1,
                "absolute": False,
            },
        ),
        (
            "Количество декад между последним днём Благодарения и Пасхой.",
            "decades",
            0,
            {
                "from_iso": "2025-11-27T00:00:00+00:00",
                "to_iso": "2026-04-05T00:00:00+00:00",
                "microseconds": 11145600000000,
                "milliseconds": 11145600000.0,
                "seconds": 11145600,
                "minutes": 185760.0,
                "hours": 3096.0,
                "days": 129.0,
                "weeks": 18.428571428571427,
                "months": 4,
                "quarters": 1,
                "decades": 0,
                "unit": "decades",
                "value": 0,
                "absolute": False,
            },
        ),
    ],
)
def test_live_state_evidence_plan_rejects_calendar_diff_for_unresolved_relative_calendar_interval(
    prompt: str,
    unit: str,
    unit_value: int | float,
    matching_payload: dict,
) -> None:
    request = _request(prompt)
    request_plan = _plan(
        "calendar.diff",
        "datetime.now",
        live_state_tool_names=("calendar.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T20:17:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    wrong_diff_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={**matching_payload, "unit": "seconds", "value": 11145600},
        parse_status=ToolParseStatus.PARSED,
    )
    matching_diff_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={
            **matching_payload,
            "source_iso": "2026-06-07T20:17:00+03:00",
        },
        parse_status=ToolParseStatus.PARSED,
    )

    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    after_now_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref,),
    )
    wrong_unit_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, wrong_diff_ref),
    )
    matching_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, matching_diff_ref),
    )

    assert unit_value == matching_payload["value"]
    assert unit == matching_payload["unit"]
    assert initial_plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert initial_plan.candidate_tool_names == frozenset(
        {"calendar.diff", "datetime.now"}
    )
    assert initial_plan.missing_tool_names == frozenset(
        {"calendar.diff", "datetime.now"}
    )
    assert after_now_plan.missing_tool_names == frozenset({"calendar.diff"})
    assert wrong_unit_plan.missing_tool_names == frozenset({"calendar.diff"})
    assert matching_plan.missing_tool_names == frozenset({"calendar.diff"})


def test_live_state_evidence_plan_does_not_require_now_for_explicit_calendar_interval_endpoints() -> None:
    request = _request(
        "Количество микросекунд между последним днём Благодарения "
        "(2025-11-27T00:00:00+00:00) и Пасхой "
        "(2026-04-05T00:00:00+00:00)."
    )
    request_plan = _plan(
        "calendar.diff",
        "datetime.now",
        live_state_tool_names=("calendar.diff", "datetime.now"),
    )
    diff_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={
            "from_iso": "2025-11-27T00:00:00+00:00",
            "to_iso": "2026-04-05T00:00:00+00:00",
            "unit": "microseconds",
            "value": 11145600000000,
        },
        parse_status=ToolParseStatus.PARSED,
    )

    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    after_diff_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(diff_ref,),
    )

    assert initial_plan.family is LiveStateEvidenceFamily.CALENDAR_INTERVAL
    assert initial_plan.evidence_required is True
    assert initial_plan.candidate_tool_names == frozenset({"calendar.diff"})
    assert initial_plan.missing_tool_names == frozenset({"calendar.diff"})
    assert after_diff_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_calendar_diff_with_wrong_explicit_interval_endpoints() -> None:
    request = _request(
        "Количество микросекунд между последним днём Благодарения "
        "(2025-11-27T00:00:00+00:00) и Пасхой "
        "(2026-04-05T00:00:00+00:00)."
    )
    request_plan = _plan(
        "calendar.diff",
        "datetime.now",
        live_state_tool_names=("calendar.diff", "datetime.now"),
    )
    wrong_diff_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={
            "from_iso": "2001-01-01T00:00:00+00:00",
            "to_iso": "2026-04-05T00:00:00+00:00",
            "unit": "microseconds",
            "value": 796953600000000,
        },
        parse_status=ToolParseStatus.PARSED,
    )

    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(wrong_diff_ref,),
    )

    assert plan.family is LiveStateEvidenceFamily.CALENDAR_INTERVAL
    assert plan.candidate_tool_names == frozenset({"calendar.diff"})
    assert plan.missing_tool_names == frozenset({"calendar.diff"})


def test_live_state_evidence_plan_rejects_bare_date_calendar_interval_endpoints() -> None:
    request = _request("Количество дней между 2025-11-27 и 2026-04-05.")
    request_plan = _plan(
        "calendar.diff",
        live_state_tool_names=("calendar.diff",),
    )
    diff_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={
            "from_iso": "2025-11-27T00:00:00+00:00",
            "to_iso": "2026-04-05T00:00:00+00:00",
            "unit": "days",
            "value": 129.0,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )

    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(diff_ref,),
    )

    assert plan.family is LiveStateEvidenceFamily.CALENDAR_INTERVAL
    assert plan.candidate_tool_names == frozenset({"calendar.diff"})
    assert plan.missing_tool_names == frozenset({"calendar.diff"})


def test_live_state_evidence_plan_rejects_model_resolved_relative_event_calendar_interval() -> None:
    request = _request("Количество микросекунд между последним днём Благодарения и Пасхой.")
    request_plan = _plan(
        "calendar.diff",
        "datetime.now",
        live_state_tool_names=("calendar.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T20:17:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    diff_without_source_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={
            "from_iso": "2025-11-27T00:00:00+00:00",
            "to_iso": "2026-04-05T00:00:00+00:00",
            "unit": "microseconds",
            "value": 11145600000000,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )
    diff_wrong_source_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={
            "from_iso": "2025-11-27T00:00:00+00:00",
            "to_iso": "2026-04-05T00:00:00+00:00",
            "source_iso": "2000-01-01T00:00:00+00:00",
            "unit": "microseconds",
            "value": 11145600000000,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )
    diff_matching_source_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={
            "from_iso": "2025-11-27T00:00:00+00:00",
            "to_iso": "2026-04-05T00:00:00+00:00",
            "source_iso": "2026-06-07T20:17:00+03:00",
            "unit": "microseconds",
            "value": 11145600000000,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )

    after_now_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref,),
    )
    without_source_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, diff_without_source_ref),
    )
    wrong_source_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, diff_wrong_source_ref),
    )
    matching_source_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, diff_matching_source_ref),
    )

    assert after_now_plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert after_now_plan.candidate_tool_names == frozenset({"calendar.diff", "datetime.now"})
    assert after_now_plan.missing_tool_names == frozenset({"calendar.diff"})
    assert without_source_plan.missing_tool_names == frozenset({"calendar.diff"})
    assert wrong_source_plan.missing_tool_names == frozenset({"calendar.diff"})
    assert matching_source_plan.missing_tool_names == frozenset({"calendar.diff"})


def test_live_state_evidence_plan_requires_calculator_for_derived_calendar_interval() -> None:
    request = _request(
        "Десятичный логарифм количества месяцев между последним днём Благодарения и Пасхой."
    )
    request_plan = _plan(
        "calendar.diff",
        "calculator.evaluate",
        "datetime.now",
        live_state_tool_names=("calendar.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T20:17:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    diff_ref = _completed_ref(
        "calendar.diff",
        structured_schema="calendar.diff",
        structured_content={
            "from_iso": "2025-11-27T00:00:00+00:00",
            "to_iso": "2026-04-05T00:00:00+00:00",
            "source_iso": "2026-06-07T20:17:00+03:00",
            "months": 4,
            "unit": "months",
            "value": 4,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )

    after_diff_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, diff_ref),
    )
    wrong_operation_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            now_ref,
            diff_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "sqrt(4)"}),
        ),
    )
    matching_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            now_ref,
            diff_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "log10(4)"}),
        ),
    )

    assert after_diff_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert "calendar.diff" in after_diff_plan.missing_tool_names
    assert "calendar.diff" in wrong_operation_plan.missing_tool_names
    assert "calendar.diff" in matching_plan.missing_tool_names


def test_live_state_evidence_plan_does_not_clear_invented_elapsed_time_unit() -> None:
    request = _request(
        "логарифм количества выдуманных единиц времени, прошедших с 1 сентября прошлого года."
    )
    request_plan = _plan(
        "calculator.evaluate",
        "datetime.diff",
        "datetime.now",
        live_state_tool_names=("datetime.diff", "datetime.now"),
    )
    now_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-07T20:17:00+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    seconds_diff_ref = _completed_ref(
        "datetime.diff",
        structured_schema="datetime.diff",
        structured_content={
            "from_iso": "2025-09-01T00:00:00+03:00",
            "to_iso": "2026-06-07T20:17:00+03:00",
            "seconds": 24178620,
            "unit": "seconds",
            "value": 24178620,
            "absolute": False,
        },
        parse_status=ToolParseStatus.PARSED,
    )
    calculator_ref = _completed_ref(
        "calculator.evaluate",
        arguments={"expression": "log10(24178620)"},
    )

    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    after_seconds_diff_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(now_ref, seconds_diff_ref, calculator_ref),
    )

    assert initial_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert initial_plan.candidate_tool_names == frozenset(
        {"datetime.now", "calculator.evaluate"}
    )
    assert after_seconds_diff_plan.missing_tool_names == frozenset(
        {"calculator.evaluate"}
    )


def test_live_state_evidence_plan_rejects_schema_less_datetime_until_for_derived_countdown_value() -> None:
    request = _request(
        "посчитай с точностью до 4 знаков после запятой натуральный логарифм "
        "количества секунд, оставшихся до Нового года"
    )
    request_plan = _plan(
        "calculator.evaluate",
        "datetime.now",
        "datetime.until",
        live_state_tool_names=("datetime.now", "datetime.until"),
    )
    datetime_ref = _completed_ref(
        "datetime.now",
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-05T20:59:07+03:00"},
        parse_status=ToolParseStatus.PARSED,
    )
    schema_less_countdown_ref = _completed_ref(
        "datetime.until",
        structured_content={
            "from_iso": "2026-06-05T20:59:07+03:00",
            "target": "next_new_year",
            "unit": "seconds",
            "seconds": 18337521,
            "value": 18337521,
        },
        parse_status=ToolParseStatus.PARSED,
    )
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            datetime_ref,
            schema_less_countdown_ref,
            _completed_ref(
                "calculator.evaluate",
                arguments={"expression": "round(ln(18337521), 4)"},
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset(
        {"datetime.until", "calculator.evaluate"}
    )


def test_live_state_evidence_plan_requires_calculator_for_prose_arithmetic_transform() -> None:
    request = _request("current CPU usage plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        arguments={"metric": "cpu_and_memory"},
        structured_schema="system.resource_overview",
        structured_content={
            "cpu": {"used_percent": 10.2},
            "memory": {"used_percent": 62.5},
        },
        parse_status=ToolParseStatus.PARSED,
    )
    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    after_resource_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(resource_ref,),
    )
    wrong_operation_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 - 10"}),
        ),
    )
    wrong_live_field_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "62.5 + 10"}),
        ),
    )
    extra_operation_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10 * 10"}),
        ),
    )
    compensating_operation_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10 - 10"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10"}),
        ),
    )

    assert initial_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert initial_plan.candidate_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )
    assert initial_plan.missing_tool_names == initial_plan.candidate_tool_names
    assert after_resource_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert wrong_operation_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert wrong_live_field_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert extra_operation_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert compensating_operation_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_explicit_arithmetic_that_omits_live_value() -> None:
    request = _request("calculate current CPU usage plus 2+2")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.resource_overview",
        structured_content={"cpu": {"used_percent": 10.2}},
        parse_status=ToolParseStatus.PARSED,
    )
    self_contained_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "2+2"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2+2+2"}),
        ),
    )

    assert self_contained_calculator_plan.missing_tool_names == frozenset(
        {"calculator.evaluate"}
    )
    assert grounded_calculator_plan.missing_tool_names == frozenset()


@pytest.mark.parametrize(
    "user_input",
    [
        "is the sum of current CPU usage and 10 greater than 20",
        "is 10 + current CPU usage > 20",
    ],
)
def test_live_state_evidence_plan_requires_calculator_for_transformed_threshold(
    user_input: str,
) -> None:
    request = _request(user_input)
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.resource_overview",
        structured_content={"cpu": {"used_percent": 10.2}},
        parse_status=ToolParseStatus.PARSED,
    )
    after_resource_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(resource_ref,),
    )
    wrong_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10 > 20"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10"}),
        ),
    )

    assert after_resource_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert after_resource_plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            LiveStateEvidenceFamily.LIVE_STATE_MATH,
        }
    )
    assert after_resource_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert wrong_calculator_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_requires_calculator_for_rhs_transformed_threshold() -> None:
    request = _request("is 20 less than current CPU usage plus 10?")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.resource_overview",
        structured_content={"cpu": {"used_percent": 10.2}},
        parse_status=ToolParseStatus.PARSED,
    )
    after_resource_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(resource_ref,),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10"}),
        ),
    )

    assert after_resource_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert after_resource_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_threshold_constant_as_transform_operand() -> None:
    request = _request("is current CPU usage plus 10 greater than 20?")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.resource_overview",
        structured_content={"cpu": {"used_percent": 10.2}},
        parse_status=ToolParseStatus.PARSED,
    )
    wrong_threshold_operand_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 20"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10"}),
        ),
    )

    assert wrong_threshold_operand_plan.missing_tool_names == frozenset(
        {"calculator.evaluate"}
    )
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_process_snapshot_for_global_ru_processor_transform() -> None:
    request = _request("нагрузка процессора плюс 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        "tool.system.read.process",
        live_state_tool_names=("tool.system.read.resources", "tool.system.read.process"),
    )
    process_ref = _completed_ref(
        "tool.system.read.process",
        structured_schema="system.process_resource_snapshot",
        structured_content={"processes": [{"name": "Chrome", "cpu_percent": 4.2}]},
        parse_status=ToolParseStatus.PARSED,
    )
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            process_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "4.2 + 10"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            LiveStateEvidenceFamily.LIVE_STATE_MATH,
        }
    )
    assert plan.missing_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )


def test_live_state_evidence_plan_rejects_future_system_tool_numeric_provenance() -> None:
    request = _request("current CPU usage plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        "tool.system.read.future",
        live_state_tool_names=("tool.system.read.resources", "tool.system.read.future"),
    )
    future_ref = _completed_ref(
        "tool.system.read.future",
        structured_schema="system.cpu_overview",
        structured_content={"used_percent": 10.2},
        parse_status=ToolParseStatus.PARSED,
    )
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            future_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10"}),
        ),
    )

    assert plan.missing_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )


def test_live_state_evidence_plan_requires_each_requested_live_numeric_operand() -> None:
    request = _request("add current CPU usage and current memory usage")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        arguments={"metric": "cpu_and_memory"},
        structured_schema="system.resource_overview",
        structured_content={
            "cpu": {"used_percent": 10.2},
            "memory": {"used_percent": 62.5},
        },
        parse_status=ToolParseStatus.PARSED,
    )
    duplicated_cpu_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10.2"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 62.5"}),
        ),
    )

    assert duplicated_cpu_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_wrong_same_scope_live_numeric_field() -> None:
    request = _request("current CPU usage plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        arguments={"argv": ["top", "-l", "1", "-n", "0"]},
        structured_schema="system.cpu_overview",
        structured_content={
            "idle_percent": 89.8,
            "used_percent": 10.2,
            "user_percent": 6.1,
        },
        parse_status=ToolParseStatus.PARSED,
    )
    wrong_field_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "89.8 + 10"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10"}),
        ),
    )

    assert wrong_field_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_accepts_cpu_load_transform_from_usage_payload() -> None:
    request = _request("current CPU load plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.cpu_overview",
        structured_content={"used_percent": 10.2},
        parse_status=ToolParseStatus.PARSED,
    )
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10"}),
        ),
    )

    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_wrong_memory_numeric_field() -> None:
    request = _request("current memory usage plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.memory_overview",
        structured_content={
            "available_percent": 37.5,
            "used_percent": 62.5,
        },
        parse_status=ToolParseStatus.PARSED,
    )
    wrong_field_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "37.5 + 10"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "62.5 + 10"}),
        ),
    )

    assert wrong_field_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_unparsed_live_numeric_payload_for_calculator() -> None:
    request = _request("current memory usage plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.memory_overview",
        structured_content={"used_percent": 62.5},
        parse_status=ToolParseStatus.UNPARSED,
    )
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "62.5 + 10"}),
        ),
    )

    assert plan.missing_tool_names == frozenset({"tool.system.read.resources", "calculator.evaluate"})


def test_live_state_evidence_plan_rejects_wrong_disk_numeric_field() -> None:
    request = _request("available disk percent plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.disk_free",
        structured_content={
            "filesystems": [
                {"mount": "/", "available_percent": 88.0, "used_percent_value": 12.0}
            ],
        },
        parse_status=ToolParseStatus.PARSED,
    )
    wrong_field_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "12 + 10"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "88 + 10"}),
        ),
    )

    assert wrong_field_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_process_memory_for_process_cpu_transform() -> None:
    request = _request("Chrome CPU usage plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.process",
        live_state_tool_names=("tool.system.read.process",),
    )
    process_ref = _completed_ref(
        "tool.system.read.process",
        structured_schema="system.process_resource_snapshot",
        structured_content={
            "processes": [
                {"name": "Chrome", "cpu_percent": 4.2, "memory_percent": 11.7}
            ],
        },
        parse_status=ToolParseStatus.PARSED,
    )
    wrong_field_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            process_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "11.7 + 10"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            process_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "4.2 + 10"}),
        ),
    )

    assert wrong_field_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_rejects_other_process_numeric_for_process_transform() -> None:
    request = _request("Chrome CPU usage plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.process",
        live_state_tool_names=("tool.system.read.process",),
    )
    process_ref = _completed_ref(
        "tool.system.read.process",
        structured_schema="system.process_resource_snapshot",
        structured_content={
            "processes": [
                {"name": "Chrome", "cpu_percent": 4.2},
                {"name": "Safari", "cpu_percent": 40.0},
            ],
        },
        parse_status=ToolParseStatus.PARSED,
    )
    wrong_process_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            process_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "40 + 10"}),
        ),
    )
    grounded_calculator_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            process_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "4.2 + 10"}),
        ),
    )

    assert wrong_process_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert grounded_calculator_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_accepts_process_cpu_load_transform_from_cpu_percent() -> None:
    request = _request("Chrome CPU load plus 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.process",
        live_state_tool_names=("tool.system.read.process",),
    )
    process_ref = _completed_ref(
        "tool.system.read.process",
        structured_schema="system.process_resource_snapshot",
        structured_content={"processes": [{"name": "Chrome", "cpu_percent": 4.2}]},
        parse_status=ToolParseStatus.PARSED,
    )
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            process_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "4.2 + 10"}),
        ),
    )

    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_detects_symbolic_live_numeric_transform() -> None:
    request = _request("current CPU usage + 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )

    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            LiveStateEvidenceFamily.LIVE_STATE_MATH,
        }
    )
    assert plan.missing_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )


def test_live_state_evidence_plan_accepts_symbolic_live_numeric_transform_calculator() -> None:
    request = _request("current CPU usage + 10")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        structured_schema="system.resource_overview",
        structured_content={"cpu": {"used_percent": 10.2}},
        parse_status=ToolParseStatus.PARSED,
    )
    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "10.2 + 10"}),
        ),
    )

    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_accepts_average_live_numeric_transform() -> None:
    request = _request("average current CPU usage and current memory usage")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )
    resource_ref = _completed_ref(
        "tool.system.read.resources",
        arguments={"metric": "cpu_and_memory"},
        structured_schema="system.resource_overview",
        structured_content={
            "cpu": {"used_percent": 20.0},
            "memory": {"used_percent": 60.0},
        },
        parse_status=ToolParseStatus.PARSED,
    )
    initial_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    after_resource_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(resource_ref,),
    )
    valid_average_plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(
            resource_ref,
            _completed_ref("calculator.evaluate", arguments={"expression": "(20 + 60) / 2"}),
        ),
    )

    assert initial_plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert initial_plan.missing_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )
    assert after_resource_plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert valid_average_plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_keeps_aggregate_memory_transform_out_of_process_scope() -> None:
    request = _request("add 10 to current memory percent")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )

    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.families == frozenset(
        {
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
            LiveStateEvidenceFamily.LIVE_STATE_MATH,
        }
    )
    assert LiveStateEvidenceFamily.SYSTEM_PROCESS not in plan.families
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_does_not_treat_logarithm_as_logs_history() -> None:
    plan = live_state_evidence_plan(
        _request("calculate the natural logarithm of the number of seconds until New Year"),
        _plan(
            "calculator.evaluate",
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.candidate_tool_names == frozenset(
        {"datetime.now", "datetime.until", "calculator.evaluate"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_reports_unavailable_when_relevant_tool_is_not_allowed() -> None:
    request = _request("what is current CPU usage?")
    request_plan = _plan("datetime.now", live_state_tool_names=("datetime.now",))

    plan = live_state_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )
    final_answer_plan = final_answer_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset()
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"
    assert final_answer_plan is not None
    assert final_answer_plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_math_plan_does_not_report_calculator_only_candidate_when_live_tool_unavailable() -> None:
    plan = live_state_evidence_plan(
        _request("is CPU load greater than 10*e"),
        _plan("calculator.evaluate"),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset()
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_math_plan_stays_blocked_when_calculator_is_unavailable_after_live_observation() -> None:
    plan = live_state_evidence_plan(
        _request("is CPU load greater than 10*e"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(_completed_resource_ref(),),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"calculator.evaluate"})
    assert plan.unavailable_reason == "live_state_tool_unavailable"


def test_live_state_evidence_plan_rejects_untyped_resource_observation_for_cpu() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(_completed_ref("tool.system.read.resources"),),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


def test_live_state_evidence_plan_rejects_schema_less_resource_payload_for_cpu() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                structured_content={"cpu": {"used_percent": 10.2}},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


def test_live_state_evidence_plan_rejects_content_embedded_resource_schema_for_cpu() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                structured_content={
                    "schema": "system.resource_overview",
                    "cpu": {"used_percent": 10.2},
                },
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


def test_live_state_evidence_plan_rejects_raw_json_resource_shape_for_cpu() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref(
                "tool.system.read.resources",
                content='{"cpu": {"used_percent": 10.2}, "source": "fake"}',
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset({"tool.system.read.resources"})


def test_live_state_evidence_plan_rejects_untyped_daemon_status_observation() -> None:
    plan = live_state_evidence_plan(
        _request("what is the daemon status?"),
        _plan("daemon.status", live_state_tool_names=("daemon.status",)),
        tool_observation_refs=(_completed_ref("daemon.status"),),
    )

    assert plan.family is LiveStateEvidenceFamily.DAEMON_STATUS
    assert plan.candidate_tool_names == frozenset({"daemon.status"})
    assert plan.missing_tool_names == frozenset({"daemon.status"})


def test_live_state_evidence_plan_rejects_schema_less_daemon_status_payload() -> None:
    plan = live_state_evidence_plan(
        _request("what is the daemon status?"),
        _plan("daemon.status", live_state_tool_names=("daemon.status",)),
        tool_observation_refs=(
            _completed_ref(
                "daemon.status",
                structured_content={"status": "ok"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.DAEMON_STATUS
    assert plan.candidate_tool_names == frozenset({"daemon.status"})
    assert plan.missing_tool_names == frozenset({"daemon.status"})


def test_live_state_evidence_plan_rejects_content_embedded_daemon_status_schema() -> None:
    plan = live_state_evidence_plan(
        _request("what is the daemon status?"),
        _plan("daemon.status", live_state_tool_names=("daemon.status",)),
        tool_observation_refs=(
            _completed_ref(
                "daemon.status",
                structured_content={"schema": "daemon.status", "status": "ok"},
                parse_status=ToolParseStatus.PARSED,
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.DAEMON_STATUS
    assert plan.candidate_tool_names == frozenset({"daemon.status"})
    assert plan.missing_tool_names == frozenset({"daemon.status"})


def test_failed_live_state_observation_exhausts_single_subtype_evidence() -> None:
    request = _request("what is current battery level?")
    request_plan = _plan(
        "tool.system.read.hardware",
        live_state_tool_names=("tool.system.read.hardware",),
    )
    failed_ref = _failed_ref("tool.system.read.hardware")

    assert failed_observation_exhausts_missing_evidence(
        request,
        request_plan,
        failed_ref,
        (failed_ref,),
    )


def test_failed_live_state_observation_does_not_exhaust_same_family_multi_subtype_evidence() -> None:
    request = _request("How many CPU cores are there and what is current battery level?")
    request_plan = _plan(
        "tool.system.read.hardware",
        live_state_tool_names=("tool.system.read.hardware",),
    )
    failed_ref = _failed_ref("tool.system.read.hardware")

    assert (
        failed_observation_exhausts_missing_evidence(
            request,
            request_plan,
            failed_ref,
            (failed_ref,),
        )
        is False
    )


def test_terminal_unavailable_observation_does_not_clear_same_family_multi_subtype_missing_evidence() -> None:
    request = _request("How many CPU cores are there and what is current battery level?")
    request_plan = _plan(
        "tool.system.read.hardware",
        live_state_tool_names=("tool.system.read.hardware",),
    )
    failed_ref = _failed_ref("tool.system.read.hardware")

    plan = final_answer_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=(failed_ref,),
    )

    assert plan is not None
    assert plan.missing_tool_names == frozenset({"tool.system.read.hardware"})


def test_request_requires_initial_tool_evidence_for_live_state_with_allowed_candidate() -> None:
    assert request_requires_initial_tool_evidence(
        _request("what is current CPU usage?"),
        _plan(
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
    )


def test_request_requires_initial_tool_evidence_ignores_live_state_near_miss() -> None:
    assert (
        request_requires_initial_tool_evidence(
            _request("what does CPU usage mean?"),
            _plan(
                "tool.system.read.resources",
                live_state_tool_names=("tool.system.read.resources",),
            ),
        )
        is False
    )


def test_legacy_live_state_intent_wrapper_keeps_existing_math_guard_behavior() -> None:
    request = _request("is CPU load greater than 10*e")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )

    assert contains_live_state_intent(request.user_input) is True
    assert (
        request_needs_live_state_math_evidence(
            request,
            request_plan,
            tool_observation_refs=(),
        )
        is True
    )


def test_live_state_tool_name_helper_uses_explicit_allowlist() -> None:
    assert is_live_state_tool_name("tool.system.read.resources") is True
    assert is_live_state_tool_name("tool.system.read.future") is False


def test_legacy_live_state_math_guard_uses_typed_near_miss_detection() -> None:
    request = _request("what does CPU usage above 10*e mean?")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        live_state_tool_names=("tool.system.read.resources",),
    )

    assert (
        request_needs_live_state_math_evidence(
            request,
            request_plan,
            tool_observation_refs=(),
        )
        is False
    )


def test_legacy_live_state_math_guard_requires_relevant_live_observation() -> None:
    request = _request("is CPU load greater than 10*e")
    request_plan = _plan(
        "calculator.evaluate",
        "tool.system.read.resources",
        "tool.system.read.network",
        live_state_tool_names=("tool.system.read.resources", "tool.system.read.network"),
    )

    assert (
        should_defer_final_answer_for_calculator_evidence(
            request,
            request_plan,
            tool_observation_refs=[
                _completed_ref("tool.system.read.network"),
                _completed_ref("calculator.evaluate", arguments={"expression": "10*e"}),
            ],
            used_tool_calls=1,
        )
        is True
    )
