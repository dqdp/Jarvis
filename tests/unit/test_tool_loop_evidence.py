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
from assistant_core.domain.tools import ToolObservationStatus
from assistant_core.runtime.loops.tool_loop_evidence import (
    LiveStateEvidenceFamily,
    LiveStateEvidencePlan,
    contains_live_state_intent,
    detect_live_state_family,
    live_state_evidence_plan,
    request_requires_initial_tool_evidence,
    request_needs_live_state_math_evidence,
    should_defer_final_answer_for_calculator_evidence,
)


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
) -> ToolObservationRef:
    return ToolObservationRef(
        tool_call_id=f"tool-call-{tool_name}",
        tool_name=tool_name,
        status=ToolObservationStatus.COMPLETED,
        content=content,
        content_type="application/json",
        sensitivity=Sensitivity.PROJECT,
        structured_content=structured_content,
        arguments=arguments or {},
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


def test_live_state_evidence_plan_accepts_completed_datetime_until_for_countdown() -> None:
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
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason is None


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
                structured_content={"iso": now_iso},
            ),
            _completed_ref(
                "datetime.until",
                arguments={
                    "target": "next_new_year",
                    "unit": "seconds",
                    "from_iso": now_iso,
                },
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_accepts_datetime_until_with_matching_datetime_now_content_source() -> None:
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
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_accepts_completed_datetime_now_for_countdown() -> None:
    plan = live_state_evidence_plan(
        _request("how many seconds until Christmas?"),
        _plan(
            "datetime.now",
            "datetime.until",
            live_state_tool_names=("datetime.now", "datetime.until"),
        ),
        tool_observation_refs=(_completed_ref("datetime.now"),),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason is None


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
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})


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
        tool_observation_refs=(_completed_ref("tool.system.read.resources"),),
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

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert LiveStateEvidenceFamily.SYSTEM_RESOURCES in plan.families
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )
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

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert LiveStateEvidenceFamily.SYSTEM_RESOURCES in plan.families
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )
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

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert LiveStateEvidenceFamily.SYSTEM_RESOURCES in plan.families
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )
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

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert LiveStateEvidenceFamily.SYSTEM_RESOURCES in plan.families
    assert plan.candidate_tool_names == frozenset(
        {"tool.system.read.resources", "calculator.evaluate"}
    )
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
            _completed_ref("tool.system.read.resources"),
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
            _completed_ref("tool.system.read.resources"),
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
    assert plan.candidate_tool_names == frozenset({"datetime.now"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})


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
        {"tool.system.read.network", "datetime.now"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


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
                arguments={"target": "next_new_year", "unit": "seconds"},
            ),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.CURRENT_TIME
    assert plan.candidate_tool_names == frozenset({"datetime.now", "datetime.until"})
    assert plan.missing_tool_names == frozenset({"datetime.now"})


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
        {"tool.system.read.network", "datetime.now"}
    )
    assert plan.missing_tool_names == plan.candidate_tool_names


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
        assert plan.candidate_tool_names == frozenset({"datetime.now"})
        assert plan.missing_tool_names == frozenset({"datetime.now"})


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
def test_live_state_evidence_plan_treats_threshold_comparison_as_live_state_math(
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

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert "calculator.evaluate" in plan.candidate_tool_names
    assert plan.missing_tool_names == plan.candidate_tool_names


def test_live_state_evidence_plan_keeps_threshold_calculator_missing_for_unsupported_comparison() -> None:
    plan = live_state_evidence_plan(
        _request("is current CPU usage over 80%?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref("tool.system.read.resources"),
            _completed_ref("calculator.evaluate", arguments={"expression": "72 > 80"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset({"calculator.evaluate"})


def test_live_state_evidence_plan_keeps_threshold_calculator_missing_after_mismatch() -> None:
    plan = live_state_evidence_plan(
        _request("is current CPU usage over 80%?"),
        _plan(
            "calculator.evaluate",
            "tool.system.read.resources",
            live_state_tool_names=("tool.system.read.resources",),
        ),
        tool_observation_refs=(
            _completed_ref("tool.system.read.resources"),
            _completed_ref("calculator.evaluate", arguments={"expression": "72 > 70"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset({"calculator.evaluate"})


@pytest.mark.parametrize("expression", ["80", "80+1", "180-100", "1 < 80", "80 == 80", "80 > 1"])
def test_live_state_evidence_plan_keeps_threshold_calculator_missing_for_non_comparison(
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
            _completed_ref("tool.system.read.resources"),
            _completed_ref("calculator.evaluate", arguments={"expression": expression}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset({"calculator.evaluate"})


@pytest.mark.parametrize(
    ("user_input", "expected_family"),
    [
        (
            "is the Python process memory usage greater than 10*e?",
            LiveStateEvidenceFamily.LIVE_STATE_MATH,
        ),
        (
            "what is current CPU usage of the Python process?",
            LiveStateEvidenceFamily.SYSTEM_RESOURCES,
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
    assert "tool.system.read.resources" in plan.candidate_tool_names
    if expected_family is LiveStateEvidenceFamily.LIVE_STATE_MATH:
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
            _completed_ref("tool.system.read.resources"),
            _completed_ref("calculator.evaluate", arguments={"expression": "10*e"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.evidence_required is True
    assert plan.missing_tool_names == frozenset()


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
            _completed_ref("tool.system.read.resources"),
            _completed_ref("calculator.evaluate", arguments={"expression": "10*(e+1)"}),
        ),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.missing_tool_names == frozenset()


def test_live_state_evidence_plan_reports_unavailable_when_relevant_tool_is_not_allowed() -> None:
    plan = live_state_evidence_plan(
        _request("what is current CPU usage?"),
        _plan("datetime.now", live_state_tool_names=("datetime.now",)),
        tool_observation_refs=(),
    )

    assert plan.family is LiveStateEvidenceFamily.SYSTEM_RESOURCES
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset()
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"


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
        tool_observation_refs=(_completed_ref("tool.system.read.resources"),),
    )

    assert plan.family is LiveStateEvidenceFamily.LIVE_STATE_MATH
    assert plan.evidence_required is True
    assert plan.candidate_tool_names == frozenset({"tool.system.read.resources"})
    assert plan.missing_tool_names == frozenset()
    assert plan.unavailable_reason == "live_state_tool_unavailable"


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
