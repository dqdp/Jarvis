from __future__ import annotations

import asyncio

import pytest

from assistant_core.domain.events import EventType
from assistant_core.domain.policy import PolicyDecision, PolicyDecisionOutcome
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolCallRequest, ToolObservationStatus, ToolParseStatus
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.ports.tools import ToolGatewayPort
from assistant_core.tools.builtin import (
    calendar_diff_tool,
    calculator_tool,
    daemon_status_tool,
    datetime_diff_tool,
    datetime_now_tool,
    datetime_until_tool,
)
from assistant_core.tools.fake import fake_echo_tool, fake_fail_tool
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry


pytestmark = pytest.mark.contract


class AllowPolicy:
    def __init__(self, outcome: PolicyDecisionOutcome = PolicyDecisionOutcome.ALLOW) -> None:
        self.outcome = outcome
        self.requests = []

    async def evaluate_capability_request(self, request):
        self.requests.append(request)
        return PolicyDecision(
            allowed=self.outcome == PolicyDecisionOutcome.ALLOW,
            code=self.outcome.value,
            reason="contract policy decision",
            outcome=self.outcome,
            capability=request.capability,
            risk_classes=request.risk_classes,
            sensitivity=request.sensitivity,
            permission_mode=request.permission_mode,
        )


def _gateway(
    *,
    policy: AllowPolicy | None = None,
    event_log: InMemoryEventLog | None = None,
) -> tuple[ToolGatewayPort, AllowPolicy, InMemoryEventLog]:
    effective_policy = policy or AllowPolicy()
    effective_event_log = event_log or InMemoryEventLog()
    gateway = ToolGateway(
        registry=ToolRegistry(
            [
                fake_echo_tool(),
                fake_fail_tool(),
                calendar_diff_tool(),
                datetime_diff_tool(),
                datetime_now_tool(),
                datetime_until_tool(),
                calculator_tool(),
                daemon_status_tool(),
            ],
        ),
        policy=effective_policy,
        event_log=effective_event_log,
    )
    return gateway, effective_policy, effective_event_log


def _request(
    tool_name: str,
    arguments: dict[str, object] | None = None,
    *,
    sensitivity: Sensitivity = Sensitivity.PROJECT,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name=tool_name,
        arguments=arguments or {},
        request_id="req-contract-tool",
        conversation_id="conv-contract-tool",
        user_id="user-contract-tool",
        sensitivity=sensitivity,
    )


def test_tool_gateway_lists_enabled_tools() -> None:
    gateway, _policy, _event_log = _gateway()

    specs = asyncio.run(gateway.list_tools())

    assert [spec.name for spec in specs] == [
        "calculator.evaluate",
        "calendar.diff",
        "daemon.status",
        "datetime.diff",
        "datetime.now",
        "datetime.until",
        "fake.echo",
        "fake.fail",
    ]


def test_tool_gateway_gets_tool_by_name() -> None:
    gateway, _policy, _event_log = _gateway()

    spec = asyncio.run(gateway.get_tool("fake.echo"))

    assert spec is not None
    assert spec.name == "fake.echo"


def test_tool_gateway_invokes_fake_echo() -> None:
    gateway, policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(_request("fake.echo", {"message": "hello"})),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content == "hello"
    assert policy.requests[0].tool_name == "fake.echo"


def test_tool_gateway_invokes_datetime_now() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(gateway.invoke(_request("datetime.now", sensitivity=Sensitivity.PUBLIC)))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    assert observation.structured_schema == "datetime.now"
    assert "iso" in observation.content


def test_tool_gateway_invokes_daemon_status_as_typed_json() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(_request("daemon.status", sensitivity=Sensitivity.PUBLIC))
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    assert observation.structured_schema == "daemon.status"
    assert observation.parse_status is ToolParseStatus.PARSED
    assert observation.structured_content == {"status": "ok"}


def test_tool_gateway_invokes_datetime_until_next_new_year() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(
            _request(
                "datetime.until",
                {
                    "from_iso": "2026-12-31T23:59:50+03:00",
                    "target": "next_new_year",
                    "unit": "seconds",
                },
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    assert '"seconds": 10' in observation.content
    assert '"target_iso": "2027-01-01T00:00:00+03:00"' in observation.content


def test_tool_gateway_invokes_datetime_until_without_source_timestamp() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(
            _request(
                "datetime.until",
                {
                    "target": "next_new_year",
                    "unit": "seconds",
                },
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    assert '"target": "next_new_year"' in observation.content
    assert '"unit": "seconds"' in observation.content
    assert '"seconds":' in observation.content


def test_tool_gateway_invokes_datetime_until_with_subsecond_units() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(
            _request(
                "datetime.until",
                {
                    "from_iso": "2026-12-31T23:59:59.500000+03:00",
                    "target": "next_new_year",
                    "unit": "microseconds",
                },
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.structured_schema == "datetime.until"
    assert observation.structured_content["microseconds"] == 500000
    assert observation.structured_content["milliseconds"] == 500.0
    assert observation.structured_content["seconds"] == 0.5
    assert observation.structured_content["value"] == 500000


def test_tool_gateway_invokes_datetime_diff_between_timestamps() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(
            _request(
                "datetime.diff",
                {
                    "from_iso": "2026-06-07T20:17:00+03:00",
                    "to_iso": "2026-06-07T20:47:30+03:00",
                    "unit": "minutes",
                },
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    assert observation.structured_schema == "datetime.diff"
    assert observation.structured_content == {
        "from_iso": "2026-06-07T20:17:00+03:00",
        "to_iso": "2026-06-07T20:47:30+03:00",
        "microseconds": 1830000000,
        "milliseconds": 1830000.0,
        "seconds": 1830,
        "minutes": 30.5,
        "hours": 0.5083333333333333,
        "days": 0.021180555555555557,
        "weeks": 0.0030257936507936507,
        "unit": "minutes",
        "value": 30.5,
        "absolute": False,
    }


@pytest.mark.parametrize(
    ("unit", "expected_value"),
    [
        ("microseconds", 1830000000),
        ("milliseconds", 1830000.0),
        ("weeks", 0.0030257936507936507),
    ],
)
def test_tool_gateway_invokes_datetime_diff_for_extended_time_units(
    unit: str,
    expected_value: float,
) -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(
            _request(
                "datetime.diff",
                {
                    "from_iso": "2026-06-07T20:17:00+03:00",
                    "to_iso": "2026-06-07T20:47:30+03:00",
                    "unit": unit,
                },
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.structured_schema == "datetime.diff"
    assert observation.structured_content["unit"] == unit
    assert observation.structured_content["value"] == expected_value


@pytest.mark.parametrize(
    ("unit", "expected_value"),
    [
        ("microseconds", 11145600000000),
        ("minutes", 185760.0),
        ("hours", 3096.0),
        ("days", 129.0),
        ("weeks", 18.428571428571427),
        ("months", 4),
        ("quarters", 1),
        ("decades", 0),
    ],
)
def test_tool_gateway_invokes_calendar_diff_between_timestamps(
    unit: str,
    expected_value: int | float,
) -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(
            _request(
                "calendar.diff",
                {
                    "from_iso": "2025-11-27T00:00:00+00:00",
                    "to_iso": "2026-04-05T00:00:00+00:00",
                    "unit": unit,
                },
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    assert observation.structured_schema == "calendar.diff"
    assert observation.structured_content["unit"] == unit
    assert observation.structured_content["value"] == expected_value


def test_tool_gateway_invokes_calculator_evaluate() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(
            _request(
                "calculator.evaluate",
                {"expression": "1 + 2 * 3"},
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content == "7"


def test_tool_gateway_invokes_scientific_calculator_expression() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(
            _request(
                "calculator.evaluate",
                {"expression": "(42^3)^2 - 123 * 432 + sqrt(81) + sin(pi / 2) + log(e)"},
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content == "5488978619"


def test_tool_gateway_invokes_daemon_status() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(gateway.invoke(_request("daemon.status", sensitivity=Sensitivity.PUBLIC)))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    assert '"status": "ok"' in observation.content


def test_tool_gateway_records_completed_lifecycle_events() -> None:
    event_log = InMemoryEventLog()
    gateway, _policy, _event_log = _gateway(event_log=event_log)

    observation = asyncio.run(
        gateway.invoke(_request("fake.echo", {"message": "hello"})),
    )
    events = asyncio.run(event_log.query(EventFilter(request_id="req-contract-tool")))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert [event.event_type for event in events] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_OBSERVATION_RECORDED,
    ]


def test_tool_gateway_records_failed_lifecycle_events() -> None:
    event_log = InMemoryEventLog()
    gateway, _policy, _event_log = _gateway(event_log=event_log)

    observation = asyncio.run(gateway.invoke(_request("fake.fail")))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-contract-tool")))

    assert observation.status == ToolObservationStatus.FAILED
    assert EventType.TOOL_CALL_FAILED in [event.event_type for event in events]
    assert events[-1].event_type == EventType.TOOL_OBSERVATION_RECORDED


def test_tool_gateway_records_denied_lifecycle_events() -> None:
    event_log = InMemoryEventLog()
    gateway, _policy, _event_log = _gateway(
        policy=AllowPolicy(PolicyDecisionOutcome.DENY),
        event_log=event_log,
    )

    observation = asyncio.run(
        gateway.invoke(_request("fake.echo", {"message": "hello"})),
    )
    events = asyncio.run(event_log.query(EventFilter(request_id="req-contract-tool")))

    assert observation.status == ToolObservationStatus.DENIED
    assert [event.event_type for event in events] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_DENIED,
        EventType.TOOL_OBSERVATION_RECORDED,
    ]
