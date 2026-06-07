from __future__ import annotations

import asyncio
from datetime import datetime

from assistant_core.domain.conversations import UpdateAssistantRequestStatusCommand
from assistant_core.domain.loops import LoopExecutionRequest, ToolObservationRef, ToolProposal, ToolProposalParseError
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.tools import ToolCallRequest, ToolObservationStatus
from assistant_core.privacy.redaction import redact_content
from assistant_core.runtime.loops.tool_loop_time_delta_units import CALENDAR_DIFF_UNITS, DATETIME_DIFF_UNITS


class ToolProposalExecutor:
    def __init__(
        self,
        *,
        tool_gateway,
        conversation_store,
        approval_waiter=None,
        state_recorder=None,
    ) -> None:
        self._tool_gateway = tool_gateway
        self._conversation_store = conversation_store
        self._approval_waiter = approval_waiter
        self._state_recorder = state_recorder

    async def execute(
        self,
        request: LoopExecutionRequest,
        proposal: ToolProposal,
        *,
        step_id: str,
        causation_event_id: str,
        used_tool_calls: int,
        loop_deadline: float,
        step_index: int | None = None,
    ) -> ToolObservationRef:
        if proposal.tool_name is None:
            raise ToolProposalParseError("tool_call requires tool_name")
        _ensure_tool_budget(used_tool_calls=used_tool_calls, request=request)
        await self._record_tool_running(
            request=request,
            proposal=proposal,
            step_id=step_id,
            step_index=step_index,
            causation_event_id=causation_event_id,
        )
        observation = await self._invoke_gateway(
            request,
            proposal,
            step_id=step_id,
            causation_event_id=causation_event_id,
            approval_id=None,
            loop_deadline=loop_deadline,
        )
        if observation.status != ToolObservationStatus.APPROVAL_REQUIRED:
            return ToolObservationRef.from_observation(
                observation,
                arguments=_safe_observation_arguments(proposal),
            )

        observation_ref = ToolObservationRef.from_observation(
            observation,
            arguments=_safe_observation_arguments(proposal),
        )
        approval_id = observation.metadata.get("approval_id")
        if approval_id is None or self._approval_waiter is None:
            await self._record_waiting_approval(
                request=request,
                proposal=proposal,
                step_id=step_id,
                step_index=step_index,
                causation_event_id=causation_event_id,
                observation_ref=observation_ref,
            )
            return observation_ref

        await self._conversation_store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request_id,
                status=RequestStatus.WAITING_APPROVAL,
            ),
        )
        await self._record_waiting_approval(
            request=request,
            proposal=proposal,
            step_id=step_id,
            step_index=step_index,
            causation_event_id=causation_event_id,
            observation_ref=observation_ref,
        )
        await self._approval_waiter.wait(
            approval_id,
            loop_deadline=loop_deadline,
            actor_id=request.user_id,
        )
        await self._conversation_store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        await self._record_tool_running(
            request=request,
            proposal=proposal,
            step_id=step_id,
            step_index=step_index,
            causation_event_id=causation_event_id,
        )
        observation = await self._invoke_gateway(
            request,
            proposal,
            step_id=step_id,
            causation_event_id=causation_event_id,
            approval_id=approval_id,
            loop_deadline=loop_deadline,
        )
        return ToolObservationRef.from_observation(
            observation,
            arguments=_safe_observation_arguments(proposal),
        )

    async def _record_waiting_approval(
        self,
        *,
        request: LoopExecutionRequest,
        proposal: ToolProposal,
        step_id: str,
        step_index: int | None,
        causation_event_id: str,
        observation_ref: ToolObservationRef,
    ) -> None:
        if self._state_recorder is None:
            return
        await self._state_recorder(
            request=request,
            proposal=proposal,
            step_id=step_id,
            step_index=step_index,
            causation_event_id=causation_event_id,
            state="waiting_approval",
            observation_ref=observation_ref,
        )

    async def _record_tool_running(
        self,
        *,
        request: LoopExecutionRequest,
        proposal: ToolProposal,
        step_id: str,
        step_index: int | None,
        causation_event_id: str,
    ) -> None:
        if self._state_recorder is None:
            return
        await self._state_recorder(
            request=request,
            proposal=proposal,
            step_id=step_id,
            step_index=step_index,
            causation_event_id=causation_event_id,
            state="tool_running",
        )

    async def _invoke_gateway(
        self,
        request: LoopExecutionRequest,
        proposal: ToolProposal,
        *,
        step_id: str,
        causation_event_id: str,
        approval_id: str | None,
        loop_deadline: float,
    ):
        assert proposal.tool_name is not None
        return await self._tool_gateway.invoke(
            ToolCallRequest(
                tool_name=proposal.tool_name,
                arguments=proposal.arguments,
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                correlation_id=request.correlation_id or request.request_id,
                causation_event_id=causation_event_id,
                step_id=step_id,
                user_id=request.user_id,
                project_namespace=request.active_project_namespace,
                working_directory=request.working_directory,
                sensitivity=request.current_message_sensitivity,
                permission_mode=request.permission_mode,
                approval_id=approval_id,
                timeout_seconds=_remaining_timeout(
                    loop_deadline,
                    request.budget.max_model_call_seconds,
                ),
                metadata={"loop_strategy": request.strategy_name.value},
            ),
        )


def _ensure_tool_budget(*, used_tool_calls: int, request: LoopExecutionRequest) -> None:
    if used_tool_calls >= request.budget.max_tool_calls:
        raise RuntimeError("max_tool_calls_exceeded")


def _safe_observation_arguments(proposal: ToolProposal) -> dict[str, object]:
    if proposal.tool_name == "calculator.evaluate":
        return _safe_calculator_arguments(proposal)
    if proposal.tool_name == "datetime.until":
        return _safe_datetime_until_arguments(proposal)
    if proposal.tool_name == "datetime.diff":
        return _safe_datetime_diff_arguments(proposal)
    if proposal.tool_name == "calendar.diff":
        return _safe_calendar_diff_arguments(proposal)
    if proposal.tool_name in _SYSTEM_DIAGNOSTICS_TOOL_NAMES:
        return _safe_system_diagnostics_arguments(proposal)
    return {}


_SYSTEM_DIAGNOSTICS_TOOL_NAMES = frozenset(
    {
        "tool.system.read.resources",
        "tool.system.read.network",
        "tool.system.read.hardware",
        "tool.system.read.process",
        "tool.system.read.sensors",
    }
)
_SAFE_SYSTEM_DIAGNOSTICS_ARGV = frozenset(
    {
        ("df", "-h"),
        ("df", "-k"),
        ("df", "-P"),
        ("free",),
        ("free", "-h"),
        ("free", "-m"),
        ("ifconfig",),
        ("ip", "addr"),
        ("lscpu",),
        ("lshw",),
        ("lshw", "-short"),
        ("pmset", "-g", "batt"),
        ("ps", "-Ao", "pid,command"),
        ("ps", "-Ao", "pid,comm,command"),
        ("ps", "-Ao", "pid,ppid,comm,command"),
        ("ps", "-ef"),
        ("ps", "aux"),
        ("scutil", "--nc", "list"),
        ("sw_vers",),
        ("sysctl", "-n", "hw.logicalcpu"),
        ("sysctl", "-n", "hw.memsize"),
        ("sysctl", "-n", "hw.ncpu"),
        ("sysctl", "-n", "hw.physicalcpu"),
        ("sysctl", "-n", "machdep.cpu.brand_string"),
        ("top", "-b", "-n", "1"),
        ("top", "-l", "1"),
        ("top", "-l", "1", "-n", "0"),
        ("uname", "-a"),
        ("upower", "-i", "/org/freedesktop/UPower/devices/DisplayDevice"),
        ("uptime",),
        ("vm_stat",),
    }
)
_SAFE_SYSTEM_RESOURCE_METRICS = frozenset({"resources", "cpu_and_memory"})


def _safe_system_diagnostics_arguments(proposal: ToolProposal) -> dict[str, object]:
    safe_arguments: dict[str, object] = {}
    argv = proposal.arguments.get("argv")
    if (
        isinstance(argv, list)
        and all(isinstance(arg, str) for arg in argv)
        and tuple(argv) in _SAFE_SYSTEM_DIAGNOSTICS_ARGV
    ):
        safe_arguments["argv"] = list(argv)
    metric = proposal.arguments.get("metric")
    if isinstance(metric, str) and metric in _SAFE_SYSTEM_RESOURCE_METRICS:
        safe_arguments["metric"] = metric
    return safe_arguments


def _safe_calculator_arguments(proposal: ToolProposal) -> dict[str, object]:
    expression = proposal.arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        return {}
    expression = expression.strip()[:500]
    if redact_content(expression, content_type="text/plain") != expression:
        return {}
    return {"expression": expression}


def _safe_datetime_until_arguments(proposal: ToolProposal) -> dict[str, object]:
    target = _safe_short_string_argument(proposal, "target", max_length=100)
    unit = _safe_short_string_argument(proposal, "unit", max_length=20)
    if target != "next_new_year" or unit not in DATETIME_DIFF_UNITS:
        return {}
    safe_arguments: dict[str, object] = {"target": target, "unit": unit}
    if "from_iso" in proposal.arguments:
        from_iso = _safe_iso_datetime_argument(proposal, "from_iso", max_length=80)
        if from_iso is None:
            return {}
        safe_arguments["from_iso"] = from_iso
    return safe_arguments


def _safe_datetime_diff_arguments(proposal: ToolProposal) -> dict[str, object]:
    unit = _safe_short_string_argument(proposal, "unit", max_length=20)
    if unit not in DATETIME_DIFF_UNITS:
        return {}
    from_iso = _safe_iso_datetime_argument(proposal, "from_iso", max_length=80)
    to_iso = _safe_iso_datetime_argument(proposal, "to_iso", max_length=80)
    if from_iso is None or to_iso is None:
        return {}
    safe_arguments: dict[str, object] = {
        "from_iso": from_iso,
        "to_iso": to_iso,
        "unit": unit,
    }
    absolute = proposal.arguments.get("absolute")
    if isinstance(absolute, bool):
        safe_arguments["absolute"] = absolute
    return safe_arguments


def _safe_calendar_diff_arguments(proposal: ToolProposal) -> dict[str, object]:
    unit = _safe_short_string_argument(proposal, "unit", max_length=20)
    if unit not in CALENDAR_DIFF_UNITS:
        return {}
    from_iso = _safe_iso_datetime_argument(proposal, "from_iso", max_length=80)
    to_iso = _safe_iso_datetime_argument(proposal, "to_iso", max_length=80)
    if from_iso is None or to_iso is None:
        return {}
    safe_arguments: dict[str, object] = {
        "from_iso": from_iso,
        "to_iso": to_iso,
        "unit": unit,
    }
    absolute = proposal.arguments.get("absolute")
    if isinstance(absolute, bool):
        safe_arguments["absolute"] = absolute
    return safe_arguments


def _safe_iso_datetime_argument(
    proposal: ToolProposal,
    argument_name: str,
    *,
    max_length: int,
) -> str | None:
    value = _safe_short_string_argument(
        proposal,
        argument_name,
        max_length=max_length,
    )
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return value


def _safe_short_string_argument(
    proposal: ToolProposal,
    argument_name: str,
    *,
    max_length: int,
) -> str | None:
    value = proposal.arguments.get(argument_name)
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()[:max_length]
    if redact_content(value, content_type="text/plain") != value:
        return None
    return value


def _remaining_timeout(deadline: float, operation_timeout: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise RuntimeError("max_wall_time_exceeded")
    return min(float(operation_timeout), remaining)
