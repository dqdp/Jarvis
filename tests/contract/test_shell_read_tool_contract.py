from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from assistant_core.domain.events import EventType
from assistant_core.domain.policy import Capability, PolicyDecision, PolicyDecisionOutcome, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolCallRequest, ToolObservationStatus
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry
from assistant_core.tools.shell_read import (
    ProjectShellReadTool,
    ShellExecutionResult,
    ShellExecutionTimeout,
)


pytestmark = pytest.mark.contract


class AllowPolicy:
    def __init__(
        self,
        call_log: list[str] | None = None,
        outcome: PolicyDecisionOutcome = PolicyDecisionOutcome.ALLOW,
    ) -> None:
        self.requests = []
        self.call_log = call_log
        self.outcome = outcome

    async def evaluate_capability_request(self, request):
        if self.call_log is not None:
            self.call_log.append("policy")
        self.requests.append(request)
        return PolicyDecision(
            allowed=self.outcome == PolicyDecisionOutcome.ALLOW,
            code="allowed_shell_read" if self.outcome == PolicyDecisionOutcome.ALLOW else self.outcome.value,
            reason="contract policy decision",
            outcome=self.outcome,
            capability=request.capability,
            risk_classes=request.risk_classes,
            sensitivity=request.sensitivity,
            permission_mode=request.permission_mode,
        )


class RecordingShellExecutor:
    def __init__(
        self,
        result: ShellExecutionResult | None = None,
        *,
        timeout: bool = False,
        call_log: list[str] | None = None,
    ) -> None:
        self.result = result or ShellExecutionResult(exit_code=0, stdout="ok\n", stderr="")
        self.timeout = timeout
        self.call_log = call_log
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        *,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> ShellExecutionResult:
        if self.call_log is not None:
            self.call_log.append("executor")
        self.calls.append(
            {
                "argv": argv,
                "cwd": cwd,
                "env": env,
                "timeout_seconds": timeout_seconds,
            },
        )
        if self.timeout:
            raise ShellExecutionTimeout("timed out")
        return self.result


def _tool(tmp_path: Path, executor: RecordingShellExecutor) -> ProjectShellReadTool:
    return ProjectShellReadTool(
        allowed_roots=[tmp_path],
        executor=executor,
        max_stdout_bytes=10,
        max_stderr_bytes=10,
        max_lines=3,
        timeout_seconds=0.5,
    )


def _gateway(
    tmp_path: Path,
    executor: RecordingShellExecutor,
    *,
    policy: AllowPolicy | None = None,
):
    event_log = InMemoryEventLog()
    policy = policy or AllowPolicy()
    gateway = ToolGateway(
        registry=ToolRegistry([_tool(tmp_path, executor)]),
        policy=policy,
        event_log=event_log,
    )
    return gateway, policy, event_log


def _request(
    tmp_path: Path,
    argv: list[str],
    *,
    sensitivity: Sensitivity = Sensitivity.PROJECT,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name="tool.shell.read.project",
        arguments={"argv": argv, "cwd": str(tmp_path)},
        request_id="req-shell-contract",
        conversation_id="conv-shell-contract",
        user_id="user-shell-contract",
        working_directory=str(tmp_path),
        sensitivity=sensitivity,
    )


def _request_with_cwd(root: Path, cwd: Path, argv: list[str]) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name="tool.shell.read.project",
        arguments={"argv": argv, "cwd": str(cwd)},
        request_id="req-shell-contract",
        conversation_id="conv-shell-contract",
        user_id="user-shell-contract",
        working_directory=str(cwd),
    )


def _json_content(observation) -> dict[str, Any]:
    return json.loads(observation.content)


def test_shell_read_tool_executes_allowed_argv_command(tmp_path: Path) -> None:
    executor = RecordingShellExecutor(ShellExecutionResult(exit_code=0, stdout="cwd\n", stderr=""))
    gateway, policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert _json_content(observation)["stdout"] == "cwd\n"
    assert executor.calls[0]["argv"] == ["pwd"]
    assert policy.requests[0].capability == Capability.TOOL_SHELL_READ
    assert policy.requests[0].risk_classes == frozenset({RiskClass.READ_ONLY})


def test_shell_read_tool_uses_request_working_directory_when_cwd_argument_missing(
    tmp_path: Path,
) -> None:
    executor = RecordingShellExecutor(ShellExecutionResult(exit_code=0, stdout="cwd\n", stderr=""))
    gateway, policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(
        gateway.invoke(
            ToolCallRequest(
                tool_name="tool.shell.read.project",
                arguments={"argv": ["pwd"]},
                request_id="req-shell-contract",
                conversation_id="conv-shell-contract",
                user_id="user-shell-contract",
                working_directory=str(tmp_path),
                sensitivity=Sensitivity.PROJECT,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert executor.calls[0]["cwd"] == tmp_path
    assert policy.requests[0].working_directory == str(tmp_path)


def test_shell_read_tool_prefers_request_working_directory_over_cwd_argument(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent
    executor = RecordingShellExecutor(ShellExecutionResult(exit_code=0, stdout="cwd\n", stderr=""))
    gateway, policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(
        gateway.invoke(
            ToolCallRequest(
                tool_name="tool.shell.read.project",
                arguments={"argv": ["pwd"], "cwd": str(outside)},
                request_id="req-shell-contract",
                conversation_id="conv-shell-contract",
                user_id="user-shell-contract",
                working_directory=str(tmp_path),
                sensitivity=Sensitivity.PROJECT,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert executor.calls[0]["cwd"] == tmp_path
    assert policy.requests[0].working_directory == str(tmp_path)


def test_shell_read_tool_does_not_use_cwd_argument_as_request_scope(
    tmp_path: Path,
) -> None:
    executor = RecordingShellExecutor(ShellExecutionResult(exit_code=0, stdout="cwd\n", stderr=""))
    gateway, policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(
        gateway.invoke(
            ToolCallRequest(
                tool_name="tool.shell.read.project",
                arguments={"argv": ["pwd"], "cwd": str(tmp_path)},
                request_id="req-shell-contract",
                conversation_id="conv-shell-contract",
                user_id="user-shell-contract",
                sensitivity=Sensitivity.PROJECT,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.DENIED
    assert observation.error["code"] == "working_directory_required"
    assert executor.calls == []
    assert policy.requests == []


def test_shell_read_tool_returns_bounded_stdout(tmp_path: Path) -> None:
    executor = RecordingShellExecutor(ShellExecutionResult(exit_code=0, stdout="a\nb\nc\nd\n", stderr=""))
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))
    content = _json_content(observation)

    assert observation.truncated is True
    assert content["stdout"] == "a\nb\nc\n"
    assert observation.metadata["stdout_truncated"] is True


def test_shell_read_tool_returns_bounded_stderr(tmp_path: Path) -> None:
    executor = RecordingShellExecutor(ShellExecutionResult(exit_code=2, stdout="", stderr="x" * 20))
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))
    content = _json_content(observation)

    assert observation.truncated is True
    assert content["stderr"] == "x" * 10
    assert observation.metadata["stderr_truncated"] is True


def test_shell_read_tool_truncates_large_output_with_metadata(tmp_path: Path) -> None:
    executor = RecordingShellExecutor(ShellExecutionResult(exit_code=0, stdout="x" * 20, stderr=""))
    gateway, _policy, event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-shell-contract")))

    assert observation.truncated is True
    assert observation.metadata["raw_stdout_bytes"] == 20
    assert EventType.TOOL_SHELL_OUTPUT_TRUNCATED in [event.event_type for event in events]


def test_shell_read_tool_times_out(tmp_path: Path) -> None:
    executor = RecordingShellExecutor(timeout=True)
    gateway, _policy, event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-shell-contract")))

    assert observation.status == ToolObservationStatus.TIMEOUT
    assert EventType.TOOL_SHELL_TIMEOUT in [event.event_type for event in events]


def test_shell_read_tool_emits_classified_started_completed_events(tmp_path: Path) -> None:
    executor = RecordingShellExecutor(ShellExecutionResult(exit_code=0, stdout="ok\n", stderr=""))
    gateway, _policy, event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-shell-contract")))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert [
        event.event_type
        for event in events
        if event.event_type.value.startswith("tool.shell.")
    ] == [
        EventType.TOOL_SHELL_CLASSIFIED,
        EventType.TOOL_SHELL_STARTED,
        EventType.TOOL_SHELL_COMPLETED,
    ]
    completed = next(event for event in events if event.event_type == EventType.TOOL_SHELL_COMPLETED)
    assert completed.payload["policy_outcome"] == PolicyDecisionOutcome.ALLOW.value
    assert isinstance(completed.payload["duration_ms"], int)


def test_shell_read_tool_observation_is_at_least_project_sensitivity(
    tmp_path: Path,
) -> None:
    executor = RecordingShellExecutor(ShellExecutionResult(exit_code=0, stdout="ok\n", stderr=""))
    gateway, _policy, event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(
        gateway.invoke(_request(tmp_path, ["pwd"], sensitivity=Sensitivity.PUBLIC)),
    )
    events = asyncio.run(event_log.query(EventFilter(request_id="req-shell-contract")))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.sensitivity == Sensitivity.PROJECT
    shell_completed = next(event for event in events if event.event_type == EventType.TOOL_SHELL_COMPLETED)
    observation_event = next(
        event for event in events if event.event_type == EventType.TOOL_OBSERVATION_RECORDED
    )
    assert shell_completed.sensitivity == Sensitivity.PROJECT
    assert observation_event.sensitivity == Sensitivity.PROJECT


def test_shell_read_tool_approval_required_observation_is_at_least_project_sensitivity(
    tmp_path: Path,
) -> None:
    executor = RecordingShellExecutor()
    gateway, _policy, event_log = _gateway(
        tmp_path,
        executor,
        policy=AllowPolicy(outcome=PolicyDecisionOutcome.APPROVAL_REQUIRED),
    )

    observation = asyncio.run(
        gateway.invoke(_request(tmp_path, ["pwd"], sensitivity=Sensitivity.PUBLIC)),
    )
    events = asyncio.run(event_log.query(EventFilter(request_id="req-shell-contract")))

    assert observation.status == ToolObservationStatus.APPROVAL_REQUIRED
    assert observation.sensitivity == Sensitivity.PROJECT
    observation_event = next(
        event for event in events if event.event_type == EventType.TOOL_OBSERVATION_RECORDED
    )
    assert observation_event.sensitivity == Sensitivity.PROJECT


def test_shell_read_tool_emits_denied_event_without_execution(tmp_path: Path) -> None:
    executor = RecordingShellExecutor()
    gateway, policy, event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["rm", "notes.md"])))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-shell-contract")))

    assert observation.status == ToolObservationStatus.DENIED
    assert observation.error["code"] == "command_family_denied"
    assert executor.calls == []
    assert len(policy.requests) == 1
    assert EventType.TOOL_SHELL_DENIED in [event.event_type for event in events]
    assert EventType.TOOL_SHELL_STARTED not in [event.event_type for event in events]
    denied = next(event for event in events if event.event_type == EventType.TOOL_SHELL_DENIED)
    assert denied.payload["policy_decision_id"]


def test_shell_read_tool_redacts_secret_like_denied_cwd_in_shell_events(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    secret_cwd = tmp_path / ".ssh"
    workspace.mkdir()
    secret_cwd.mkdir()
    executor = RecordingShellExecutor()
    gateway, _policy, event_log = _gateway(workspace, executor)

    observation = asyncio.run(
        gateway.invoke(_request_with_cwd(workspace, secret_cwd, ["pwd"])),
    )
    events = asyncio.run(event_log.query(EventFilter(request_id="req-shell-contract")))

    assert observation.status == ToolObservationStatus.DENIED
    assert executor.calls == []
    shell_payloads = [
        event.payload
        for event in events
        if event.event_type
        in {EventType.TOOL_SHELL_CLASSIFIED, EventType.TOOL_SHELL_DENIED}
    ]
    assert shell_payloads
    assert str(secret_cwd) not in str(shell_payloads)
    assert all(payload["cwd"] == "<redacted>" for payload in shell_payloads)


def test_shell_read_tool_redacts_secret_like_output_names(tmp_path: Path) -> None:
    executor = RecordingShellExecutor(
        ShellExecutionResult(exit_code=0, stdout="D .env\n", stderr=""),
    )
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))

    assert observation.status == ToolObservationStatus.COMPLETED
    content = _json_content(observation)
    assert content == {"redacted": True}


def test_shell_read_tool_redacts_secret_like_certificate_output_names(tmp_path: Path) -> None:
    executor = RecordingShellExecutor(
        ShellExecutionResult(exit_code=0, stdout="D docs/client.pem\n", stderr=""),
    )
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))

    assert observation.status == ToolObservationStatus.COMPLETED
    content = _json_content(observation)
    assert content == {"redacted": True}


def test_shell_read_tool_gateway_redacts_secret_marker_as_json(tmp_path: Path) -> None:
    executor = RecordingShellExecutor(
        ShellExecutionResult(exit_code=0, stdout="-----BEGIN PRIVATE KEY-----\n", stderr=""),
    )
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    content = _json_content(observation)
    assert content == {"redacted": True}


def test_shell_read_tool_uses_minimal_environment(tmp_path: Path) -> None:
    executor = RecordingShellExecutor()
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))

    env = executor.calls[0]["env"]
    assert sorted(env) == [
        "GIT_CONFIG_NOSYSTEM",
        "GIT_EXTERNAL_DIFF",
        "GIT_OPTIONAL_LOCKS",
        "GIT_PAGER",
        "LANG",
        "LC_ALL",
        "PATH",
    ]
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert "HOME" not in env
    assert "TOKEN" not in str(env).upper()


def test_shell_read_tool_hardens_git_execution_arguments(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    executor = RecordingShellExecutor()
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(
        gateway.invoke(_request(tmp_path, ["git", "diff", "--", "notes.md"])),
    )

    argv = executor.calls[0]["argv"]
    assert observation.status == ToolObservationStatus.COMPLETED
    assert argv[:10] == [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.pager=cat",
        "-c",
        "diff.external=",
        "-c",
        "pager.show=false",
        "--no-pager",
    ]
    assert argv[10:13] == ["diff", "--no-ext-diff", "--no-textconv"]


def test_toolgateway_consults_policy_before_shell_execution(tmp_path: Path) -> None:
    call_log: list[str] = []
    executor = RecordingShellExecutor(call_log=call_log)
    event_log = InMemoryEventLog()
    policy = AllowPolicy(call_log=call_log)
    gateway = ToolGateway(
        registry=ToolRegistry([_tool(tmp_path, executor)]),
        policy=policy,
        event_log=event_log,
    )

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["pwd"])))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert call_log == ["policy", "executor"]
