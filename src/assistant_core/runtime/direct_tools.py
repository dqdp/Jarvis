from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Any

from assistant_core.domain.loop_selection import (
    LoopSelectionDecision,
    SelectionDecisionStatus,
)
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.policy import Capability
from assistant_core.runtime.routing import CapabilityRoutingRegistry, DirectScenarioDescriptor


_DIRECT_PLAN_METADATA_KEY = "loop_selection_direct_tool_plan"
_DIRECT_PLAN_VERSION = 1
_DIRECT_AUTHORITY_SOURCES = {
    "deterministic",
    "guardrail",
    "fake",
    "override",
    "request_resolver",
}
_DIRECT_PATTERN_SHELL_SYNTAX_MARKERS = (
    "|",
    ";",
    "&&",
    "||",
    ">",
    "<",
    "`",
    "$(",
    "\n",
    "\r",
)


@dataclass(frozen=True)
class DirectToolPlan:
    scenario: str
    tool_names: tuple[str, ...]
    capabilities: tuple[Capability, ...]
    scope_hint: str | None
    classification_source: str
    provenance: tuple[str, ...] = ()
    required_arguments: dict[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_names", tuple(self.tool_names))
        object.__setattr__(
            self,
            "capabilities",
            tuple(
                capability if isinstance(capability, Capability) else Capability(capability)
                for capability in self.capabilities
            ),
        )
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "required_arguments", dict(self.required_arguments or {}))

    def redacted_metadata(self) -> dict[str, Any]:
        return {
            "version": _DIRECT_PLAN_VERSION,
            "scenario": self.scenario,
            "tool_names": list(self.tool_names),
            "capabilities": [capability.value for capability in self.capabilities],
            "scope_hint": self.scope_hint,
            "classification_source": self.classification_source,
            "provenance": list(self.provenance),
            "required_arguments": dict(self.required_arguments or {}),
        }


class DirectToolPlanner:
    def __init__(self, routing_registry: CapabilityRoutingRegistry) -> None:
        self._routing_registry = routing_registry

    def plan(
        self,
        decision: LoopSelectionDecision,
        *,
        user_input: str,
    ) -> DirectToolPlan | None:
        if decision.selected_loop_strategy is not LoopStrategyName.TOOL_REACT_LOOP:
            return None
        if decision.decision_status is not SelectionDecisionStatus.SELECTED:
            return None
        if decision.classification_source not in _DIRECT_AUTHORITY_SOURCES:
            return None
        scenarios = tuple(
            scenario
            for candidate in decision.candidate_capabilities
            if (scenario := self._routing_registry.direct_scenario_for_candidate(candidate))
            is not None
        )
        if not scenarios or len(scenarios) != len(decision.candidate_capabilities):
            return None
        if not _compatible_scenarios(scenarios):
            return None
        required_arguments: dict[str, str] = {}
        provenance = []
        for candidate, scenario in zip(decision.candidate_capabilities, scenarios, strict=True):
            provenance.extend(candidate.evidence_codes)
            if scenario.requires_argument_extractor == "process_name_search_pattern":
                pattern = _process_search_pattern(user_input)
                if pattern is None:
                    return None
                required_arguments["process_pattern"] = pattern
                provenance.append("process_name_search_pattern")
        return DirectToolPlan(
            scenario=scenarios[0].scenario,
            tool_names=tuple(tool_name for scenario in scenarios for tool_name in scenario.tool_names),
            capabilities=tuple(
                capability for scenario in scenarios for capability in scenario.capabilities
            ),
            scope_hint=scenarios[0].scope_hint,
            classification_source=decision.classification_source,
            provenance=tuple(dict.fromkeys(provenance)),
            required_arguments=required_arguments,
        )


def direct_tool_plan_from_metadata(metadata: dict[str, Any]) -> DirectToolPlan | None:
    payload = metadata.get(_DIRECT_PLAN_METADATA_KEY)
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _DIRECT_PLAN_VERSION:
        return None
    try:
        scenario = payload["scenario"]
        tool_names = _string_tuple(payload["tool_names"])
        capability_values = _string_tuple(payload["capabilities"])
        scope_hint = payload.get("scope_hint")
        classification_source = payload["classification_source"]
        provenance = _string_tuple(payload.get("provenance", ()))
        required_arguments = _required_arguments_from_metadata(
            payload.get("required_arguments", {}),
        )
        if (
            not isinstance(scenario, str)
            or tool_names is None
            or capability_values is None
            or (scope_hint is not None and not isinstance(scope_hint, str))
            or not isinstance(classification_source, str)
            or provenance is None
            or required_arguments is None
        ):
            return None
        plan = DirectToolPlan(
            scenario=scenario,
            tool_names=tool_names,
            capabilities=tuple(Capability(value) for value in capability_values),
            scope_hint=scope_hint,
            classification_source=classification_source,
            provenance=provenance,
            required_arguments=required_arguments,
        )
        return plan if _valid_direct_plan_from_metadata(plan) else None
    except (KeyError, TypeError, ValueError):
        return None


def _string_tuple(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _required_arguments_from_metadata(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        return None
    return dict(value)


def direct_tool_arguments(
    plan: DirectToolPlan,
    tool_name: str,
    *,
    working_directory: str | None,
    platform: str | None = None,
) -> dict[str, Any]:
    if tool_name == "datetime.now":
        return {}
    cwd = working_directory or "."
    return {
        "argv": _direct_tool_argv(plan, tool_name, platform=platform or sys.platform),
        "cwd": cwd,
    }


def _valid_direct_plan_from_metadata(plan: DirectToolPlan) -> bool:
    if plan.classification_source not in _DIRECT_AUTHORITY_SOURCES:
        return False
    scenarios = _scenarios_for_plan(plan)
    if not scenarios or not _compatible_scenarios(scenarios):
        return False
    expected_arguments: set[str] = set()
    for scenario in scenarios:
        if scenario.requires_argument_extractor is None:
            continue
        if scenario.requires_argument_extractor != "process_name_search_pattern":
            return False
        expected_arguments.add("process_pattern")
    if set(plan.required_arguments or {}) != expected_arguments:
        return False
    if plan.scenario == "process_name_search":
        pattern = (plan.required_arguments or {}).get("process_pattern")
        return pattern is not None and _safe_direct_pattern(pattern)
    return True


def _scenarios_for_plan(plan: DirectToolPlan) -> tuple[DirectScenarioDescriptor, ...]:
    if len(plan.tool_names) != len(plan.capabilities):
        return ()
    registry = CapabilityRoutingRegistry()
    scenarios: list[DirectScenarioDescriptor] = []
    for tool_name, capability in zip(plan.tool_names, plan.capabilities, strict=True):
        matches = tuple(
            scenario
            for scenario in registry.direct_scenarios()
            if scenario.scenario == plan.scenario
            and scenario.scope_hint == plan.scope_hint
            and scenario.tool_names == (tool_name,)
            and scenario.capabilities == (capability,)
        )
        if len(matches) != 1:
            return ()
        scenarios.append(matches[0])
    return tuple(scenarios)


def _compatible_scenarios(scenarios: tuple[DirectScenarioDescriptor, ...]) -> bool:
    scenario_names = {scenario.scenario for scenario in scenarios}
    if len(scenarios) == 1:
        return True
    return scenario_names == {"cpu_overview"} and {
        tool_name for scenario in scenarios for tool_name in scenario.tool_names
    } == {"tool.system.read.hardware", "tool.system.read.resources"}


def _direct_tool_argv(
    plan: DirectToolPlan,
    tool_name: str,
    *,
    platform: str,
) -> list[str]:
    if tool_name == "datetime.now":
        return []
    if tool_name == "tool.system.read.hardware":
        if plan.scenario == "os_version":
            return ["sw_vers"] if platform == "darwin" else ["uname", "-a"]
        if plan.scenario == "battery_charge":
            return (
                ["pmset", "-g", "batt"]
                if platform == "darwin"
                else ["upower", "-i", "/org/freedesktop/UPower/devices/DisplayDevice"]
            )
        return ["sysctl", "-n", "hw.logicalcpu"] if platform == "darwin" else ["lscpu"]
    if tool_name == "tool.system.read.sensors":
        if platform == "darwin":
            return ["powermetrics", "--samplers", "thermal", "-n", "1"]
        if platform.startswith("linux"):
            return ["thermal-sysfs"]
        return ["sensors"]
    if tool_name == "tool.system.read.resources":
        if plan.scenario == "disk_free":
            return ["df", "-h"]
        if plan.scenario == "cpu_overview":
            return ["top", "-l", "1", "-n", "0"] if platform == "darwin" else ["top", "-b", "-n", "1"]
        return ["vm_stat"] if platform == "darwin" else ["free", "-m"]
    if tool_name == "tool.system.read.network":
        return ["scutil", "--nc", "list"] if platform == "darwin" else ["ip", "addr"]
    if tool_name == "tool.system.read.process":
        pattern = (plan.required_arguments or {}).get("process_pattern", "")
        return ["pgrep", "-l", re.escape(pattern or "__jarvis_missing_process_name__")]
    return []


def _process_search_pattern(text: str) -> str | None:
    return _quoted_process_pattern(text) or _unquoted_process_pattern(text)


def _quoted_process_pattern(text: str) -> str | None:
    for pattern in (
        r'"([^"]+)"',
        r"'([^']+)'",
        r"«([^»]+)»",
    ):
        match = re.search(pattern, text)
        if match is None:
            continue
        value = match.group(1).strip()
        if _safe_direct_pattern(value):
            return value
    return None


def _unquoted_process_pattern(text: str) -> str | None:
    for pattern in (
        r"(?:process|процесс(?:а|е|ом)?)(?:\s+(?:named|called|с именем|имени|под названием))?\s+(?P<value>[A-Za-z0-9_.:-]{1,128})",
        r"(?P<value>[A-Za-z0-9_.:-]{1,128})\s+(?:process|процесс(?:а|е|ом)?)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        value = match.group("value").strip()
        if value.casefold() in {"process", "процесс", "now", "сейчас"}:
            continue
        if _safe_direct_pattern(value):
            return value
    return None


def _safe_direct_pattern(value: str) -> bool:
    return 0 < len(value) <= 128 and not any(
        marker in value for marker in _DIRECT_PATTERN_SHELL_SYNTAX_MARKERS
    )
