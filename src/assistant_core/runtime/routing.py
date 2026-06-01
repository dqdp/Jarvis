from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from assistant_core.config.settings import Settings
from assistant_core.domain.loop_selection import (
    CapabilityCandidate,
    IntentClassification,
    IntentFamily,
)
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity


@dataclass(frozen=True)
class RoutingToolDescriptor:
    tool_name: str
    capability: Capability
    intent_families: frozenset[IntentFamily]
    description: str
    requires_live_state: bool
    requires_execution: bool
    requires_write: bool
    risk_classes: frozenset[RiskClass]
    sensitivity_ceiling: Sensitivity
    system_family: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "capability": self.capability.value,
            "sensitivity_ceiling": self.sensitivity_ceiling.value,
            "intent_families": sorted(intent.value for intent in self.intent_families),
            "description": self.description,
            "requires_live_state": self.requires_live_state,
            "requires_execution": self.requires_execution,
            "requires_write": self.requires_write,
            "risk_classes": sorted(risk.value for risk in self.risk_classes),
        }


@dataclass(frozen=True)
class DirectScenarioDescriptor:
    scenario: str
    tool_names: tuple[str, ...]
    capabilities: tuple[Capability, ...]
    intent_family: IntentFamily
    scope_hint: str | None
    requires_argument_extractor: str | None = None


class CapabilityRoutingRegistry:
    def __init__(
        self,
        descriptors: tuple[RoutingToolDescriptor, ...] | None = None,
        *,
        direct_scenarios: tuple[DirectScenarioDescriptor, ...] | None = None,
        enabled_tool_names: frozenset[str] | None = None,
    ) -> None:
        descriptors = descriptors or _DEFAULT_TOOL_DESCRIPTORS
        by_name: dict[str, RoutingToolDescriptor] = {}
        for descriptor in descriptors:
            if descriptor.tool_name in by_name:
                raise ValueError(f"duplicate routing tool name: {descriptor.tool_name}")
            by_name[descriptor.tool_name] = descriptor
        self._descriptors = by_name
        self._direct_scenarios = direct_scenarios or _DEFAULT_DIRECT_SCENARIOS
        self._enabled_tool_names = enabled_tool_names or frozenset(by_name)

    @classmethod
    def from_settings(cls, settings: Settings) -> CapabilityRoutingRegistry:
        enabled = {"datetime.now", "calculator.evaluate", "daemon.status"}
        if "tool.shell.read" in settings.capabilities:
            enabled.add("tool.shell.read.project")
        system_read = settings.capabilities.get("tool.system.read", {})
        enabled_families = set(system_read.get("enabled_families", ()))
        for descriptor in _DEFAULT_TOOL_DESCRIPTORS:
            if descriptor.system_family is not None and descriptor.system_family in enabled_families:
                enabled.add(descriptor.tool_name)
        return cls(enabled_tool_names=frozenset(enabled))

    @classmethod
    def from_available_tools_summary(
        cls,
        available_tools_summary: tuple[Any, ...],
    ) -> CapabilityRoutingRegistry:
        enabled = frozenset(
            item["tool_name"]
            for item in available_tools_summary
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        )
        return cls(enabled_tool_names=enabled)

    def available_tools_summary(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._descriptors[name].summary()
            for name in sorted(self._enabled_tool_names)
            if name in self._descriptors
        )

    def available_capabilities(self) -> frozenset[Capability]:
        return frozenset(
            descriptor.capability
            for name, descriptor in self._descriptors.items()
            if name in self._enabled_tool_names
        )

    def valid_tool_names(
        self,
        capability: Capability,
        tool_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            tool_name
            for tool_name in tool_names
            if tool_name in self._enabled_tool_names
            and (descriptor := self._descriptors.get(tool_name)) is not None
            and descriptor.capability is capability
        )

    def descriptor(self, tool_name: str) -> RoutingToolDescriptor | None:
        if tool_name not in self._enabled_tool_names:
            return None
        return self._descriptors.get(tool_name)

    def direct_scenarios(self) -> tuple[DirectScenarioDescriptor, ...]:
        return self._direct_scenarios

    def direct_scenario_for_candidate(
        self,
        candidate: CapabilityCandidate,
    ) -> DirectScenarioDescriptor | None:
        valid_names = self.valid_tool_names(candidate.capability, candidate.tool_names)
        for scenario in self._direct_scenarios:
            if scenario.intent_family is not candidate.intent_family:
                continue
            if scenario.scope_hint != candidate.scope_hint:
                continue
            if scenario.capabilities != (candidate.capability,):
                continue
            if scenario.tool_names == valid_names:
                return scenario
        return None


def classification_has_registry_direct_scope(
    classification: IntentClassification,
    available_tools_summary: tuple[Any, ...],
) -> bool:
    registry = CapabilityRoutingRegistry.from_available_tools_summary(available_tools_summary)
    scenarios = _scenario_set_for_classification(classification, registry)
    return scenarios is not None


def _scenario_set_for_classification(
    classification: IntentClassification,
    registry: CapabilityRoutingRegistry,
) -> tuple[DirectScenarioDescriptor, ...] | None:
    scenarios = tuple(
        scenario
        for candidate in classification.candidate_capabilities
        if (scenario := registry.direct_scenario_for_candidate(candidate)) is not None
    )
    if not scenarios or len(scenarios) != len(classification.candidate_capabilities):
        return None
    if len(scenarios) == 1:
        return scenarios
    if {scenario.scenario for scenario in scenarios} == {"cpu_overview"} and {
        tool_name
        for scenario in scenarios
        for tool_name in scenario.tool_names
    } == {"tool.system.read.hardware", "tool.system.read.resources"}:
        return scenarios
    return None


_DEFAULT_TOOL_DESCRIPTORS: tuple[RoutingToolDescriptor, ...] = (
    RoutingToolDescriptor(
        tool_name="datetime.now",
        capability=Capability.TOOL_SAFE,
        intent_families=frozenset({IntentFamily.SAFE_BUILTIN_TOOL}),
        description="current local date and time",
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.SAFE}),
        sensitivity_ceiling=Sensitivity.PROJECT,
    ),
    RoutingToolDescriptor(
        tool_name="calculator.evaluate",
        capability=Capability.TOOL_SAFE,
        intent_families=frozenset({IntentFamily.SAFE_BUILTIN_TOOL}),
        description="deterministic arithmetic evaluation",
        requires_live_state=False,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.SAFE}),
        sensitivity_ceiling=Sensitivity.PUBLIC,
    ),
    RoutingToolDescriptor(
        tool_name="daemon.status",
        capability=Capability.TOOL_SAFE,
        intent_families=frozenset({IntentFamily.SAFE_BUILTIN_TOOL, IntentFamily.SYSTEM_DIAGNOSTICS}),
        description="assistant daemon runtime status",
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.SAFE}),
        sensitivity_ceiling=Sensitivity.PUBLIC,
    ),
    RoutingToolDescriptor(
        tool_name="tool.shell.read.project",
        capability=Capability.TOOL_SHELL_READ,
        intent_families=frozenset({IntentFamily.PROJECT_INSPECTION}),
        description="allowlisted read-only project inspection commands",
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        sensitivity_ceiling=Sensitivity.PROJECT,
    ),
    RoutingToolDescriptor(
        tool_name="tool.system.read.process",
        capability=Capability.TOOL_SYSTEM_READ_PROCESS,
        intent_families=frozenset({IntentFamily.SYSTEM_DIAGNOSTICS}),
        description="read process list and process status",
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        sensitivity_ceiling=Sensitivity.INFRA,
        system_family="process",
    ),
    RoutingToolDescriptor(
        tool_name="tool.system.read.resources",
        capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
        intent_families=frozenset({IntentFamily.SYSTEM_DIAGNOSTICS}),
        description="read CPU, memory and resource usage",
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        sensitivity_ceiling=Sensitivity.INFRA,
        system_family="resources",
    ),
    RoutingToolDescriptor(
        tool_name="tool.system.read.hardware",
        capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
        intent_families=frozenset({IntentFamily.SYSTEM_DIAGNOSTICS}),
        description="read hardware and operating system metadata",
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        sensitivity_ceiling=Sensitivity.INFRA,
        system_family="hardware",
    ),
    RoutingToolDescriptor(
        tool_name="tool.system.read.network",
        capability=Capability.TOOL_SYSTEM_READ_NETWORK,
        intent_families=frozenset({IntentFamily.SYSTEM_DIAGNOSTICS}),
        description="read local network sockets and interfaces",
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        sensitivity_ceiling=Sensitivity.INFRA,
        system_family="network",
    ),
    RoutingToolDescriptor(
        tool_name="tool.system.read.sensors",
        capability=Capability.TOOL_SYSTEM_READ_SENSORS,
        intent_families=frozenset({IntentFamily.SYSTEM_DIAGNOSTICS}),
        description="read temperature and thermal sensor state",
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        sensitivity_ceiling=Sensitivity.INFRA,
        system_family="sensors",
    ),
)


_DEFAULT_DIRECT_SCENARIOS: tuple[DirectScenarioDescriptor, ...] = (
    DirectScenarioDescriptor(
        scenario="current_time",
        tool_names=("datetime.now",),
        capabilities=(Capability.TOOL_SAFE,),
        intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
        scope_hint=None,
    ),
    DirectScenarioDescriptor(
        scenario="christmas_countdown",
        tool_names=("datetime.now",),
        capabilities=(Capability.TOOL_SAFE,),
        intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
        scope_hint="christmas_countdown",
    ),
    DirectScenarioDescriptor(
        scenario="sensor_temperature",
        tool_names=("tool.system.read.sensors",),
        capabilities=(Capability.TOOL_SYSTEM_READ_SENSORS,),
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        scope_hint=None,
    ),
    DirectScenarioDescriptor(
        scenario="memory_overview",
        tool_names=("tool.system.read.resources",),
        capabilities=(Capability.TOOL_SYSTEM_READ_RESOURCES,),
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        scope_hint=None,
    ),
    DirectScenarioDescriptor(
        scenario="disk_free",
        tool_names=("tool.system.read.resources",),
        capabilities=(Capability.TOOL_SYSTEM_READ_RESOURCES,),
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        scope_hint="disk_free",
    ),
    DirectScenarioDescriptor(
        scenario="battery_charge",
        tool_names=("tool.system.read.hardware",),
        capabilities=(Capability.TOOL_SYSTEM_READ_HARDWARE,),
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        scope_hint="battery_charge",
    ),
    DirectScenarioDescriptor(
        scenario="os_version",
        tool_names=("tool.system.read.hardware",),
        capabilities=(Capability.TOOL_SYSTEM_READ_HARDWARE,),
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        scope_hint="os_version",
    ),
    DirectScenarioDescriptor(
        scenario="process_name_search",
        tool_names=("tool.system.read.process",),
        capabilities=(Capability.TOOL_SYSTEM_READ_PROCESS,),
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        scope_hint="process_name_search",
        requires_argument_extractor="process_name_search_pattern",
    ),
    DirectScenarioDescriptor(
        scenario="vpn_status",
        tool_names=("tool.system.read.network",),
        capabilities=(Capability.TOOL_SYSTEM_READ_NETWORK,),
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        scope_hint="vpn_status",
    ),
    DirectScenarioDescriptor(
        scenario="cpu_overview",
        tool_names=("tool.system.read.hardware",),
        capabilities=(Capability.TOOL_SYSTEM_READ_HARDWARE,),
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        scope_hint="cpu_overview",
    ),
    DirectScenarioDescriptor(
        scenario="cpu_overview",
        tool_names=("tool.system.read.resources",),
        capabilities=(Capability.TOOL_SYSTEM_READ_RESOURCES,),
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        scope_hint="cpu_overview",
    ),
)
