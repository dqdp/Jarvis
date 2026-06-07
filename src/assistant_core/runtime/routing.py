from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from assistant_core.config.settings import Settings
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity


class IntentFamily(StrEnum):
    SAFE_BUILTIN_TOOL = "safe_builtin_tool"
    PROJECT_INSPECTION = "project_inspection"
    SYSTEM_DIAGNOSTICS = "system_diagnostics"


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


class CapabilityRoutingRegistry:
    def __init__(
        self,
        descriptors: tuple[RoutingToolDescriptor, ...] | None = None,
        *,
        enabled_tool_names: frozenset[str] | None = None,
    ) -> None:
        descriptors = descriptors or _DEFAULT_TOOL_DESCRIPTORS
        by_name: dict[str, RoutingToolDescriptor] = {}
        for descriptor in descriptors:
            if descriptor.tool_name in by_name:
                raise ValueError(f"duplicate routing tool name: {descriptor.tool_name}")
            by_name[descriptor.tool_name] = descriptor
        self._descriptors = by_name
        self._enabled_tool_names = enabled_tool_names or frozenset(by_name)

    @classmethod
    def from_settings(cls, settings: Settings) -> CapabilityRoutingRegistry:
        enabled = {
            "calendar.diff",
            "datetime.diff",
            "datetime.now",
            "datetime.until",
            "calculator.evaluate",
            "daemon.status",
        }
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


_DEFAULT_TOOL_DESCRIPTORS: tuple[RoutingToolDescriptor, ...] = (
    RoutingToolDescriptor(
        tool_name="calendar.diff",
        capability=Capability.TOOL_SAFE,
        intent_families=frozenset({IntentFamily.SAFE_BUILTIN_TOOL}),
        description=(
            "Use for differences between two known timezone-aware ISO timestamps, "
            "including calendar units such as months, quarters and decades. Provide "
            "explicit from_iso/to_iso; this tool does not resolve event names or holidays."
        ),
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.SAFE}),
        sensitivity_ceiling=Sensitivity.PROJECT,
    ),
    RoutingToolDescriptor(
        tool_name="datetime.diff",
        capability=Capability.TOOL_SAFE,
        intent_families=frozenset({IntentFamily.SAFE_BUILTIN_TOOL}),
        description=(
            "Use for elapsed time between two known timezone-aware ISO timestamps "
            "in microseconds through weeks. Provide explicit from_iso/to_iso; this "
            "tool does not resolve event names or holidays."
        ),
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.SAFE}),
        sensitivity_ceiling=Sensitivity.PROJECT,
    ),
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
        tool_name="datetime.until",
        capability=Capability.TOOL_SAFE,
        intent_families=frozenset({IntentFamily.SAFE_BUILTIN_TOOL}),
        description=(
            "Use for countdowns to supported calendar targets such as next_new_year. "
            "Omit from_iso to use the tool's current local timestamp, or pass a "
            "timezone-aware from_iso explicitly."
        ),
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
        sensitivity_ceiling=Sensitivity.PROJECT,
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
        description="read CPU load, memory usage and resource utilization",
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
        description="read hardware and operating system metadata, not live CPU load or memory utilization",
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
