from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.policy import Capability
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.runtime.routing import (
    CapabilityRoutingRegistry,
    IntentFamily,
    RoutingToolDescriptor,
)
from assistant_core.tools.builtin import (
    calendar_diff_tool,
    calculator_tool,
    daemon_status_tool,
    datetime_diff_tool,
    datetime_now_tool,
    datetime_until_tool,
)
from assistant_core.tools.registry import ToolRegistry


pytestmark = pytest.mark.unit


def test_capability_routing_registry_lists_enabled_tools_from_settings() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    registry = CapabilityRoutingRegistry.from_settings(settings)

    tool_names = {item["tool_name"] for item in registry.available_tools_summary()}
    assert "calendar.diff" in tool_names
    assert "datetime.diff" in tool_names
    assert "datetime.now" in tool_names
    assert "tool.shell.read.project" in tool_names
    assert "tool.system.read.hardware" in tool_names
    assert "tool.system.read.resources" in tool_names
    assert "tool.system.read.network" in tool_names
    assert Capability.TOOL_SYSTEM_READ_HARDWARE in registry.available_capabilities()


def test_capability_routing_registry_respects_disabled_system_families() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    capabilities = {
        **settings.capabilities,
        "tool.system.read": {
            **settings.capabilities["tool.system.read"],
            "enabled_families": ["resources"],
        },
    }

    registry = CapabilityRoutingRegistry.from_settings(replace(settings, capabilities=capabilities))

    tool_names = {item["tool_name"] for item in registry.available_tools_summary()}
    assert "tool.system.read.resources" in tool_names
    assert "tool.system.read.hardware" not in tool_names
    assert Capability.TOOL_SYSTEM_READ_RESOURCES in registry.available_capabilities()
    assert Capability.TOOL_SYSTEM_READ_HARDWARE not in registry.available_capabilities()


def test_capability_routing_registry_rejects_duplicate_tool_names() -> None:
    descriptor = RoutingToolDescriptor(
        tool_name="datetime.now",
        capability=Capability.TOOL_SAFE,
        intent_families=frozenset({IntentFamily.SAFE_BUILTIN_TOOL}),
        description="duplicate",
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset(),
        sensitivity_ceiling=Sensitivity.PROJECT,
    )

    with pytest.raises(ValueError, match="duplicate routing tool name"):
        CapabilityRoutingRegistry(descriptors=(descriptor, descriptor))


def test_capability_routing_registry_validates_tool_names_for_capability() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    registry = CapabilityRoutingRegistry.from_settings(settings)

    assert registry.valid_tool_names(Capability.TOOL_SYSTEM_READ_NETWORK, ("tool.system.read.network",)) == (
        "tool.system.read.network",
    )
    assert registry.valid_tool_names(Capability.TOOL_SYSTEM_READ_NETWORK, ("tool.system.read.hardware",)) == ()


def test_app_factory_request_plan_guard_rejects_missing_gateway_tool() -> None:
    from assistant_core.app_factory import _validate_request_plan_tool_surface

    settings = replace(ConfigLoader(Path("config")).load("test"), capabilities={})
    registry = ToolRegistry([calendar_diff_tool(), calculator_tool(), daemon_status_tool(), datetime_diff_tool()])

    with pytest.raises(RuntimeError, match="request-plan tool is not registered.*datetime.now"):
        _validate_request_plan_tool_surface(settings, registry)


def test_app_factory_request_plan_guard_rejects_gateway_metadata_drift() -> None:
    from assistant_core.app_factory import _validate_request_plan_tool_surface

    settings = replace(ConfigLoader(Path("config")).load("test"), capabilities={})
    drifted_datetime = datetime_now_tool()
    drifted_datetime.spec = replace(
        drifted_datetime.spec,
        capability=Capability.TOOL_SHELL_READ,
    )
    registry = ToolRegistry(
        [
            drifted_datetime,
            calendar_diff_tool(),
            datetime_diff_tool(),
            datetime_until_tool(),
            calculator_tool(),
            daemon_status_tool(),
        ]
    )

    with pytest.raises(RuntimeError, match="request-plan tool metadata differs.*datetime.now"):
        _validate_request_plan_tool_surface(settings, registry)
