from __future__ import annotations

from assistant_core.domain.context import ContextAssemblyRequest


def active_namespaces(request: ContextAssemblyRequest) -> list[str]:
    namespaces = [
        "user.preferences",
        "user.working_style",
        "system.runtime_rules",
        "environment.inference_node",
    ]
    if request.active_project_namespace:
        namespaces.insert(2, request.active_project_namespace)
    return namespaces
