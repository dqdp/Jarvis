from __future__ import annotations


def allowed_tool_catalog(
    summaries: tuple[dict[str, str], ...],
    *,
    allowed_tool_names: frozenset[str],
) -> list[str]:
    catalog: list[str] = []
    seen: set[str] = set()
    for item in summaries:
        tool_name = item.get("tool_name")
        description = item.get("description")
        if (
            not isinstance(tool_name, str)
            or tool_name not in allowed_tool_names
            or tool_name in seen
            or not isinstance(description, str)
        ):
            continue
        clean_description = " ".join(description.split())[:240]
        if not clean_description:
            continue
        seen.add(tool_name)
        catalog.append(f"{tool_name}: {clean_description}.")
    for tool_name in sorted(allowed_tool_names - seen):
        catalog.append(f"{tool_name}.")
    return catalog


__all__ = ["allowed_tool_catalog"]
