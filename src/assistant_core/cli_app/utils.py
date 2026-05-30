from __future__ import annotations

from typing import Any

from assistant_core.cli_app.client import CliUserError


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CliUserError(f"daemon response missing string field: {key}")
    return value


def _display_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()
