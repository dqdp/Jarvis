from __future__ import annotations

from typing import TextIO

from assistant_core.cli_app.client import JarvisClient
from assistant_core.cli_app.role_rendering import write_role_line
from assistant_core.cli_app.terminal_rendering import TerminalColorScheme
from assistant_core.cli_app.utils import _display_text


async def write_status(
    *,
    client: JarvisClient,
    stdout: TextIO,
    color_scheme: TerminalColorScheme | None = None,
) -> dict:
    payload = await client.health()
    write_role_line(
        stdout,
        color_scheme=color_scheme,
        role="status",
        text=f"status> {_display_text(payload.get('status'))}",
    )
    readiness = payload.get("readiness")
    if not isinstance(readiness, dict):
        return payload
    reasons = readiness.get("reasons")
    if not isinstance(reasons, dict):
        return payload
    for component, reason in sorted(reasons.items()):
        write_role_line(
            stdout,
            color_scheme=color_scheme,
            role="status",
            text=f"reason> {_display_text(component)}: {_display_text(reason)}",
        )
    return payload


async def write_model_status(
    *,
    client: JarvisClient,
    stdout: TextIO,
    color_scheme: TerminalColorScheme | None = None,
) -> dict:
    payload = await client.runtime_status()
    profile_name = _display_text(payload.get("default_model_profile"))
    profiles = payload.get("model_profiles", {})
    profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
    provider = _display_text(profile.get("provider")) if isinstance(profile, dict) else ""
    model = _display_text(profile.get("model")) if isinstance(profile, dict) else ""
    max_output_tokens = profile.get("max_output_tokens") if isinstance(profile, dict) else None
    temperature = profile.get("temperature") if isinstance(profile, dict) else None
    line = f"model> {profile_name} {provider} {model}"
    if max_output_tokens is not None:
        line += f" max_output_tokens={max_output_tokens}"
    if temperature is not None:
        line += f" temperature={temperature}"
    write_role_line(stdout, color_scheme=color_scheme, role="status", text=line)
    return payload


__all__ = ["write_model_status", "write_status"]
