from __future__ import annotations

import json
import re
from typing import Any

_SENSITIVE_VALUE_MARKERS = (
    "-----begin",
    ".crt",
    ".env",
    ".key",
    ".pem",
    ".ssh",
    "api_key",
    "apikey",
    "authorization",
    "authorization:",
    "bearer ",
    "credential",
    "ghp_",
    "github_pat_",
    "id_ed25519",
    "id_rsa",
    "known_hosts",
    "openssh",
    "pat_",
    "password",
    "private_key",
    "private key",
    "prompt",
    "secret",
    "sk-",
    "sk_",
    "token",
)
_AWS_ACCESS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")


def looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _SENSITIVE_VALUE_MARKERS) or bool(
        _AWS_ACCESS_KEY_RE.search(value)
    )


def redact_content(content: str, *, content_type: str) -> str:
    if not looks_sensitive(content):
        return content
    if content_type == "application/json":
        return json.dumps({"redacted": True}, sort_keys=True)
    return "<redacted>"


def redact_structured_content(value: Any, *, key_hint: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            key: redact_structured_content(child, key_hint=str(key))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_structured_content(child, key_hint=key_hint) for child in value]
    if isinstance(value, tuple):
        return tuple(redact_structured_content(child, key_hint=key_hint) for child in value)
    if isinstance(value, str) and (
        (key_hint is not None and looks_sensitive(key_hint)) or looks_sensitive(value)
    ):
        return "<redacted>"
    return value
