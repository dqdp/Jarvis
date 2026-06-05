from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from assistant_core.api.presenters import request_payload
from assistant_core.domain.requests import RequestStatus


pytestmark = pytest.mark.unit


def test_request_payload_redacts_internal_tool_summaries_from_metadata() -> None:
    now = datetime.now(UTC)
    request = SimpleNamespace(
        request_id="request-1",
        conversation_id="conversation-1",
        user_message_id="user-message-1",
        assistant_message_id=None,
        status=RequestStatus.RUNNING,
        created_at=now,
        started_at=now,
        completed_at=None,
        error_code=None,
        error_message=None,
        metadata={
            "agent_allowed_tool_names": ["calculator.evaluate"],
            "agent_allowed_tool_summaries": [
                {
                    "tool_name": "calculator.evaluate",
                    "description": "deterministic arithmetic evaluation",
                }
            ],
            "working_directory": "/Users/alex/Jarvis",
        },
    )

    payload = request_payload(request)

    assert payload["metadata"]["agent_allowed_tool_names"] == ["calculator.evaluate"]
    assert payload["metadata"]["working_directory"] == "<redacted>"
    assert "agent_allowed_tool_summaries" not in payload["metadata"]
