from __future__ import annotations

import pytest

from assistant_core.domain.events import EventType
from assistant_core.runtime.request_streaming import PUBLIC_STREAM_FIELDS, STREAM_REPLAY_EVENT_TYPES


pytestmark = pytest.mark.contract


def test_public_sse_replay_contract_includes_cli_activity_phase_events() -> None:
    assert EventType.LOOP_SELECTION_STARTED.value in STREAM_REPLAY_EVENT_TYPES
    assert EventType.LOOP_SELECTION_COMPLETED.value in STREAM_REPLAY_EVENT_TYPES
    assert EventType.LOOP_SELECTION_FAILED.value in STREAM_REPLAY_EVENT_TYPES
    assert EventType.CONTENT_RETRIEVED.value in STREAM_REPLAY_EVENT_TYPES
    assert EventType.MODEL_REQUEST_CREATED.value in STREAM_REPLAY_EVENT_TYPES
    assert EventType.MODEL_RESPONSE_RECEIVED.value in STREAM_REPLAY_EVENT_TYPES
    assert EventType.ASSISTANT_MESSAGE_CREATED.value in STREAM_REPLAY_EVENT_TYPES
    assert EventType.TOOL_CALL_STARTED.value in STREAM_REPLAY_EVENT_TYPES
    assert EventType.TOOL_CALL_COMPLETED.value in STREAM_REPLAY_EVENT_TYPES

    assert "raw_prompt" not in PUBLIC_STREAM_FIELDS[EventType.LOOP_SELECTION_STARTED.value]
    assert "query" not in PUBLIC_STREAM_FIELDS[EventType.CONTENT_RETRIEVED.value]
    assert "retrieved_content_refs" not in PUBLIC_STREAM_FIELDS[EventType.CONTENT_RETRIEVED.value]
    assert "messages" not in PUBLIC_STREAM_FIELDS[EventType.MODEL_REQUEST_CREATED.value]
    assert "content" not in PUBLIC_STREAM_FIELDS[EventType.ASSISTANT_MESSAGE_CREATED.value]
    assert "arguments" not in PUBLIC_STREAM_FIELDS[EventType.TOOL_CALL_STARTED.value]
    assert "content" not in PUBLIC_STREAM_FIELDS[EventType.TOOL_CALL_COMPLETED.value]
