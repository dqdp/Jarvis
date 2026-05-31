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

    assert "raw_prompt" not in PUBLIC_STREAM_FIELDS[EventType.LOOP_SELECTION_STARTED.value]
    assert "query" not in PUBLIC_STREAM_FIELDS[EventType.CONTENT_RETRIEVED.value]
    assert "retrieved_content_refs" not in PUBLIC_STREAM_FIELDS[EventType.CONTENT_RETRIEVED.value]
