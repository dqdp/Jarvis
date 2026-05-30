from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
import json

from assistant_core.runtime.request_execution import RequestStreamEvent


async def sse_stream(events: AsyncIterator[RequestStreamEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield sse(event.event_type, event.data)


def sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=json_value)}\n\n"


def json_value(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value
