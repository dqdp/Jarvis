from __future__ import annotations

import json


def parse_sse_blocks(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.strip().split("\n\n"):
        if not block:
            continue
        event_type = "message"
        data = "{}"
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((event_type, json.loads(data)))
    return events

__all__ = ["parse_sse_blocks"]
