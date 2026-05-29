from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SystemDiagnosticsFamily(StrEnum):
    PROCESS = "process"
    RESOURCES = "resources"
    HARDWARE = "hardware"
    NETWORK = "network"
    SENSORS = "sensors"


@dataclass(frozen=True)
class SystemDiagnosticsDecision:
    allowed: bool
    code: str
    reason: str
    family: SystemDiagnosticsFamily | None = None
    argv: tuple[str, ...] = ()
    cwd: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorReading:
    label: str
    value: float
    unit: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_celsius(self) -> SensorReading:
        unit = self.unit.upper()
        if unit == "C":
            return replace(self, unit="C")
        if unit == "F":
            return replace(self, value=(self.value - 32.0) * 5.0 / 9.0, unit="C")
        if unit == "K":
            return replace(self, value=self.value - 273.15, unit="C")
        return self


@dataclass(frozen=True)
class SensorSnapshot:
    source: str
    readings: list[SensorReading] = field(default_factory=list)
    available: bool = True
    reason: str | None = None
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unavailable(cls, *, source: str, reason: str) -> SensorSnapshot:
        return cls(source=source, available=False, reason=reason, readings=[])

    def normalized_celsius(self) -> SensorSnapshot:
        return replace(
            self,
            readings=[reading.normalized_celsius() for reading in self.readings],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "available": self.available,
            "reason": self.reason,
            "captured_at": self.captured_at.isoformat(),
            "readings": [
                {
                    "label": reading.label,
                    "value": reading.value,
                    "unit": reading.unit,
                    "source": reading.source,
                    "metadata": reading.metadata,
                }
                for reading in self.readings
            ],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DiagnosticsOutputLimits:
    max_stdout_bytes: int = 20_000
    max_stderr_bytes: int = 20_000
    max_lines: int = 200
    timeout_seconds: float = 10.0
