from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import sys
from typing import Any


class InProcessDarwinResourceProvider:
    def __init__(self, *, sample_interval_seconds: float = 0.1) -> None:
        self._sample_interval_seconds = sample_interval_seconds

    async def snapshot_cpu_and_memory(self) -> dict[str, Any] | None:
        if _normalize_platform(sys.platform) != "darwin":
            return None
        try:
            before = _darwin_cpu_ticks()
            await asyncio.sleep(self._sample_interval_seconds)
            after = _darwin_cpu_ticks()
            memory = _darwin_memory_snapshot()
        except (OSError, AttributeError, ValueError):
            return None
        cpu = _cpu_percentages(before, after)
        if cpu is None:
            return None
        return {
            "cpu": cpu,
            "memory": memory,
            "source": "mach",
        }


def _normalize_platform(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith("darwin"):
        return "darwin"
    if lowered.startswith("linux"):
        return "linux"
    return lowered


def _darwin_libc() -> ctypes.CDLL:
    return ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib")


def _darwin_cpu_ticks() -> tuple[int, int, int, int]:
    libc = _darwin_libc()
    libc.mach_host_self.restype = ctypes.c_uint
    host = libc.mach_host_self()
    cpu_load_info = 2
    cpu_info = ctypes.POINTER(ctypes.c_int)()
    cpu_count = ctypes.c_uint()
    info_count = ctypes.c_uint()
    status = libc.host_processor_info(
        ctypes.c_uint(host),
        ctypes.c_int(cpu_load_info),
        ctypes.byref(cpu_count),
        ctypes.byref(cpu_info),
        ctypes.byref(info_count),
    )
    if status != 0 or cpu_count.value <= 0 or info_count.value < cpu_count.value * 4:
        raise OSError(status)
    ticks = [0, 0, 0, 0]
    try:
        for cpu_index in range(cpu_count.value):
            offset = cpu_index * 4
            for state_index in range(4):
                ticks[state_index] += int(cpu_info[offset + state_index])
    finally:
        _darwin_vm_deallocate(libc, cpu_info, info_count.value * ctypes.sizeof(ctypes.c_int))
    return tuple(ticks)


def _darwin_vm_deallocate(
    libc: ctypes.CDLL,
    pointer: ctypes.POINTER(ctypes.c_int),
    byte_count: int,
) -> None:
    try:
        task_self = ctypes.c_uint.in_dll(libc, "mach_task_self_").value
        libc.vm_deallocate(ctypes.c_uint(task_self), ctypes.cast(pointer, ctypes.c_void_p), byte_count)
    except (AttributeError, ValueError, OSError):
        return


def _cpu_percentages(
    before: tuple[int, int, int, int],
    after: tuple[int, int, int, int],
) -> dict[str, float] | None:
    deltas = [max(0, current - previous) for previous, current in zip(before, after, strict=True)]
    total = sum(deltas)
    if total <= 0:
        return None
    user_percent = round((deltas[0] / total) * 100, 2)
    system_percent = round((deltas[1] / total) * 100, 2)
    idle_percent = round((deltas[2] / total) * 100, 2)
    nice_percent = round((deltas[3] / total) * 100, 2)
    return {
        "user_percent": user_percent,
        "system_percent": system_percent,
        "idle_percent": idle_percent,
        "nice_percent": nice_percent,
        "used_percent": round(max(0.0, 100.0 - idle_percent), 2),
    }


class _DarwinVMStats64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint),
        ("active_count", ctypes.c_uint),
        ("inactive_count", ctypes.c_uint),
        ("wire_count", ctypes.c_uint),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint),
        ("speculative_count", ctypes.c_uint),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint),
        ("throttled_count", ctypes.c_uint),
        ("external_page_count", ctypes.c_uint),
        ("internal_page_count", ctypes.c_uint),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


def _darwin_memory_snapshot() -> dict[str, Any]:
    libc = _darwin_libc()
    libc.mach_host_self.restype = ctypes.c_uint
    host = libc.mach_host_self()
    page_size = ctypes.c_uint()
    if libc.host_page_size(ctypes.c_uint(host), ctypes.byref(page_size)) != 0:
        raise OSError("host_page_size failed")
    stats = _DarwinVMStats64()
    count = ctypes.c_uint(ctypes.sizeof(_DarwinVMStats64) // ctypes.sizeof(ctypes.c_int))
    host_vm_info64 = 4
    status = libc.host_statistics64(
        ctypes.c_uint(host),
        ctypes.c_int(host_vm_info64),
        ctypes.byref(stats),
        ctypes.byref(count),
    )
    if status != 0:
        raise OSError(status)
    total_bytes = _darwin_total_memory_bytes(libc)
    available_pages = int(stats.free_count) + int(stats.speculative_count)
    available_bytes = min(total_bytes, available_pages * int(page_size.value))
    used_bytes = max(0, total_bytes - available_bytes)
    return {
        "total": _format_bytes(total_bytes),
        "used": _format_bytes(used_bytes),
        "available": _format_bytes(available_bytes),
        "used_percent": round((used_bytes / total_bytes) * 100, 2) if total_bytes else None,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "available_bytes": available_bytes,
        "page_size_bytes": int(page_size.value),
        "source": "mach",
    }


def _format_bytes(value: int) -> str:
    gib = value / (1024**3)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{value / (1024**2):.0f} MiB"


def _darwin_total_memory_bytes(libc: ctypes.CDLL) -> int:
    value = ctypes.c_uint64(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    sysctlbyname = libc.sysctlbyname
    sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    sysctlbyname.restype = ctypes.c_int
    status = sysctlbyname(b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0)
    if status != 0 or value.value <= 0:
        raise OSError(status)
    return int(value.value)
