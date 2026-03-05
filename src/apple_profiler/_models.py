"""Data models for parsed xctrace data."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraceInfo:
    """Metadata about a trace recording."""

    device_name: str = ""
    device_model: str = ""
    os_version: str = ""
    platform: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_seconds: float = 0.0
    instruments_version: str = ""
    template_name: str = ""
    recording_mode: str = ""
    end_reason: str = ""
    target_process: str | None = None
    target_pid: int | None = None


@dataclass
class Process:
    """A process captured in the trace."""

    pid: int
    name: str

    def __hash__(self) -> int:
        return hash((self.pid, self.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Process):
            return NotImplemented
        return self.pid == other.pid and self.name == other.name


@dataclass
class Thread:
    """A thread captured in the trace."""

    tid: int
    name: str  # fmt string like "Main Thread 0x644a1c3 (Finder, pid: 751)"
    process: Process


@dataclass
class Frame:
    """A single stack frame."""

    name: str
    address: str
    binary_name: str | None = None
    binary_path: str | None = None
    binary_uuid: str | None = None


def _empty_frame_list() -> list[Frame]:
    return []


def _empty_str_dict() -> dict[str, str]:
    return {}


@dataclass
class CpuSample:
    """A single CPU profile sample."""

    time_ns: int
    thread: Thread
    process: Process
    core: str
    state: str
    weight: int
    backtrace: list[Frame] = field(default_factory=_empty_frame_list)


@dataclass
class Hang:
    """A detected hang/unresponsiveness interval."""

    start_ns: int
    duration_ns: int
    hang_type: str
    thread: Thread
    process: Process


@dataclass
class SignpostEvent:
    """A raw os-signpost event (Begin, End, or Event)."""

    time_ns: int
    event_type: str
    name: str
    subsystem: str
    category: str
    message: str
    scope: str | None = None
    identifier: int = 0
    thread: Thread | None = None
    process: Process | None = None


@dataclass
class SignpostInterval:
    """A matched signpost interval (begin + end pair)."""

    start_ns: int
    duration_ns: int
    name: str
    subsystem: str
    category: str
    identifier: int
    process: Process | None = None
    start_thread: Thread | None = None
    end_thread: Thread | None = None
    start_message: str = ""
    end_message: str = ""


@dataclass
class TableInfo:
    """Information about a data table in the trace."""

    schema: str
    attributes: dict[str, str] = field(default_factory=_empty_str_dict)
