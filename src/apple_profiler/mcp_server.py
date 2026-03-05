"""MCP server exposing apple-profiler functionality as tools.

Provides tools for opening .trace files, inspecting metadata, querying
CPU samples, hangs, signpost events/intervals, and computing top functions.

Usage:
    uv run python -m apple_profiler.mcp_server
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from apple_profiler._models import (
    CpuSample,
    Frame,
    Hang,
    SignpostEvent,
    SignpostInterval,
)
from apple_profiler.trace import TraceFile

logger = logging.getLogger(__name__)

mcp = FastMCP("apple_profiler_mcp")

# ── State: open trace files ──

_open_traces: dict[str, TraceFile] = {}


def _get_trace(trace_path: str) -> TraceFile:
    """Get or open a trace file, caching by path."""
    resolved = str(Path(trace_path).resolve())
    if resolved not in _open_traces:
        _open_traces[resolved] = TraceFile(resolved)
    return _open_traces[resolved]


# ── Serialization helpers ──


def _frame_dict(f: Frame) -> dict[str, str | None]:
    return {
        "name": f.name,
        "address": f.address,
        "binary": f.binary_name,
    }


def _sample_dict(s: CpuSample) -> dict[str, object]:
    return {
        "time_ns": s.time_ns,
        "process": f"{s.process.name} ({s.process.pid})",
        "thread": s.thread.name,
        "core": s.core,
        "state": s.state,
        "weight": s.weight,
        "backtrace": [_frame_dict(f) for f in s.backtrace],
    }


def _hang_dict(h: Hang) -> dict[str, object]:
    return {
        "start_ns": h.start_ns,
        "duration_ns": h.duration_ns,
        "duration_ms": round(h.duration_ns / 1_000_000, 1),
        "hang_type": h.hang_type,
        "process": f"{h.process.name} ({h.process.pid})",
        "thread": h.thread.name,
    }


def _signpost_event_dict(e: SignpostEvent) -> dict[str, object]:
    return {
        "time_ns": e.time_ns,
        "event_type": e.event_type,
        "name": e.name,
        "subsystem": e.subsystem,
        "category": e.category,
        "message": e.message,
        "scope": e.scope,
        "identifier": e.identifier,
        "process": f"{e.process.name} ({e.process.pid})" if e.process else None,
        "thread": e.thread.name if e.thread else None,
    }


def _signpost_interval_dict(i: SignpostInterval) -> dict[str, object]:
    return {
        "start_ns": i.start_ns,
        "duration_ns": i.duration_ns,
        "duration_ms": round(i.duration_ns / 1_000_000, 1),
        "name": i.name,
        "subsystem": i.subsystem,
        "category": i.category,
        "identifier": i.identifier,
        "process": f"{i.process.name} ({i.process.pid})" if i.process else None,
        "start_message": i.start_message,
        "end_message": i.end_message,
    }


# ── Input Models ──


class TracePathInput(BaseModel):
    """Input requiring a trace file path."""

    model_config = ConfigDict(str_strip_whitespace=True)
    trace_path: str = Field(..., description="Path to the .trace file", min_length=1)


class CpuSamplesInput(BaseModel):
    """Input for querying CPU samples."""

    model_config = ConfigDict(str_strip_whitespace=True)
    trace_path: str = Field(..., description="Path to the .trace file", min_length=1)
    limit: int | None = Field(
        default=None,
        description="Maximum number of samples to return. None returns all.",
        ge=1,
    )


class TopFunctionsInput(BaseModel):
    """Input for top functions query."""

    model_config = ConfigDict(str_strip_whitespace=True)
    trace_path: str = Field(..., description="Path to the .trace file", min_length=1)
    n: int = Field(
        default=20,
        description="Number of top functions to return",
        ge=1,
        le=500,
    )


class SignpostFilterInput(BaseModel):
    """Input for querying signpost events or intervals with filtering."""

    model_config = ConfigDict(str_strip_whitespace=True)
    trace_path: str = Field(..., description="Path to the .trace file", min_length=1)
    subsystem: str | None = Field(
        default=None, description="Filter by subsystem (e.g., 'com.apple.SkyLight')"
    )
    category: str | None = Field(
        default=None, description="Filter by category (e.g., 'networking')"
    )
    name: str | None = Field(
        default=None, description="Filter by signpost name (e.g., 'NetworkRequest')"
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of results to return. None returns all.",
        ge=1,
    )


# ── Tools ──


@mcp.tool(
    name="profiler_open_trace",
    annotations=ToolAnnotations(
        title="Open Trace File",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_open_trace(params: TracePathInput) -> str:
    """Open a .trace file and return its metadata and available tables.

    This is typically the first tool to call. It parses the trace's table of
    contents and returns device info, recording duration, target process, and
    a list of all available data tables.

    Returns:
        JSON with trace info and available table schemas.
    """
    try:
        t = _get_trace(params.trace_path)
        info = t.info
        tables = t.tables()
        return json.dumps(
            {
                "info": {
                    "device_name": info.device_name,
                    "device_model": info.device_model,
                    "os_version": info.os_version,
                    "platform": info.platform,
                    "start_date": info.start_date,
                    "end_date": info.end_date,
                    "duration_seconds": info.duration_seconds,
                    "instruments_version": info.instruments_version,
                    "template_name": info.template_name,
                    "recording_mode": info.recording_mode,
                    "end_reason": info.end_reason,
                    "target_process": info.target_process,
                    "target_pid": info.target_pid,
                },
                "tables": [{"schema": tb.schema, "attributes": tb.attributes} for tb in tables],
                "processes": [{"pid": p.pid, "name": p.name} for p in t.processes()],
            },
            indent=2,
        )
    except Exception as e:
        return f"Error opening trace: {e}"


@mcp.tool(
    name="profiler_cpu_samples",
    annotations=ToolAnnotations(
        title="Get CPU Profile Samples",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_cpu_samples(params: CpuSamplesInput) -> str:
    """Get CPU profile samples from a trace.

    Each sample includes timestamp, thread, process, CPU core, thread state,
    cycle weight, and full backtrace with symbol names and binary info.

    Returns:
        JSON array of CPU samples with backtraces.
    """
    try:
        t = _get_trace(params.trace_path)
        if not t.has_table("cpu-profile"):
            return "Error: This trace does not contain CPU profile data."
        samples = t.cpu_samples()
        if params.limit is not None:
            samples = samples[: params.limit]
        return json.dumps(
            {
                "total_samples": len(t.cpu_samples()),
                "returned": len(samples),
                "samples": [_sample_dict(s) for s in samples],
            },
            indent=2,
        )
    except Exception as e:
        return f"Error reading CPU samples: {e}"


@mcp.tool(
    name="profiler_top_functions",
    annotations=ToolAnnotations(
        title="Top Functions by Cycle Weight",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_top_functions(params: TopFunctionsInput) -> str:
    """Get the top N functions by total CPU cycle weight.

    Aggregates cycle weights across all samples for each function symbol,
    returning the heaviest functions first. Useful for identifying CPU hotspots.

    Returns:
        JSON array of {function, total_weight} sorted by weight descending.
    """
    try:
        t = _get_trace(params.trace_path)
        if not t.has_table("cpu-profile"):
            return "Error: This trace does not contain CPU profile data."
        top = t.top_functions(params.n)
        return json.dumps(
            {
                "top_functions": [
                    {"function": name, "total_weight": weight} for name, weight in top
                ],
            },
            indent=2,
        )
    except Exception as e:
        return f"Error computing top functions: {e}"


@mcp.tool(
    name="profiler_hangs",
    annotations=ToolAnnotations(
        title="Get Detected Hangs",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_hangs(params: TracePathInput) -> str:
    """Get all detected hangs/unresponsiveness intervals from a trace.

    Returns hang type (Severe Hang, Hang), duration, affected process/thread.
    Requires the trace to have been recorded with the Hangs instrument.

    Returns:
        JSON array of hang intervals.
    """
    try:
        t = _get_trace(params.trace_path)
        if not t.has_table("potential-hangs"):
            return "Error: This trace does not contain hang detection data."
        hangs = t.hangs()
        return json.dumps(
            {
                "total_hangs": len(hangs),
                "hangs": [_hang_dict(h) for h in hangs],
            },
            indent=2,
        )
    except Exception as e:
        return f"Error reading hangs: {e}"


@mcp.tool(
    name="profiler_signpost_events",
    annotations=ToolAnnotations(
        title="Get Signpost Events",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_signpost_events(params: SignpostFilterInput) -> str:
    """Get os-signpost events with optional filtering.

    Returns raw Begin/End/Event signpost entries. Filter by subsystem,
    category, or signpost name to focus on specific instrumentation.

    Returns:
        JSON array of signpost events.
    """
    try:
        t = _get_trace(params.trace_path)
        if not t.has_table("os-signpost"):
            return "Error: This trace does not contain signpost data."
        events = t.signpost_events(
            subsystem=params.subsystem,
            category=params.category,
            name=params.name,
        )
        total = len(events)
        if params.limit is not None:
            events = events[: params.limit]
        return json.dumps(
            {
                "total_events": total,
                "returned": len(events),
                "events": [_signpost_event_dict(e) for e in events],
            },
            indent=2,
        )
    except Exception as e:
        return f"Error reading signpost events: {e}"


@mcp.tool(
    name="profiler_signpost_intervals",
    annotations=ToolAnnotations(
        title="Get Signpost Intervals",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_signpost_intervals(params: SignpostFilterInput) -> str:
    """Get matched signpost intervals (begin+end pairs) with optional filtering.

    Returns paired intervals with start time, duration, and start/end messages.
    Filter by subsystem, category, or signpost name.

    Returns:
        JSON array of signpost intervals with durations.
    """
    try:
        t = _get_trace(params.trace_path)
        if not t.has_table("os-signpost-interval"):
            return "Error: This trace does not contain signpost interval data."
        intervals = t.signpost_intervals(
            subsystem=params.subsystem,
            category=params.category,
            name=params.name,
        )
        total = len(intervals)
        if params.limit is not None:
            intervals = intervals[: params.limit]
        return json.dumps(
            {
                "total_intervals": total,
                "returned": len(intervals),
                "intervals": [_signpost_interval_dict(i) for i in intervals],
            },
            indent=2,
        )
    except Exception as e:
        return f"Error reading signpost intervals: {e}"


@mcp.tool(
    name="profiler_list_tables",
    annotations=ToolAnnotations(
        title="List Available Tables",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_list_tables(params: TracePathInput) -> str:
    """List all data tables available in the trace.

    Each table has a schema name (e.g., 'cpu-profile', 'potential-hangs',
    'os-signpost') and optional attributes like target-pid or codes.

    Returns:
        JSON array of table schemas and their attributes.
    """
    try:
        t = _get_trace(params.trace_path)
        tables = t.tables()
        return json.dumps(
            {
                "total_tables": len(tables),
                "tables": [{"schema": tb.schema, "attributes": tb.attributes} for tb in tables],
            },
            indent=2,
        )
    except Exception as e:
        return f"Error listing tables: {e}"


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
