"""MCP server exposing apple-profiler functionality as tools.

Provides tools for opening .trace files, inspecting metadata, querying
CPU samples, hangs, signpost events/intervals, and computing top functions.

Also provides GPU trace analysis tools that delegate to tools/*.py scripts
via subprocess (since they require DYLD_FRAMEWORK_PATH for Apple private
GPU frameworks).

Usage:
    uv run python -m apple_profiler.mcp_server
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import sys
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
    start_time_ns: int | None = Field(
        default=None,
        description="Include only samples at or after this timestamp (nanoseconds).",
    )
    end_time_ns: int | None = Field(
        default=None,
        description="Include only samples at or before this timestamp (nanoseconds).",
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
    start_time_ns: int | None = Field(
        default=None,
        description="Include only samples at or after this timestamp (nanoseconds).",
    )
    end_time_ns: int | None = Field(
        default=None,
        description="Include only samples at or before this timestamp (nanoseconds).",
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


class TableQueryInput(BaseModel):
    """Input for querying any table by schema name."""

    model_config = ConfigDict(str_strip_whitespace=True)
    trace_path: str = Field(..., description="Path to the .trace file", min_length=1)
    schema: str = Field(
        ...,
        description=(
            "The table schema name to query (e.g., 'metal-gpu-intervals', "
            "'metal-driver-intervals', 'metal-application-command-buffer-submissions'). "
            "Use profiler_list_tables to discover available schemas."
        ),
        min_length=1,
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of rows to return. None returns all.",
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
        logger.exception("Error opening trace: %s", params.trace_path)
        return f"Error opening trace: {e}"


@mcp.tool(
    name="profiler_cpu_samples",
    annotations=ToolAnnotations(
        title="Get CPU Samples with Stack Traces",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_cpu_samples(params: CpuSamplesInput) -> str:
    """Get CPU profile samples, each with a full stack trace (backtrace).

    Each sample includes timestamp, thread, process, CPU core, thread state,
    cycle weight, and full backtrace with symbol names and binary info.
    Works with traces from CPU Profiler, Time Profiler, and Metal System Trace.
    Optionally filter to a time range using start_time_ns / end_time_ns.

    Returns:
        JSON array of CPU samples with backtraces.
    """
    try:
        t = _get_trace(params.trace_path)
        if not t.has_cpu_samples():
            return "Error: This trace does not contain CPU profile data."
        all_samples = t.cpu_samples(
            start_ns=params.start_time_ns,
            end_ns=params.end_time_ns,
        )
        total = len(all_samples)
        samples = all_samples
        if params.limit is not None:
            samples = samples[: params.limit]
        return json.dumps(
            {
                "total_samples": total,
                "returned": len(samples),
                "samples": [_sample_dict(s) for s in samples],
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Error reading CPU samples: %s", params.trace_path)
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
    Works with traces from CPU Profiler, Time Profiler, and Metal System Trace.
    Optionally scope to a time range using start_time_ns / end_time_ns.

    Returns:
        JSON array of {function, total_weight} sorted by weight descending.
    """
    try:
        t = _get_trace(params.trace_path)
        if not t.has_cpu_samples():
            return "Error: This trace does not contain CPU profile data."
        top = t.top_functions(
            params.n,
            start_ns=params.start_time_ns,
            end_ns=params.end_time_ns,
        )
        return json.dumps(
            {
                "top_functions": [
                    {"function": name, "total_weight": weight} for name, weight in top
                ],
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Error computing top functions: %s", params.trace_path)
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
        logger.exception("Error reading hangs: %s", params.trace_path)
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
        logger.exception("Error reading signpost events: %s", params.trace_path)
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
        logger.exception("Error reading signpost intervals: %s", params.trace_path)
        return f"Error reading signpost intervals: {e}"


@mcp.tool(
    name="profiler_query_table",
    annotations=ToolAnnotations(
        title="Query Any Table by Schema",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_query_table(params: TableQueryInput) -> str:
    """Query any data table by schema name, returning rows as JSON.

    This is a generic tool for accessing any table in the trace, including
    Metal GPU tables (metal-gpu-intervals, metal-driver-intervals,
    metal-application-command-buffer-submissions, etc.), thermal data,
    or any other instrument data.

    Use profiler_list_tables first to discover available schemas.
    Each row is returned as a dict mapping column mnemonics to values.

    Returns:
        JSON with schema info, column definitions, and row data.
    """
    try:
        t = _get_trace(params.trace_path)
        if not t.has_table(params.schema):
            available = [tb.schema for tb in t.tables()]
            return json.dumps(
                {
                    "error": f"Table '{params.schema}' not found in trace.",
                    "available_schemas": available,
                },
                indent=2,
            )
        table = t.load_table(params.schema)
        col_index = {col.mnemonic: i for i, col in enumerate(table.columns)}

        rows = table.rows
        total_rows = len(rows)
        if params.limit is not None:
            rows = rows[: params.limit]

        def _serialize_row(row: list) -> dict[str, str]:
            result: dict[str, str] = {}
            for mnemonic, idx in col_index.items():
                if idx < len(row):
                    elem = row[idx]
                    result[mnemonic] = elem.value
            return result

        return json.dumps(
            {
                "schema": params.schema,
                "columns": [
                    {"mnemonic": col.mnemonic, "name": col.name, "type": col.engineering_type}
                    for col in table.columns
                ],
                "total_rows": total_rows,
                "returned": len(rows),
                "rows": [_serialize_row(r) for r in rows],
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Error querying table '%s': %s", params.schema, params.trace_path)
        return f"Error querying table '{params.schema}': {e}"


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
        logger.exception("Error listing tables: %s", params.trace_path)
        return f"Error listing tables: {e}"


# ── GPU Trace Tools ──
# These delegate to tools/*.py as subprocesses with DYLD_FRAMEWORK_PATH set,
# since those scripts require Apple private GPU frameworks loaded at startup.

_TOOLS_DIR = Path(__file__).resolve().parent / "tools"
_DYLD_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"

# Cache for parsed GPU trace data (avoids re-running subprocess)
_gpu_trace_cache: dict[str, dict] = {}
_gpu_counter_cache: dict[str, dict] = {}


async def _run_gpu_tool(
    script: str,
    args: list[str],
    *,
    timeout: int = 120,
    cache_key: str | None = None,
    cache_store: dict[str, dict] | None = None,
) -> dict:
    """Run a tools/*.py script with DYLD_FRAMEWORK_PATH, return parsed JSON.

    Args:
        script: Script filename in tools/ directory.
        args: CLI arguments (trace path, flags, etc.). --json is appended.
        timeout: Subprocess timeout in seconds.
        cache_key: If set, check/store result in cache_store.
        cache_store: Dict to use for caching (e.g., _gpu_trace_cache).

    Returns:
        Parsed JSON dict from subprocess stdout.

    Raises:
        FileNotFoundError: If script doesn't exist.
        RuntimeError: If subprocess fails or returns invalid JSON.
        TimeoutError: If subprocess exceeds timeout.
    """
    if cache_key and cache_store is not None and cache_key in cache_store:
        return cache_store[cache_key]

    script_path = _TOOLS_DIR / script
    if not script_path.exists():
        raise FileNotFoundError(f"GPU tool script not found: {script_path}")

    env = {**os.environ, "DYLD_FRAMEWORK_PATH": _DYLD_FW}
    cmd = [sys.executable, str(script_path), *args]
    if "--json" not in args and "--counters-json" not in args:
        cmd.append("--json")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise TimeoutError(
            f"{script} timed out after {timeout}s"
        ) from None

    if proc.returncode != 0:
        raise RuntimeError(f"{script} failed (rc={proc.returncode}): {stderr.decode()}")

    try:
        result = json.loads(stdout.decode())
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{script} returned invalid JSON: {e}\nstdout: {stdout.decode()[:500]}"
        ) from e

    if cache_key and cache_store is not None:
        cache_store[cache_key] = result

    return result


# ── GPU Input Models ──


class GpuTracePathInput(BaseModel):
    """Input requiring a .gputrace file path."""

    model_config = ConfigDict(str_strip_whitespace=True)
    gputrace_path: str = Field(
        ..., description="Path to .gputrace file", min_length=1,
    )


class GpuTimelineInput(BaseModel):
    """Input for GPU timeline queries with filtering."""

    model_config = ConfigDict(str_strip_whitespace=True)
    gputrace_path: str = Field(
        ..., description="Path to .gputrace file", min_length=1,
    )
    kernel_filter: str | None = Field(
        None, description="Glob pattern to filter by kernel name (e.g., 'lu_factor*')",
    )
    cb_filter: int | None = Field(
        None, description="Filter to specific command buffer index",
    )
    encoder_filter: int | None = Field(
        None, description="Filter to specific encoder index",
    )
    limit: int | None = Field(
        None, description="Max events to return", ge=1,
    )
    offset: int = Field(
        0, description="Skip first N events", ge=0,
    )


class GpuDepsInput(BaseModel):
    """Input for GPU dependency graph analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)
    gputrace_path: str = Field(
        ..., description="Path to .gputrace file", min_length=1,
    )
    scale: str = Field(
        "encoder",
        description="Graph scale: dispatch, encoder, kernel, or cb",
    )
    kernel_filter: str | None = Field(
        None, description="Glob pattern to filter kernels",
    )
    cb_filter: int | None = Field(
        None, description="Filter to command buffer index",
    )
    encoder_filter: int | None = Field(
        None, description="Filter to encoder index",
    )


class GpuCountersInput(BaseModel):
    """Input for GPU performance counter queries."""

    model_config = ConfigDict(str_strip_whitespace=True)
    gputrace_path: str = Field(
        ..., description="Path to .gputrace file", min_length=1,
    )
    counter_filter: list[str] | None = Field(
        None, description="Specific counter names to return (default: all)",
    )
    summary: bool = Field(
        True,
        description="Return summary stats (min/max/avg) instead of full time-series",
    )


class GpuExportInput(BaseModel):
    """Input for GPU trace Perfetto export."""

    model_config = ConfigDict(str_strip_whitespace=True)
    gputrace_path: str = Field(
        ..., description="Path to .gputrace file", min_length=1,
    )
    output_path: str | None = Field(
        None, description="Output .pftrace path (default: auto-generated)",
    )
    group_by: str = Field(
        "pipeline",
        description="Track grouping: 'pipeline' groups by kernel name, "
                    "'cb' groups by command buffer/encoder",
    )
    include_counters: bool = Field(
        True, description="Include GPU performance counter tracks",
    )


# ── GPU Tools ──


@mcp.tool(
    name="profiler_gpu_open",
    annotations=ToolAnnotations(
        title="Open GPU Trace",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_gpu_open(params: GpuTracePathInput) -> str:
    """Open a .gputrace and return structural overview.

    Returns metadata, kernel list, command buffer/encoder structure,
    and dispatch counts. Use this first to understand the GPU workload
    before diving into dependencies or counters.
    """
    try:
        resolved = str(Path(params.gputrace_path).resolve())
        data = await _run_gpu_tool(
            "gputrace_timeline.py",
            [params.gputrace_path],
            cache_key=resolved,
            cache_store=_gpu_trace_cache,
        )
        return json.dumps(
            {
                "metadata": data["metadata"],
                "total_functions": data["total_functions"],
                "kernels": data["kernels"],
                "pipelines": data.get("pipelines", {}),
                "num_dispatches": len(
                    [e for e in data["events"] if e["type"] == "dispatch"]
                ),
                "num_barriers": len(
                    [e for e in data["events"] if e["type"] == "barrier"]
                ),
                "command_buffers": [
                    {
                        "index": i,
                        "addr": cb.get("addr", ""),
                        "num_dispatches": len(cb["dispatches"]),
                    }
                    for i, cb in enumerate(data["command_buffers"])
                ],
                "compute_encoders": [
                    {
                        "encoder_idx": enc["encoder_idx"],
                        "cb_idx": enc["command_buffer_idx"],
                        "num_dispatches": len(enc["dispatches"]),
                    }
                    for enc in data["compute_encoders"]
                ],
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Error opening GPU trace: %s", params.gputrace_path)
        return f"Error opening GPU trace: {e}"


@mcp.tool(
    name="profiler_gpu_timeline",
    annotations=ToolAnnotations(
        title="GPU Timeline Events",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_gpu_timeline(params: GpuTimelineInput) -> str:
    """Get detailed Metal API timeline events from a .gputrace.

    Returns dispatch events with kernel names, threadgroup sizes,
    buffer bindings, and barrier events. Use filters to scope to
    specific command buffers, encoders, or kernel patterns.
    """
    try:
        resolved = str(Path(params.gputrace_path).resolve())
        data = await _run_gpu_tool(
            "gputrace_timeline.py",
            [params.gputrace_path],
            cache_key=resolved,
            cache_store=_gpu_trace_cache,
        )
        events = data["events"]

        # Apply filters
        if params.cb_filter is not None:
            # Get dispatch indices from the target command buffer
            if params.cb_filter < len(data["command_buffers"]):
                cb = data["command_buffers"][params.cb_filter]
                cb_indices = {d["index"] for d in cb["dispatches"]}
                events = [
                    e for e in events
                    if e["type"] != "dispatch" or e.get("index") in cb_indices
                ]
            else:
                events = []

        if params.encoder_filter is not None:
            # Get dispatch indices from the target encoder
            matching_encs = [
                enc for enc in data["compute_encoders"]
                if enc["encoder_idx"] == params.encoder_filter
            ]
            if matching_encs:
                enc_indices = set()
                for enc in matching_encs:
                    enc_indices.update(d["index"] for d in enc["dispatches"])
                events = [
                    e for e in events
                    if e["type"] != "dispatch" or e.get("index") in enc_indices
                ]
            else:
                events = []

        if params.kernel_filter:
            events = [
                e for e in events
                if e["type"] != "dispatch"
                or fnmatch.fnmatch(e.get("kernel", ""), params.kernel_filter)
            ]

        total = len(events)

        # Apply offset/limit
        if params.offset > 0:
            events = events[params.offset:]
        if params.limit is not None:
            events = events[: params.limit]

        return json.dumps(
            {"total": total, "returned": len(events), "events": events},
            indent=2,
        )
    except Exception as e:
        logger.exception("Error reading GPU timeline: %s", params.gputrace_path)
        return f"Error reading GPU timeline: {e}"


@mcp.tool(
    name="profiler_gpu_dependencies",
    annotations=ToolAnnotations(
        title="GPU Dependency Graph",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_gpu_dependencies(params: GpuDepsInput) -> str:
    """Analyze buffer dependencies between GPU dispatches.

    Returns dependency DAG with critical path length, parallelism
    metrics, per-kernel edge counts, and buffer hazard details.
    Use scale parameter to view at dispatch, encoder, kernel, or
    command buffer granularity.
    """
    try:
        args = [params.gputrace_path, "--scale", params.scale]
        if params.kernel_filter:
            args += ["--filter-kernel", params.kernel_filter]
        if params.cb_filter is not None:
            args += ["--filter-cb", str(params.cb_filter)]
        if params.encoder_filter is not None:
            args += ["--filter-encoder", str(params.encoder_filter)]
        data = await _run_gpu_tool("gputrace_depgraph.py", args)
        return json.dumps(data, indent=2)
    except Exception as e:
        logger.exception("Error analyzing GPU dependencies: %s", params.gputrace_path)
        return f"Error analyzing GPU dependencies: {e}"


@mcp.tool(
    name="profiler_gpu_counters",
    annotations=ToolAnnotations(
        title="GPU Performance Counters",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_gpu_counters(params: GpuCountersInput) -> str:
    """Get GPU performance counter data from shader profiling.

    Returns counter values from the MIO pipeline — occupancy, memory
    bandwidth, instruction throughput, cache hit rates, etc. Requires
    streamData from Xcode shader profiling (Replay with Profile enabled).

    Use summary=true (default) for quick stats per counter, or
    summary=false for full time-series data.
    """
    try:
        resolved = str(Path(params.gputrace_path).resolve())
        data = await _run_gpu_tool(
            "gputrace_timeline.py",
            [params.gputrace_path, "--counters-json"],
            cache_key=resolved,
            cache_store=_gpu_counter_cache,
        )

        counter_names = data["counter_names"]
        num_samples = data["num_samples"]
        samples = data["samples"]

        # Filter to requested counters
        if params.counter_filter:
            indices = [
                i for i, name in enumerate(counter_names)
                if name in params.counter_filter
            ]
            counter_names = [counter_names[i] for i in indices]
            samples = [[row[i] for i in indices] for row in samples]
        else:
            indices = list(range(len(counter_names)))

        if params.summary:
            # Compute min/max/avg for each counter
            result_counters = []
            for j, name in enumerate(counter_names):
                values = [samples[s][j] for s in range(num_samples)]
                result_counters.append({
                    "name": name,
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "num_samples": len(values),
                })
            return json.dumps(
                {"total_counters": len(result_counters), "counters": result_counters},
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "counter_names": counter_names,
                    "num_samples": num_samples,
                    "timestamps_ns": data["timestamps_ns"],
                    "samples": samples,
                },
                indent=2,
            )
    except Exception as e:
        logger.exception("Error reading GPU counters: %s", params.gputrace_path)
        return f"Error reading GPU counters: {e}"


@mcp.tool(
    name="profiler_gpu_export_perfetto",
    annotations=ToolAnnotations(
        title="Export GPU Trace to Perfetto",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profiler_gpu_export_perfetto(params: GpuExportInput) -> str:
    """Export GPU trace to Perfetto .pftrace format for visualization.

    Creates a .pftrace file viewable at ui.perfetto.dev with GPU dispatch
    timeline, barrier events, and optionally performance counter tracks.
    """
    try:
        args = [params.gputrace_path, "--format", "pftrace",
                "--group-by", params.group_by]
        if params.include_counters:
            args.append("--counters")
        if params.output_path:
            args += ["-o", params.output_path]
        data = await _run_gpu_tool(
            "gputrace_perfetto.py", args, timeout=180,
        )
        return json.dumps(
            {"output_path": data["output_path"], "size_bytes": data["size"]},
            indent=2,
        )
    except Exception as e:
        logger.exception("Error exporting GPU trace: %s", params.gputrace_path)
        return f"Error exporting GPU trace: {e}"


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
