"""Lazy MCP server for apple-profiler.

Starts instantly with no heavy imports (mcp, pydantic, pyobjc).
Responds to initialize and tools/list with pre-computed static data.
On first tools/call, imports the real server modules and delegates.

Startup cost: ~30ms (pure stdlib) vs ~1.1s (full FastMCP).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable, Coroutine
from io import TextIOWrapper
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMAS_PATH = Path(__file__).parent / "_tool_schemas.json"
_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC message/response type
JsonRpc = dict[str, Any]

# tool name -> (async handler fn, pydantic input model class)
_schemas_cache: dict[str, Any] | None = None
_handlers: dict[str, tuple[Callable[..., Coroutine[Any, Any, str]], type[Any]]] | None = None


def _load_schemas() -> dict[str, Any]:
    global _schemas_cache
    if _schemas_cache is None:
        _schemas_cache = json.loads(_SCHEMAS_PATH.read_text())
    assert _schemas_cache is not None
    return _schemas_cache


def _get_handlers() -> dict[str, tuple[Callable[..., Coroutine[Any, Any, str]], type[Any]]]:
    """Lazy-import the real MCP server module and build handler map."""
    global _handlers
    if _handlers is not None:
        return _handlers

    from apple_profiler.mcp_server import (
        CorrelatedTimelineInput,
        CpuSamplesInput,
        GpuCountersInput,
        GpuDepsInput,
        GpuExportInput,
        GpuTimelineInput,
        GpuTracePathInput,
        SignpostFilterInput,
        TableQueryInput,
        TopFunctionsInput,
        TracePathInput,
        profiler_correlated_timeline,
        profiler_cpu_samples,
        profiler_gpu_counters,
        profiler_gpu_dependencies,
        profiler_gpu_export_perfetto,
        profiler_gpu_open,
        profiler_gpu_scheduling,
        profiler_gpu_timeline,
        profiler_hangs,
        profiler_list_tables,
        profiler_open_trace,
        profiler_query_table,
        profiler_signpost_events,
        profiler_signpost_intervals,
        profiler_top_functions,
    )

    _handlers = {
        "profiler_open_trace": (profiler_open_trace, TracePathInput),
        "profiler_cpu_samples": (profiler_cpu_samples, CpuSamplesInput),
        "profiler_top_functions": (profiler_top_functions, TopFunctionsInput),
        "profiler_hangs": (profiler_hangs, TracePathInput),
        "profiler_signpost_events": (profiler_signpost_events, SignpostFilterInput),
        "profiler_signpost_intervals": (profiler_signpost_intervals, SignpostFilterInput),
        "profiler_query_table": (profiler_query_table, TableQueryInput),
        "profiler_list_tables": (profiler_list_tables, TracePathInput),
        "profiler_correlated_timeline": (profiler_correlated_timeline, CorrelatedTimelineInput),
        "profiler_gpu_open": (profiler_gpu_open, GpuTracePathInput),
        "profiler_gpu_timeline": (profiler_gpu_timeline, GpuTimelineInput),
        "profiler_gpu_dependencies": (profiler_gpu_dependencies, GpuDepsInput),
        "profiler_gpu_counters": (profiler_gpu_counters, GpuCountersInput),
        "profiler_gpu_export_perfetto": (profiler_gpu_export_perfetto, GpuExportInput),
        "profiler_gpu_scheduling": (profiler_gpu_scheduling, GpuTracePathInput),
    }
    return _handlers


def _make_response(id: int | str | None, result: object) -> JsonRpc:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _make_error(id: int | str | None, code: int, message: str) -> JsonRpc:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def _handle_initialize(id: int | str | None) -> JsonRpc:
    schemas = _load_schemas()
    server_info: dict[str, Any] = schemas["server_info"]
    return _make_response(id, {
        "protocolVersion": _PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {
            "name": server_info["name"],
            "version": server_info["version"],
        },
    })


def _handle_tools_list(id: int | str | None) -> JsonRpc:
    schemas = _load_schemas()
    return _make_response(id, {"tools": schemas["tools"]})


def _handle_tools_call(id: int | str | None, params: dict[str, Any]) -> JsonRpc:
    name: str = params.get("name", "")
    arguments: dict[str, Any] = params.get("arguments", {})

    handlers = _get_handlers()
    if name not in handlers:
        return _make_error(id, -32601, f"Unknown tool: {name}")

    handler_fn, input_model = handlers[name]
    try:
        # FastMCP wraps the input model in a "params" key
        inner: dict[str, Any] = arguments.get("params", arguments)
        input_obj = input_model(**inner)
        result_text = asyncio.run(handler_fn(input_obj))
        return _make_response(id, {
            "content": [{"type": "text", "text": result_text}],
            "isError": False,
        })
    except Exception as e:
        logger.exception("Error calling tool %s", name)
        return _make_response(id, {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "isError": True,
        })


def _handle_message(msg: JsonRpc) -> JsonRpc | None:
    method: str = msg.get("method", "")
    params: dict[str, Any] = msg.get("params") or {}
    msg_id: int | str | None = msg.get("id")

    if method == "initialize":
        return _handle_initialize(msg_id)
    elif method == "tools/list":
        return _handle_tools_list(msg_id)
    elif method == "tools/call":
        return _handle_tools_call(msg_id, params)
    elif method == "ping":
        return _make_response(msg_id, {})
    elif method.startswith("notifications/") or "id" not in msg:
        return None
    else:
        return _make_error(msg_id, -32601, f"Method not found: {method}")


def main() -> None:
    stdin = TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    stdout = TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg: JsonRpc = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = _handle_message(msg)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


if __name__ == "__main__":
    main()
