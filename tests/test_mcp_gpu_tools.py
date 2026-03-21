"""Tests for GPU MCP tools in mcp_server.py.

Tests _run_gpu_tool error handling, each GPU MCP tool with mocked subprocess
output, cache behavior, and input validation.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from apple_profiler.mcp_server import (
    _gpu_trace_cache,
    _run_gpu_tool,
    profiler_gpu_counters,
    profiler_gpu_dependencies,
    profiler_gpu_export_perfetto,
    profiler_gpu_open,
    profiler_gpu_timeline,
)

# ── Sample test data ──

SAMPLE_TRACE_DATA = {
    "metadata": {"device": "Apple M2 Max", "capture_type": "compute"},
    "total_functions": 500,
    "kernels": {"0x1000": "lu_factor", "0x2000": "lu_solve"},
    "pipelines": {"0x3000": "lu_factor_pipeline"},
    "events": [
        {
            "type": "dispatch",
            "index": 0,
            "kernel": "lu_factor",
            "threadgroups": [16, 1, 1],
            "threads_per_threadgroup": [256, 1, 1],
            "buffers_bound": {"0x100": 0, "0x200": 1},
        },
        {
            "type": "dispatch",
            "index": 1,
            "kernel": "lu_solve",
            "threadgroups": [8, 1, 1],
            "threads_per_threadgroup": [128, 1, 1],
            "buffers_bound": {"0x200": 0, "0x300": 1},
        },
        {"type": "barrier", "index": 2},
        {
            "type": "dispatch",
            "index": 3,
            "kernel": "lu_factor",
            "threadgroups": [16, 1, 1],
            "threads_per_threadgroup": [256, 1, 1],
            "buffers_bound": {"0x400": 0},
        },
    ],
    "command_buffers": [
        {
            "addr": "0xCB0",
            "dispatches": [
                {"index": 0, "kernel": "lu_factor"},
                {"index": 1, "kernel": "lu_solve"},
                {"index": 3, "kernel": "lu_factor"},
            ],
        },
    ],
    "compute_encoders": [
        {
            "encoder_idx": 0,
            "command_buffer_idx": 0,
            "dispatches": [
                {"index": 0, "kernel": "lu_factor"},
                {"index": 1, "kernel": "lu_solve"},
            ],
        },
        {
            "encoder_idx": 1,
            "command_buffer_idx": 0,
            "dispatches": [
                {"index": 3, "kernel": "lu_factor"},
            ],
        },
    ],
}

SAMPLE_COUNTER_DATA = {
    "counter_names": ["ALU Utilization", "Memory Bandwidth", "Occupancy"],
    "num_samples": 3,
    "timestamps_ns": [0, 1000, 2000],
    "samples": [
        [0.5, 100.0, 0.75],
        [0.6, 120.0, 0.80],
        [0.7, 90.0, 0.70],
    ],
}

SAMPLE_DEPS_DATA = {
    "nodes": [
        {"id": 0, "func_idx": 10, "kernel": "lu_factor", "buffers": [],
         "command_buffer": 0, "encoder": 0},
        {"id": 1, "func_idx": 20, "kernel": "lu_solve", "buffers": [],
         "command_buffer": 0, "encoder": 0},
    ],
    "edges": [
        {"source": 0, "target": 1, "type": "raw", "buffer": "0x200"},
    ],
    "summary": {
        "total_dispatches": 2,
        "total_edges": 1,
        "edge_types": {"raw": 1},
        "critical_path_length": 2,
        "isolated_nodes": 0,
        "is_dag": True,
    },
    "scale": "encoder",
    "metadata": {
        "num_command_buffers": 1,
        "num_encoders": 2,
        "num_barriers": 1,
    },
}

SAMPLE_EXPORT_DATA = {
    "output_path": "/tmp/test_perfetto.pftrace",
    "size": 12345,
}


# ── Helper to create mock subprocess ──


def _mock_process(stdout_data: str, returncode: int = 0, stderr: str = ""):
    """Create a mock asyncio subprocess."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(
        return_value=(stdout_data.encode(), stderr.encode())
    )
    proc.kill = AsyncMock()
    return proc


# ── Tests for _run_gpu_tool ──


class TestRunGpuTool:
    """Tests for the _run_gpu_tool subprocess helper."""

    @pytest.mark.asyncio
    async def test_successful_run(self, tmp_path):
        """Successful subprocess returns parsed JSON."""
        proc = _mock_process(json.dumps({"key": "value"}))
        with patch("apple_profiler.mcp_server.asyncio.create_subprocess_exec",
                    return_value=proc):
            # Create a dummy script so the file check passes
            with patch("apple_profiler.mcp_server._TOOLS_DIR", tmp_path):
                (tmp_path / "test.py").touch()
                result = await _run_gpu_tool("test.py", ["/path/to/trace"])
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_script_not_found(self, tmp_path):
        """Missing script raises FileNotFoundError."""
        with patch("apple_profiler.mcp_server._TOOLS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="GPU tool script not found"):
                await _run_gpu_tool("nonexistent.py", ["/path"])

    @pytest.mark.asyncio
    async def test_subprocess_failure(self, tmp_path):
        """Non-zero exit code raises RuntimeError with stderr."""
        proc = _mock_process("", returncode=1, stderr="Error: bad input")
        with patch("apple_profiler.mcp_server.asyncio.create_subprocess_exec",
                    return_value=proc):
            with patch("apple_profiler.mcp_server._TOOLS_DIR", tmp_path):
                (tmp_path / "test.py").touch()
                with pytest.raises(RuntimeError, match="failed.*Error: bad input"):
                    await _run_gpu_tool("test.py", ["/path"])

    @pytest.mark.asyncio
    async def test_invalid_json_output(self, tmp_path):
        """Invalid JSON stdout raises RuntimeError."""
        proc = _mock_process("not json at all")
        with patch("apple_profiler.mcp_server.asyncio.create_subprocess_exec",
                    return_value=proc):
            with patch("apple_profiler.mcp_server._TOOLS_DIR", tmp_path):
                (tmp_path / "test.py").touch()
                with pytest.raises(RuntimeError, match="invalid JSON"):
                    await _run_gpu_tool("test.py", ["/path"])

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path):
        """Subprocess exceeding timeout raises TimeoutError."""
        proc = AsyncMock()
        proc.communicate = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        proc.kill = AsyncMock()
        with patch("apple_profiler.mcp_server.asyncio.create_subprocess_exec",
                    return_value=proc):
            with patch("apple_profiler.mcp_server._TOOLS_DIR", tmp_path):
                (tmp_path / "test.py").touch()
                with pytest.raises(TimeoutError, match="timed out"):
                    await _run_gpu_tool("test.py", ["/path"], timeout=1)

    @pytest.mark.asyncio
    async def test_cache_hit(self, tmp_path):
        """Cached results are returned without spawning subprocess."""
        cache: dict[str, dict] = {"my_key": {"cached": True}}
        result = await _run_gpu_tool(
            "test.py", ["/path"],
            cache_key="my_key",
            cache_store=cache,
        )
        assert result == {"cached": True}

    @pytest.mark.asyncio
    async def test_cache_store(self, tmp_path):
        """Successful results are stored in cache."""
        cache: dict[str, dict] = {}
        proc = _mock_process(json.dumps({"fresh": True}))
        with patch("apple_profiler.mcp_server.asyncio.create_subprocess_exec",
                    return_value=proc):
            with patch("apple_profiler.mcp_server._TOOLS_DIR", tmp_path):
                (tmp_path / "test.py").touch()
                result = await _run_gpu_tool(
                    "test.py", ["/path"],
                    cache_key="new_key",
                    cache_store=cache,
                )
        assert result == {"fresh": True}
        assert cache["new_key"] == {"fresh": True}

    @pytest.mark.asyncio
    async def test_json_flag_appended(self, tmp_path):
        """--json is appended to command when not already present."""
        proc = _mock_process(json.dumps({}))
        with patch("apple_profiler.mcp_server.asyncio.create_subprocess_exec",
                    return_value=proc) as mock_exec:
            with patch("apple_profiler.mcp_server._TOOLS_DIR", tmp_path):
                (tmp_path / "test.py").touch()
                await _run_gpu_tool("test.py", ["/path"])
        cmd = mock_exec.call_args[0]
        assert cmd[-1] == "--json"

    @pytest.mark.asyncio
    async def test_json_flag_not_duplicated(self, tmp_path):
        """--json is not duplicated when already in args."""
        proc = _mock_process(json.dumps({}))
        with patch("apple_profiler.mcp_server.asyncio.create_subprocess_exec",
                    return_value=proc) as mock_exec:
            with patch("apple_profiler.mcp_server._TOOLS_DIR", tmp_path):
                (tmp_path / "test.py").touch()
                await _run_gpu_tool("test.py", ["/path", "--json"])
        cmd = mock_exec.call_args[0]
        assert cmd.count("--json") == 1


# ── Tests for MCP tools ──


def _patch_gpu_tool(return_data):
    """Patch _run_gpu_tool to return given data."""
    return patch(
        "apple_profiler.mcp_server._run_gpu_tool",
        new_callable=AsyncMock,
        return_value=return_data,
    )


class TestProfilerGpuOpen:
    """Tests for profiler_gpu_open tool."""

    @pytest.mark.asyncio
    async def test_returns_overview(self):
        with _patch_gpu_tool(SAMPLE_TRACE_DATA):
            from apple_profiler.mcp_server import GpuTracePathInput
            result = await profiler_gpu_open(
                GpuTracePathInput(gputrace_path="/tmp/test.gputrace")
            )
        data = json.loads(result)
        assert data["total_functions"] == 500
        assert data["num_dispatches"] == 3
        assert data["num_barriers"] == 1
        assert len(data["kernels"]) == 2
        assert len(data["command_buffers"]) == 1
        assert len(data["compute_encoders"]) == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        with patch("apple_profiler.mcp_server._run_gpu_tool",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("test error")):
            from apple_profiler.mcp_server import GpuTracePathInput
            result = await profiler_gpu_open(
                GpuTracePathInput(gputrace_path="/tmp/missing.gputrace")
            )
        assert "Error" in result
        assert "test error" in result


class TestProfilerGpuTimeline:
    """Tests for profiler_gpu_timeline tool."""

    @pytest.mark.asyncio
    async def test_returns_all_events(self):
        with _patch_gpu_tool(SAMPLE_TRACE_DATA):
            from apple_profiler.mcp_server import GpuTimelineInput
            result = await profiler_gpu_timeline(
                GpuTimelineInput(gputrace_path="/tmp/test.gputrace")
            )
        data = json.loads(result)
        assert data["total"] == 4  # 3 dispatches + 1 barrier

    @pytest.mark.asyncio
    async def test_kernel_filter(self):
        with _patch_gpu_tool(SAMPLE_TRACE_DATA):
            from apple_profiler.mcp_server import GpuTimelineInput
            result = await profiler_gpu_timeline(
                GpuTimelineInput(
                    gputrace_path="/tmp/test.gputrace",
                    kernel_filter="lu_factor",
                )
            )
        data = json.loads(result)
        # 2 lu_factor dispatches + 1 barrier (not filtered since type != dispatch)
        assert data["total"] == 3

    @pytest.mark.asyncio
    async def test_cb_filter(self):
        with _patch_gpu_tool(SAMPLE_TRACE_DATA):
            from apple_profiler.mcp_server import GpuTimelineInput
            result = await profiler_gpu_timeline(
                GpuTimelineInput(
                    gputrace_path="/tmp/test.gputrace",
                    cb_filter=0,
                )
            )
        data = json.loads(result)
        # CB0 has all 3 dispatches + barrier passes through
        assert data["total"] == 4

    @pytest.mark.asyncio
    async def test_encoder_filter(self):
        with _patch_gpu_tool(SAMPLE_TRACE_DATA):
            from apple_profiler.mcp_server import GpuTimelineInput
            result = await profiler_gpu_timeline(
                GpuTimelineInput(
                    gputrace_path="/tmp/test.gputrace",
                    encoder_filter=1,
                )
            )
        data = json.loads(result)
        # Encoder 1 has 1 dispatch (index 3) + barrier passes through
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_offset_and_limit(self):
        with _patch_gpu_tool(SAMPLE_TRACE_DATA):
            from apple_profiler.mcp_server import GpuTimelineInput
            result = await profiler_gpu_timeline(
                GpuTimelineInput(
                    gputrace_path="/tmp/test.gputrace",
                    offset=1,
                    limit=2,
                )
            )
        data = json.loads(result)
        assert data["total"] == 4
        assert data["returned"] == 2


class TestProfilerGpuDependencies:
    """Tests for profiler_gpu_dependencies tool."""

    @pytest.mark.asyncio
    async def test_returns_graph(self):
        with _patch_gpu_tool(SAMPLE_DEPS_DATA):
            from apple_profiler.mcp_server import GpuDepsInput
            result = await profiler_gpu_dependencies(
                GpuDepsInput(gputrace_path="/tmp/test.gputrace")
            )
        data = json.loads(result)
        assert data["summary"]["total_dispatches"] == 2
        assert data["summary"]["critical_path_length"] == 2

    @pytest.mark.asyncio
    async def test_scale_and_filters_passed(self):
        with _patch_gpu_tool(SAMPLE_DEPS_DATA) as mock_tool:
            from apple_profiler.mcp_server import GpuDepsInput
            await profiler_gpu_dependencies(
                GpuDepsInput(
                    gputrace_path="/tmp/test.gputrace",
                    scale="kernel",
                    kernel_filter="lu*",
                    cb_filter=0,
                    encoder_filter=1,
                )
            )
        call_args = mock_tool.call_args
        cli_args = call_args[0][1]
        assert "--scale" in cli_args
        assert "kernel" in cli_args
        assert "--filter-kernel" in cli_args
        assert "lu*" in cli_args
        assert "--filter-cb" in cli_args
        assert "0" in cli_args
        assert "--filter-encoder" in cli_args
        assert "1" in cli_args


class TestProfilerGpuCounters:
    """Tests for profiler_gpu_counters tool."""

    @pytest.mark.asyncio
    async def test_summary_mode(self):
        with _patch_gpu_tool(SAMPLE_COUNTER_DATA):
            from apple_profiler.mcp_server import GpuCountersInput
            result = await profiler_gpu_counters(
                GpuCountersInput(gputrace_path="/tmp/test.gputrace", summary=True)
            )
        data = json.loads(result)
        assert data["total_counters"] == 3
        alu = next(c for c in data["counters"] if c["name"] == "ALU Utilization")
        assert alu["min"] == 0.5
        assert alu["max"] == 0.7

    @pytest.mark.asyncio
    async def test_full_timeseries(self):
        with _patch_gpu_tool(SAMPLE_COUNTER_DATA):
            from apple_profiler.mcp_server import GpuCountersInput
            result = await profiler_gpu_counters(
                GpuCountersInput(gputrace_path="/tmp/test.gputrace", summary=False)
            )
        data = json.loads(result)
        assert data["num_samples"] == 3
        assert len(data["counter_names"]) == 3
        assert len(data["samples"]) == 3

    @pytest.mark.asyncio
    async def test_counter_filter(self):
        with _patch_gpu_tool(SAMPLE_COUNTER_DATA):
            from apple_profiler.mcp_server import GpuCountersInput
            result = await profiler_gpu_counters(
                GpuCountersInput(
                    gputrace_path="/tmp/test.gputrace",
                    counter_filter=["Occupancy"],
                    summary=True,
                )
            )
        data = json.loads(result)
        assert data["total_counters"] == 1
        assert data["counters"][0]["name"] == "Occupancy"


class TestProfilerGpuExportPerfetto:
    """Tests for profiler_gpu_export_perfetto tool."""

    @pytest.mark.asyncio
    async def test_export_returns_path(self):
        with _patch_gpu_tool(SAMPLE_EXPORT_DATA):
            from apple_profiler.mcp_server import GpuExportInput
            result = await profiler_gpu_export_perfetto(
                GpuExportInput(gputrace_path="/tmp/test.gputrace")
            )
        data = json.loads(result)
        assert "output_path" in data
        assert "size_bytes" in data

    @pytest.mark.asyncio
    async def test_args_passed_correctly(self):
        with _patch_gpu_tool(SAMPLE_EXPORT_DATA) as mock_tool:
            from apple_profiler.mcp_server import GpuExportInput
            await profiler_gpu_export_perfetto(
                GpuExportInput(
                    gputrace_path="/tmp/test.gputrace",
                    output_path="/tmp/out.pftrace",
                    group_by="cb",
                    include_counters=False,
                )
            )
        call_args = mock_tool.call_args
        cli_args = call_args[0][1]
        assert "--format" in cli_args
        assert "pftrace" in cli_args
        assert "--group-by" in cli_args
        assert "cb" in cli_args
        assert "-o" in cli_args
        assert "/tmp/out.pftrace" in cli_args
        assert "--counters" not in cli_args

    @pytest.mark.asyncio
    async def test_counters_flag_included(self):
        with _patch_gpu_tool(SAMPLE_EXPORT_DATA) as mock_tool:
            from apple_profiler.mcp_server import GpuExportInput
            await profiler_gpu_export_perfetto(
                GpuExportInput(
                    gputrace_path="/tmp/test.gputrace",
                    include_counters=True,
                )
            )
        call_args = mock_tool.call_args
        cli_args = call_args[0][1]
        assert "--counters" in cli_args


# ── Cache behavior tests ──


class TestCacheBehavior:
    """Test that caching works across tool invocations."""

    @pytest.mark.asyncio
    async def test_gpu_open_uses_cache(self):
        """Second call to profiler_gpu_open should hit cache."""
        _gpu_trace_cache.clear()
        call_count = 0

        async def mock_run(script, args, **kwargs):
            nonlocal call_count
            cache_key = kwargs.get("cache_key")
            cache_store = kwargs.get("cache_store")
            if cache_key and cache_store is not None and cache_key in cache_store:
                return cache_store[cache_key]
            call_count += 1
            result = SAMPLE_TRACE_DATA
            if cache_key and cache_store is not None:
                cache_store[cache_key] = result
            return result

        with patch("apple_profiler.mcp_server._run_gpu_tool", side_effect=mock_run):
            from apple_profiler.mcp_server import GpuTracePathInput
            params = GpuTracePathInput(gputrace_path="/tmp/test.gputrace")
            await profiler_gpu_open(params)
            await profiler_gpu_open(params)

        assert call_count == 1
        _gpu_trace_cache.clear()
