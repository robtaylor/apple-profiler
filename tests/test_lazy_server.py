"""Tests for the lazy MCP server proxy."""
from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest


def _send_recv(proc: subprocess.Popen, msg: dict) -> dict:
    """Send a JSON-RPC message and read the response."""
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(msg).encode() + b"\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "Server returned no response"
    return json.loads(line)


@pytest.fixture
def lazy_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "apple_profiler._lazy_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    yield proc
    proc.kill()
    proc.wait()


class TestLazyProtocol:
    def test_initialize(self, lazy_server: subprocess.Popen) -> None:
        resp = _send_recv(lazy_server, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        })
        assert resp["id"] == 1
        result = resp["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["capabilities"] == {"tools": {}}
        assert result["serverInfo"]["name"] == "apple_profiler_mcp"

    def test_tools_list(self, lazy_server: subprocess.Popen) -> None:
        # Initialize first
        _send_recv(lazy_server, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        # List tools
        resp = _send_recv(lazy_server, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert "profiler_open_trace" in tool_names
        assert "profiler_gpu_open" in tool_names
        assert len(tools) == 15

    def test_ping(self, lazy_server: subprocess.Popen) -> None:
        resp = _send_recv(lazy_server, {
            "jsonrpc": "2.0", "id": 99, "method": "ping", "params": {},
        })
        assert resp["id"] == 99
        assert resp["result"] == {}

    def test_unknown_method(self, lazy_server: subprocess.Popen) -> None:
        resp = _send_recv(lazy_server, {
            "jsonrpc": "2.0", "id": 3, "method": "foo/bar", "params": {},
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_notification_no_response(self, lazy_server: subprocess.Popen) -> None:
        assert lazy_server.stdin is not None
        assert lazy_server.stdout is not None
        # Send a notification (no id)
        lazy_server.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
            + b"\n"
        )
        lazy_server.stdin.flush()
        # Send a ping to verify server is still alive and didn't send
        # a response for the notification
        resp = _send_recv(lazy_server, {
            "jsonrpc": "2.0", "id": 42, "method": "ping", "params": {},
        })
        assert resp["id"] == 42

    def test_tools_call_unknown_tool(self, lazy_server: subprocess.Popen) -> None:
        resp = _send_recv(lazy_server, {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        assert "error" in resp
        assert "Unknown tool" in resp["error"]["message"]

    def test_tools_call_bad_args(self, lazy_server: subprocess.Popen) -> None:
        resp = _send_recv(lazy_server, {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "profiler_open_trace", "arguments": {"params": {}}},
        })
        assert resp["result"]["isError"] is True

    def test_tool_schemas_match_mcp_server(self) -> None:
        """Verify static schemas match what FastMCP generates."""
        import asyncio

        from apple_profiler._lazy_server import _load_schemas
        from apple_profiler.mcp_server import mcp

        static_tools = _load_schemas()["tools"]
        live_tools = asyncio.run(mcp.list_tools())

        static_names = sorted(t["name"] for t in static_tools)
        live_names = sorted(t.name for t in live_tools)
        assert static_names == live_names


class TestLazyStartupPerformance:
    def test_startup_under_200ms(self) -> None:
        """Lazy server should start and respond to initialize in <200ms."""
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "apple_profiler._lazy_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            resp = _send_recv(proc, {
                "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
            })
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert resp["result"]["protocolVersion"] == "2024-11-05"
            assert elapsed_ms < 200, f"Startup took {elapsed_ms:.0f}ms, expected <200ms"
        finally:
            proc.kill()
            proc.wait()
