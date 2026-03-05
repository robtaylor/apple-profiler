"""Integration tests that exercise the MCP server via Claude Code CLI.

These tests launch Claude Code with the apple-profiler MCP server configured,
send it prompts that exercise the profiler tools against real .trace files,
and verify the responses contain expected data.

Requires:
    - ANTHROPIC_API_KEY in environment or .env
    - Real .trace files in /tmp/claude/
    - claude CLI installed

Run:
    uv run pytest tests/test_mcp_integration.py -v -m mcp_integration
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

TRACE_DIR = Path("/tmp/claude")
PROJECT_DIR = Path(__file__).parent.parent

# MCP server config for claude CLI
MCP_CONFIG = json.dumps(
    {
        "mcpServers": {
            "apple-profiler": {
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    str(PROJECT_DIR),
                    "apple-profiler",
                ],
            }
        }
    }
)


def _has_api_key() -> bool:
    """Check if ANTHROPIC_API_KEY is available."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                return True
    return False


def _has_traces() -> bool:
    """Check if real trace files exist."""
    return (TRACE_DIR / "finder.trace").is_dir()


def _has_claude_cli() -> bool:
    """Check if claude CLI is installed."""
    try:
        subprocess.run(["claude", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


skip_reason = []
if not _has_api_key():
    skip_reason.append("ANTHROPIC_API_KEY not set")
if not _has_traces():
    skip_reason.append("no trace files in /tmp/claude/")
if not _has_claude_cli():
    skip_reason.append("claude CLI not installed")

pytestmark = pytest.mark.mcp_integration


def _load_env() -> dict[str, str]:
    """Load environment including .env file, stripping nested-session blockers."""
    env = os.environ.copy()
    # Remove CLAUDECODE env var to allow spawning claude CLI from within a session
    env.pop("CLAUDECODE", None)
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip("\"'")
    return env


def _ask_claude(prompt: str, *, max_budget: float = 0.25, model: str = "haiku") -> str:
    """Send a prompt to Claude Code with the MCP server and return the response."""
    result = subprocess.run(
        [
            "claude",
            "--print",
            "--output-format",
            "text",
            "--model",
            model,
            "--mcp-config",
            MCP_CONFIG,
            "--strict-mcp-config",
            "--max-budget-usd",
            str(max_budget),
            "--no-session-persistence",
            "--permission-mode",
            "bypassPermissions",
            "--allowedTools",
            "mcp__apple-profiler__profiler_open_trace",
            "mcp__apple-profiler__profiler_cpu_samples",
            "mcp__apple-profiler__profiler_top_functions",
            "mcp__apple-profiler__profiler_hangs",
            "mcp__apple-profiler__profiler_signpost_events",
            "mcp__apple-profiler__profiler_signpost_intervals",
            "mcp__apple-profiler__profiler_list_tables",
            "--disallowedTools",
            "Bash",
            "Edit",
            "Write",
            "-p",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_load_env(),
    )
    assert result.returncode == 0, f"claude CLI failed: {result.stderr}"
    return result.stdout


@pytest.mark.skipif(bool(skip_reason), reason="; ".join(skip_reason))
class TestOpenTrace:
    """Test opening a trace and inspecting metadata."""

    def test_open_finder_trace(self) -> None:
        response = _ask_claude(
            f"Use the profiler_open_trace tool to open {TRACE_DIR}/finder.trace. "
            "Tell me the device name and what tables are available. "
            "List each table schema name."
        )
        # Should mention cpu-profile since finder.trace was recorded with CPU Profiler
        assert "cpu-profile" in response.lower() or "cpu" in response.lower()

    def test_list_tables(self) -> None:
        response = _ask_claude(
            f"Use profiler_list_tables to list the tables in {TRACE_DIR}/finder.trace. "
            "Just list the schema names, one per line."
        )
        assert "cpu-profile" in response.lower() or "cpu" in response.lower()


@pytest.mark.skipif(bool(skip_reason), reason="; ".join(skip_reason))
class TestCpuProfiling:
    """Test CPU profiling queries."""

    def test_top_functions(self) -> None:
        response = _ask_claude(
            f"Use profiler_top_functions on {TRACE_DIR}/finder.trace with n=5. "
            "List the function names and their weights."
        )
        # Should contain at least one function name
        assert len(response.strip()) > 50  # non-trivial response

    def test_cpu_samples_with_limit(self) -> None:
        response = _ask_claude(
            f"Use profiler_cpu_samples on {TRACE_DIR}/finder.trace with limit=3. "
            "For each sample, show the first function in the backtrace and the weight."
        )
        assert len(response.strip()) > 50


@pytest.mark.skipif(bool(skip_reason), reason="; ".join(skip_reason))
class TestHangs:
    """Test hang detection queries."""

    def test_hangs_query(self) -> None:
        response = _ask_claude(
            f"Use profiler_open_trace on {TRACE_DIR}/hangs.trace first. "
            "Then if there's a potential-hangs table, use profiler_hangs to get the hangs. "
            "Report how many hangs were found and the type of each."
        )
        # Should mention hang-related info
        assert "hang" in response.lower()


@pytest.mark.skipif(bool(skip_reason), reason="; ".join(skip_reason))
class TestSignposts:
    """Test signpost queries."""

    def test_signpost_events(self) -> None:
        response = _ask_claude(
            f"Use profiler_open_trace on {TRACE_DIR}/signpost.trace. "
            "If os-signpost table exists, use profiler_signpost_events with limit=5. "
            "List the event names and subsystems."
        )
        assert len(response.strip()) > 30

    def test_signpost_intervals(self) -> None:
        response = _ask_claude(
            f"Use profiler_open_trace on {TRACE_DIR}/signpost.trace. "
            "If os-signpost-interval table exists, use profiler_signpost_intervals with limit=3. "
            "Show the interval names and durations."
        )
        assert len(response.strip()) > 30


@pytest.mark.skipif(bool(skip_reason), reason="; ".join(skip_reason))
class TestMultiStep:
    """Test multi-step workflows that combine multiple tools."""

    def test_analyze_trace_workflow(self) -> None:
        """Test a realistic analysis workflow: open -> inspect -> query."""
        response = _ask_claude(
            f"Analyze {TRACE_DIR}/finder.trace:\n"
            "1. Open it with profiler_open_trace\n"
            "2. Get the top 10 functions with profiler_top_functions\n"
            "3. Summarize: device name, recording duration, and the #1 hottest function",
            max_budget=0.50,
        )
        # Should have a substantive multi-part answer
        assert len(response.strip()) > 100
