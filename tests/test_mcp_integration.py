"""Integration tests that exercise the MCP server via Claude Code CLI.

These tests launch Claude Code with the apple-profiler MCP server configured,
send it prompts that exercise the profiler tools against real .trace files,
and verify the responses contain expected data.

Trace fixtures are stored as .zip files in tests/fixtures/traces/ and
extracted to a temp directory at session start.

Requires:
    - ANTHROPIC_API_KEY in environment or .env
    - claude CLI installed
    - macOS with xctrace (traces need xctrace for export)

Run:
    uv run pytest tests/test_mcp_integration.py -v -m mcp_integration
"""

from __future__ import annotations

import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "traces"

# Extracted trace directory (module-level, set up by fixture)
_trace_dir: Path | None = None


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


def _has_trace_zips() -> bool:
    """Check if zipped trace fixtures exist."""
    return (FIXTURES_DIR / "finder.trace.zip").exists()


def _has_claude_cli() -> bool:
    """Check if claude CLI is installed."""
    try:
        subprocess.run(["claude", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_xctrace() -> bool:
    """Check if xctrace is available (macOS only)."""
    try:
        result = subprocess.run(
            ["xcrun", "xctrace", "version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


skip_reasons: list[str] = []
if not _has_api_key():
    skip_reasons.append("ANTHROPIC_API_KEY not set")
if not _has_trace_zips():
    skip_reasons.append("no trace zips in tests/fixtures/traces/")
if not _has_claude_cli():
    skip_reasons.append("claude CLI not installed")
if not _has_xctrace():
    skip_reasons.append("xctrace not available (macOS only)")

pytestmark = pytest.mark.mcp_integration


@pytest.fixture(scope="session", autouse=True)
def trace_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract zipped trace fixtures to a temp directory for the session."""
    global _trace_dir  # noqa: PLW0603
    dest = tmp_path_factory.mktemp("traces")
    for zip_path in FIXTURES_DIR.glob("*.trace.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    _trace_dir = dest
    return dest


def _get_trace_path(name: str) -> str:
    """Get the path to an extracted trace file."""
    assert _trace_dir is not None, "trace_dir fixture not initialized"
    return str(_trace_dir / name)


def _mcp_config() -> str:
    """Build MCP config JSON pointing to the project's server."""
    return json.dumps(
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
            _mcp_config(),
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


@pytest.mark.skipif(bool(skip_reasons), reason="; ".join(skip_reasons))
class TestOpenTrace:
    """Test opening a trace and inspecting metadata."""

    def test_open_finder_trace(self) -> None:
        trace = _get_trace_path("finder.trace")
        response = _ask_claude(
            f"Use the profiler_open_trace tool to open {trace}. "
            "Tell me the device name and what tables are available. "
            "List each table schema name."
        )
        assert "cpu-profile" in response.lower() or "cpu" in response.lower()

    def test_list_tables(self) -> None:
        trace = _get_trace_path("finder.trace")
        response = _ask_claude(
            f"Use profiler_list_tables to list the tables in {trace}. "
            "Just list the schema names, one per line."
        )
        assert "cpu-profile" in response.lower() or "cpu" in response.lower()


@pytest.mark.skipif(bool(skip_reasons), reason="; ".join(skip_reasons))
class TestCpuProfiling:
    """Test CPU profiling queries."""

    def test_top_functions(self) -> None:
        trace = _get_trace_path("finder.trace")
        response = _ask_claude(
            f"Use profiler_top_functions on {trace} with n=5. "
            "List the function names and their weights."
        )
        assert len(response.strip()) > 50

    def test_cpu_samples_with_limit(self) -> None:
        trace = _get_trace_path("finder.trace")
        response = _ask_claude(
            f"Use profiler_cpu_samples on {trace} with limit=3. "
            "For each sample, show the first function in the backtrace and the weight."
        )
        assert len(response.strip()) > 50


@pytest.mark.skipif(bool(skip_reasons), reason="; ".join(skip_reasons))
class TestHangs:
    """Test hang detection queries."""

    def test_hangs_query(self) -> None:
        trace = _get_trace_path("hangs.trace")
        response = _ask_claude(
            f"Use profiler_open_trace on {trace} first. "
            "Then if there's a potential-hangs table, use profiler_hangs to get the hangs. "
            "Report how many hangs were found and the type of each."
        )
        assert "hang" in response.lower()


@pytest.mark.skipif(bool(skip_reasons), reason="; ".join(skip_reasons))
class TestSignposts:
    """Test signpost queries."""

    def test_signpost_events(self) -> None:
        trace = _get_trace_path("signpost.trace")
        response = _ask_claude(
            f"Use profiler_open_trace on {trace}. "
            "If os-signpost table exists, use profiler_signpost_events with limit=5. "
            "List the event names and subsystems."
        )
        assert len(response.strip()) > 30

    def test_signpost_intervals(self) -> None:
        trace = _get_trace_path("signpost.trace")
        response = _ask_claude(
            f"Use profiler_open_trace on {trace}. "
            "If os-signpost-interval table exists, use profiler_signpost_intervals with limit=3. "
            "Show the interval names and durations."
        )
        assert len(response.strip()) > 30


@pytest.mark.skipif(bool(skip_reasons), reason="; ".join(skip_reasons))
class TestMultiStep:
    """Test multi-step workflows that combine multiple tools."""

    def test_analyze_trace_workflow(self) -> None:
        """Test a realistic analysis workflow: open -> inspect -> query."""
        trace = _get_trace_path("finder.trace")
        response = _ask_claude(
            f"Analyze {trace}:\n"
            "1. Open it with profiler_open_trace\n"
            "2. Get the top 10 functions with profiler_top_functions\n"
            "3. Summarize: device name, recording duration, and the #1 hottest function",
            max_budget=0.50,
        )
        assert len(response.strip()) > 100
