# apple-profiler

A Claude Code plugin that wraps `xctrace` to parse and analyze Instruments `.trace` files. It exposes CPU profiling, hang detection, signpost analysis, Metal/GPU trace data, and generic table queries as MCP tools — letting Claude investigate performance issues directly from trace recordings.

## Installation

```bash
claude plugin add --transport stdio -- uv run --directory /path/to/apple-profiler apple-profiler
```

Or from the plugin registry:

```bash
claude plugin add apple-profiler
```

## Available Tools

| Tool | Description |
|------|-------------|
| `profiler_open_trace` | Open a `.trace` file, return device info, duration, and available tables |
| `profiler_cpu_samples` | Get CPU profile samples with backtraces and cycle weights |
| `profiler_top_functions` | Top N functions by total CPU cycle weight |
| `profiler_hangs` | Detected hang/unresponsiveness intervals |
| `profiler_signpost_events` | Raw `os-signpost` Begin/End/Event entries with optional filtering |
| `profiler_signpost_intervals` | Matched signpost intervals (begin+end pairs) with durations |
| `profiler_query_table` | Query any data table by schema name — works with Metal, thermal, disk I/O, etc. |
| `profiler_list_tables` | List all available data tables and their schemas |

## Usage Examples

### CPU Profiling

```
1. profiler_open_trace(trace_path="recording.trace")
2. profiler_top_functions(trace_path="recording.trace", n=10)
3. profiler_cpu_samples(trace_path="recording.trace", limit=50)
```

### Hang Detection

```
1. profiler_open_trace(trace_path="recording.trace")
2. profiler_hangs(trace_path="recording.trace")
```

### Metal System Trace

```
1. profiler_open_trace(trace_path="metal.trace")
2. profiler_list_tables(trace_path="metal.trace")
3. profiler_query_table(trace_path="metal.trace", schema="metal-gpu-interval", limit=20)
```

The `profiler_query_table` tool works with any table schema reported by `profiler_list_tables`, including `metal-gpu-interval`, `gpu-shader-timeline`, `thermal-state`, and others.

### Signpost Analysis

```
1. profiler_signpost_intervals(trace_path="recording.trace", subsystem="com.example.app")
2. profiler_signpost_events(trace_path="recording.trace", category="networking")
```

## Recording Traces

Use `xctrace` to record traces from the command line:

```bash
# CPU profiling
xctrace record --template "CPU Profiler" --launch -- ./my-app

# Time Profiler with an already-running process
xctrace record --template "Time Profiler" --attach <pid> --time-limit 10s

# Metal System Trace
xctrace record --template "Metal System Trace" --launch -- ./my-metal-app

# Hangs detection
xctrace record --template "Hangs" --launch -- ./my-app
```

## Development

### Setup

```bash
uv sync --group dev
```

### Tests

```bash
# Unit tests only
uv run pytest -m "not integration and not mcp_integration and not metal_integration"

# Integration tests (requires xctrace / macOS)
uv run pytest -m integration

# Metal integration tests (compiles and traces a Metal compute program)
uv run pytest -m metal_integration

# All tests
uv run pytest
```

### Linting

```bash
uv run ruff check .
uv run pyright
```

## License

MIT
