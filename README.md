# apple-profiler

A Claude Code plugin that wraps `xctrace` to parse and analyze Instruments `.trace` files and Metal `.gputrace` bundles. It exposes CPU profiling, hang detection, signpost analysis, correlated CPU+GPU timelines, Metal GPU trace analysis, and generic table queries as MCP tools — letting Claude investigate performance issues directly from trace recordings.

## Installation

```bash
claude plugin add --transport stdio -- uv run --directory /path/to/apple-profiler apple-profiler
```

Or from the plugin registry:

```bash
claude plugin add apple-profiler
```

## Available Tools

### CPU Analysis (`.trace` files)

| Tool | Description |
|------|-------------|
| `profiler_open_trace` | Open a `.trace` file, return device info, duration, and available tables |
| `profiler_cpu_samples` | Get CPU samples with full stack traces (backtraces), optional time-range filtering |
| `profiler_top_functions` | Top N functions by CPU weight, optional time-range filtering |
| `profiler_hangs` | Detected hang/unresponsiveness intervals |
| `profiler_signpost_events` | Raw `os-signpost` Begin/End/Event entries with optional filtering |
| `profiler_signpost_intervals` | Matched signpost intervals (begin+end pairs) with durations |
| `profiler_query_table` | Query any data table by schema name — works with Metal, thermal, disk I/O, etc. |
| `profiler_list_tables` | List all available data tables and their schemas |

### Correlated CPU+GPU Analysis (Metal System Trace `.trace` files)

| Tool | Description |
|------|-------------|
| `profiler_correlated_timeline` | Time-aligned CPU+GPU timeline with auto-detected phases (CPU_BOUND, GPU_BOUND, BALANCED, PIPELINE_BUBBLE, IDLE) |

### GPU Trace Analysis (`.gputrace` files)

| Tool | Description |
|------|-------------|
| `profiler_gpu_open` | Structural overview: kernels, command buffers, encoders, dispatch/barrier counts |
| `profiler_gpu_timeline` | Detailed dispatch events with kernel names, threadgroup sizes, buffer bindings |
| `profiler_gpu_dependencies` | Buffer hazard DAG with critical path analysis at dispatch, encoder, kernel, or command buffer scale |
| `profiler_gpu_counters` | Shader profiling counters (occupancy, bandwidth, cache rates) — requires Xcode replay |
| `profiler_gpu_export_perfetto` | Export GPU timeline to `.pftrace` for visualization in ui.perfetto.dev |
| `profiler_gpu_scheduling` | Scheduling overhead analysis: inter-encoder gaps and dispatch fusion candidates |

## Usage Examples

### CPU Profiling

Works with CPU Profiler, Time Profiler, and Metal System Trace templates.

```
1. profiler_open_trace(trace_path="recording.trace")
2. profiler_top_functions(trace_path="recording.trace", n=10)
3. profiler_cpu_samples(trace_path="recording.trace", limit=50)
```

### Time-Range Scoped Analysis

```
# What was hot between 1.0s and 1.5s?
profiler_top_functions(trace_path="recording.trace", n=10,
    start_time_ns=1000000000, end_time_ns=1500000000)

# Stack traces in that window
profiler_cpu_samples(trace_path="recording.trace",
    start_time_ns=1000000000, end_time_ns=1500000000, limit=20)
```

### Hang Detection

```
1. profiler_open_trace(trace_path="recording.trace")
2. profiler_hangs(trace_path="recording.trace")
```

### Correlated CPU+GPU Timeline

```
1. profiler_open_trace(trace_path="metal-system.trace")
2. profiler_correlated_timeline(trace_path="metal-system.trace")
   # Returns per-bucket CPU/GPU activity with phase classification
```

### GPU Trace Analysis

```
1. profiler_gpu_open(trace_path="capture.gputrace")
2. profiler_gpu_scheduling(trace_path="capture.gputrace")
3. profiler_gpu_dependencies(trace_path="capture.gputrace", scale="kernel")
4. profiler_gpu_timeline(trace_path="capture.gputrace", kernel_filter="matmul")
5. profiler_gpu_export_perfetto(trace_path="capture.gputrace", output_path="/tmp/out.pftrace")
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

# Metal System Trace (CPU + GPU correlation)
xctrace record --template "Metal System Trace" --launch -- ./my-metal-app

# Hangs detection
xctrace record --template "Hangs" --launch -- ./my-app
```

### Capturing GPU Traces

GPU traces (`.gputrace` bundles) capture the full Metal command stream:

```bash
# Capture via environment variables
MTL_CAPTURE_ENABLED=1 METAL_CAPTURE_ENABLED=1 ./my-metal-app

# Specify output path
MTL_CAPTURE_ENABLED=1 METAL_CAPTURE_ENABLED=1 \
  METAL_CAPTURE_OUTPUT_PATH=/tmp/capture.gputrace \
  ./my-metal-app
```

For programmatic capture of specific code sections, use `MTLCaptureManager` — see CLAUDE.md for details.

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
