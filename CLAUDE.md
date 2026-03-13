# Apple Profiler — Claude Code Plugin Guide

## Workflow

1. **Record** a trace with `xctrace record` (or ask the user to provide a `.trace` file)
2. **Open** with `profiler_open_trace` to see metadata and available tables
3. **Analyze** with the appropriate tool(s)

## Choosing a Template

Use the template that matches what you're investigating:

| Template | Use when | Key tables |
|----------|----------|------------|
| **Time Profiler** | General CPU hotspot analysis | `time-profile` (CPU samples with stack traces) |
| **CPU Profiler** | Detailed CPU profiling with cycle counts | `cpu-profile` (CPU samples with stack traces) |
| **Metal System Trace** | GPU performance, Metal API, shader execution | `time-profile`, `metal-gpu-intervals`, `metal-driver-intervals`, `metal-application-intervals` |
| **Allocations** | Memory allocation tracking, leaks | allocation tables |
| **Leaks** | Memory leak detection | leak tables |
| **System Trace** | Thread scheduling, syscalls, VMFault | system trace tables |
| **Swift Concurrency** | async/await, actors, task scheduling | Swift actor/task tables |
| **SwiftUI** | View body evaluation, update triggers | SwiftUI-specific tables |
| **App Launch** | Launch time analysis | launch-related tables |
| **Animation Hitches** | Frame drops, hitches, render performance | hitch/frame tables |
| **Network** | HTTP traffic, connection analysis | HTTP/network tables |
| **Hangs** | Main thread unresponsiveness | `potential-hangs` |
| **Core ML** | ML model inference performance | Core ML tables |
| **Game Performance** | Combined CPU/GPU/memory for games | Multiple game-related tables |
| **Power Profiler** | Energy impact, battery drain | power/energy tables |
| **RealityKit Trace** | AR/3D rendering performance | RealityKit tables |

## Recording Traces

```bash
# Profile a command
xctrace record --template "Time Profiler" --time-limit 10s --launch -- ./my-app arg1 arg2

# Attach to running process
xctrace record --template "Time Profiler" --time-limit 10s --attach <pid-or-name>

# Profile all processes (system-wide)
xctrace record --template "Time Profiler" --time-limit 5s --all-processes

# Metal/GPU profiling
xctrace record --template "Metal System Trace" --time-limit 10s --launch -- ./my-metal-app

# Output to specific path
xctrace record --template "Time Profiler" --output /tmp/recording.trace --launch -- ./my-app

# Set environment variables for launched process
xctrace record --template "Time Profiler" --env METAL_DEVICE_WRAPPER_TYPE=1 --launch -- ./my-app
```

## Available Tools

### CPU analysis (works with Time Profiler, CPU Profiler, Metal System Trace)
- `profiler_cpu_samples` — Individual samples with **full stack traces**. Use `start_time_ns`/`end_time_ns` to scope to a time window.
- `profiler_top_functions` — Aggregated view: which functions consumed the most CPU. Also supports time-range filtering.

### Hang detection (requires Hangs instrument)
- `profiler_hangs` — Lists unresponsiveness intervals with type, duration, and thread.

### Signpost analysis (requires os_signpost instrument)
- `profiler_signpost_events` — Raw begin/end/event entries. Filter by subsystem, category, name.
- `profiler_signpost_intervals` — Matched begin+end pairs with durations.

### Generic table access (works with any trace)
- `profiler_list_tables` — Discover all available schemas in the trace.
- `profiler_query_table` — Query any table by schema name. Use this for Metal GPU tables, thermal data, disk I/O, or anything not covered by the specialized tools above.
- `profiler_open_trace` — Open trace and return metadata + table list.

## Analysis Patterns

### "Why is my app slow?"
1. Record with Time Profiler: `xctrace record --template "Time Profiler" --time-limit 10s --launch -- ./app`
2. `profiler_open_trace` to confirm recording
3. `profiler_top_functions` with n=20 to find hotspots
4. `profiler_cpu_samples` with limit=50 to see stack traces for the hot functions

### "Why does the UI hang?"
1. Record with Hangs template or include Hangs instrument
2. `profiler_hangs` to find hang intervals
3. Use the hang's `start_ns` and `duration_ns` to scope `profiler_cpu_samples` to that time window

### "Is my Metal code efficient?"
1. Record with Metal System Trace: `xctrace record --template "Metal System Trace" --launch -- ./app`
2. `profiler_list_tables` to see available Metal tables
3. `profiler_query_table` with `metal-gpu-intervals` for GPU execution
4. `profiler_query_table` with `metal-driver-intervals` for driver overhead
5. `profiler_top_functions` to see CPU-side costs

### "What happened in this specific time window?"
1. Use `profiler_cpu_samples` (no limit) to find the time range of interest from timestamps
2. Call `profiler_top_functions(start_time_ns=X, end_time_ns=Y)` to see what was hot in that window
3. Call `profiler_cpu_samples(start_time_ns=X, end_time_ns=Y, limit=30)` for detailed stack traces

## All Available Instruments

These can be added to a recording with `--instrument <name>`:

Activity Monitor, Allocations, Audio Client, Audio Server, Audio Statistics,
CPU Counters, CPU Profiler, Core Animation Activity, Core Animation Commits,
Core Animation FPS, Core Animation Server, Core ML, Data Faults, Data Fetches,
Data Saves, Disk I/O Latency, Disk Usage, Display, Filesystem Activity,
Filesystem Suggestions, Foundation Models, Frame Lifetimes, GCD Performance,
GPU, HTTP Traffic, Hangs, Hitches, Leaks, Location Energy Model,
Metal Application, Metal GPU Counters, Metal Performance Overview,
Metal Resource Events, Network Connections, Neural Engine, Points of Interest,
Power Profiler, Processor Trace, RealityKit Frames, RealityKit Metrics,
Runloops, Sampler, SceneKit Application, Swift Actors, Swift Tasks, SwiftUI,
System Call Trace, System Load, Thermal State, Thread State Trace,
Time Profiler, VM Tracker, Virtual Memory Trace, dyld Activity,
os_log, os_signpost, stdout/stderr
