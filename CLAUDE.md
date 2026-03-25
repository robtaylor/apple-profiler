# Apple Profiler — Claude Code Plugin Guide

## Workflow

### CPU profiling (`.trace` files)
1. **Record** with `xctrace record` (see [Recording CPU Traces](#recording-cpu-traces))
2. **Open** with `profiler_open_trace` to see metadata and available tables
3. **Analyze** with `profiler_top_functions`, `profiler_cpu_samples`, etc.

### GPU profiling (`.gputrace` files)
1. **Capture** with Metal environment variables or `MTLCaptureManager` (see [Capturing GPU Traces](#capturing-gpu-traces))
2. **Open** with `profiler_gpu_open` to see kernels, command buffers, encoders
3. **Analyze** with `profiler_gpu_timeline`, `profiler_gpu_dependencies`, `profiler_gpu_counters`

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

## Recording CPU Traces

Use `xctrace record` to create `.trace` files for CPU/system analysis:

```bash
# Profile a command (records until program exits or time-limit)
xctrace record --template "Time Profiler" --time-limit 10s --launch -- ./my-app arg1 arg2

# Attach to running process
xctrace record --template "Time Profiler" --time-limit 10s --attach <pid-or-name>

# Profile all processes (system-wide)
xctrace record --template "Time Profiler" --time-limit 5s --all-processes

# CPU + GPU system-level profiling
xctrace record --template "Metal System Trace" --time-limit 10s --launch -- ./my-metal-app

# Output to specific path
xctrace record --template "Time Profiler" --output /tmp/recording.trace --launch -- ./my-app

# Set environment variables for launched process
xctrace record --template "Time Profiler" --env METAL_DEVICE_WRAPPER_TYPE=1 --launch -- ./my-app
```

**Profiling a specific code section**: Use `os_signpost` to mark the region of interest, then filter the trace by time range:
```swift
import os
let log = OSLog(subsystem: "com.myapp", category: .pointsOfInterest)
os_signpost(.begin, log: log, name: "HotPath")
// ... code to profile ...
os_signpost(.end, log: log, name: "HotPath")
```
Then record with `--instrument "os_signpost"` alongside the profiler template, use `profiler_signpost_intervals` to find the time range, and scope `profiler_cpu_samples`/`profiler_top_functions` to that window.

## Capturing GPU Traces

GPU traces (`.gputrace` bundles) capture the full Metal command stream — every dispatch, buffer binding, barrier, and optionally shader performance counters.

### Method 1: Environment Variables (simplest for CLI tools)

Set `MTL_CAPTURE_ENABLED=1` and `METAL_CAPTURE_ENABLED=1` to capture the entire Metal workload:

```bash
# Capture everything — produces a .gputrace in the current directory
MTL_CAPTURE_ENABLED=1 METAL_CAPTURE_ENABLED=1 ./my-metal-app

# Specify output path
MTL_CAPTURE_ENABLED=1 METAL_CAPTURE_ENABLED=1 \
  METAL_CAPTURE_OUTPUT_PATH=/tmp/my_capture.gputrace \
  ./my-metal-app
```

This captures from the first Metal command to program exit. Good for short-running compute workloads.

### Method 2: MTLCaptureManager (profile specific code sections)

For profiling a specific section of execution, add capture bracketing in the Metal code:

```swift
import Metal

func profileSection(device: MTLDevice, commandQueue: MTLCommandQueue) {
    let captureManager = MTLCaptureManager.shared()
    let descriptor = MTLCaptureDescriptor()
    descriptor.captureObject = device  // or commandQueue for narrower scope
    descriptor.destination = .gpuTraceDocument
    descriptor.outputURL = URL(fileURLWithPath: "/tmp/my_capture.gputrace")

    do {
        try captureManager.startCapture(with: descriptor)
    } catch {
        print("Capture failed to start: \(error)")
        return
    }

    // ... Metal work to profile ...
    // (encode commands, commit command buffers, wait for completion)

    captureManager.stopCapture()
}
```

For C/Objective-C Metal code:
```objc
#import <Metal/Metal.h>

// Start capture
MTLCaptureManager *mgr = [MTLCaptureManager sharedCaptureManager];
MTLCaptureDescriptor *desc = [[MTLCaptureDescriptor alloc] init];
desc.captureObject = device;  // or commandQueue
desc.destination = MTLCaptureDestinationGPUTraceDocument;
desc.outputURL = [NSURL fileURLWithPath:@"/tmp/my_capture.gputrace"];
NSError *err;
[mgr startCaptureWithDescriptor:desc error:&err];

// ... Metal work to profile ...

[mgr stopCapture];
```

**Important**: The output directory must exist and the `.gputrace` must not already exist (delete first if re-capturing).

### Method 3: Xcode (interactive)

1. Open the project in Xcode
2. Run with the GPU Frame Capture button (camera icon in debug bar)
3. Click "Capture GPU Workload" during execution
4. Save the capture as a `.gputrace` file

### Getting shader performance counters

The `profiler_gpu_counters` tool requires shader profiling data (streamData) which is only generated when the GPU trace is **replayed with profiling enabled** in Xcode:

1. Open the `.gputrace` in Xcode
2. Click the **Replay** button (with shader profiling enabled in settings)
3. Xcode writes `streamData` into the `.gputrace` bundle
4. Now `profiler_gpu_counters` can read occupancy, bandwidth, cache hit rates, etc.

## Available Tools

### CPU analysis (`.trace` files — Time Profiler, CPU Profiler, Metal System Trace)
- `profiler_open_trace` — Open trace and return metadata + table list.
- `profiler_cpu_samples` — Individual samples with **full stack traces**. Use `start_time_ns`/`end_time_ns` to scope to a time window.
- `profiler_top_functions` — Aggregated view: which functions consumed the most CPU. Also supports time-range filtering.

### Hang detection (requires Hangs instrument)
- `profiler_hangs` — Lists unresponsiveness intervals with type, duration, and thread.

### Signpost analysis (requires os_signpost instrument)
- `profiler_signpost_events` — Raw begin/end/event entries. Filter by subsystem, category, name.
- `profiler_signpost_intervals` — Matched begin+end pairs with durations.

### Generic table access (works with any `.trace`)
- `profiler_list_tables` — Discover all available schemas in the trace.
- `profiler_query_table` — Query any table by schema name. Use this for Metal GPU tables, thermal data, disk I/O, or anything not covered by the specialized tools above.

### Correlated CPU+GPU analysis (Metal System Trace `.trace` files)
- `profiler_correlated_timeline` — Time-bucketed view showing CPU and GPU activity side by side. Auto-detects execution phases: `CPU_BOUND`, `GPU_BOUND`, `BALANCED`, `PIPELINE_BUBBLE`, `IDLE`. Returns per-bucket breakdown plus summary with bottleneck classification.

### GPU trace analysis (`.gputrace` files)
- `profiler_gpu_open` — Structural overview: kernels, command buffers, encoders, dispatch/barrier counts.
- `profiler_gpu_timeline` — Detailed dispatch events with kernel names, threadgroup sizes, buffer bindings. Filter by kernel pattern, command buffer, or encoder.
- `profiler_gpu_dependencies` — Buffer hazard DAG with critical path analysis. View at dispatch, encoder, kernel, or command buffer scale.
- `profiler_gpu_counters` — Shader profiling counters (occupancy, bandwidth, cache rates). Summary stats or full time-series.
- `profiler_gpu_scheduling` — Scheduling overhead analysis. Identifies inter-encoder gaps (GPU idle between encoder submissions) and dispatch fusion candidates (many small dispatches that could be combined). Returns prioritized recommendations with estimated savings. Requires shader profiling data (streamData).
- `profiler_gpu_export_perfetto` — Export to `.pftrace` for visualization in ui.perfetto.dev.

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

### "Is my Metal code efficient?" (system-level)
1. Record with Metal System Trace: `xctrace record --template "Metal System Trace" --launch -- ./app`
2. `profiler_list_tables` to see available Metal tables
3. `profiler_query_table` with `metal-gpu-intervals` for GPU execution
4. `profiler_query_table` with `metal-driver-intervals` for driver overhead
5. `profiler_top_functions` to see CPU-side costs

### "What happened in this specific time window?"
1. Use `profiler_cpu_samples` (no limit) to find the time range of interest from timestamps
2. Call `profiler_top_functions(start_time_ns=X, end_time_ns=Y)` to see what was hot in that window
3. Call `profiler_cpu_samples(start_time_ns=X, end_time_ns=Y, limit=30)` for detailed stack traces

### "Optimize my Metal compute kernels" (GPU trace)
1. Capture a `.gputrace` (see [Capturing GPU Traces](#capturing-gpu-traces))
2. `profiler_gpu_open` to see kernel list, dispatch counts, command buffer structure
3. `profiler_gpu_scheduling` to find scheduling overhead — inter-encoder gaps and fusion candidates
4. `profiler_gpu_dependencies` at encoder or kernel scale to find the critical path
5. `profiler_gpu_timeline` with kernel_filter to inspect specific dispatches (threadgroup sizes, buffer bindings)
6. `profiler_gpu_counters` for shader performance metrics (requires Xcode replay)
7. `profiler_gpu_export_perfetto` to create a visual timeline for the user

### "Which GPU dispatches could run in parallel?"
1. `profiler_gpu_dependencies` with scale=dispatch to see the full buffer dependency DAG
2. Check the `critical_path_length` vs `total_dispatches` — large gap means parallelism exists
3. Look for `isolated_nodes` — dispatches with no dependencies that could overlap with anything
4. Use `profiler_gpu_dependencies` with scale=kernel to see which kernel types have cross-dependencies

### "Profile both CPU and GPU sides"
1. Record with Metal System Trace: `xctrace record --template "Metal System Trace" --time-limit 10s --launch -- ./app`
2. `profiler_correlated_timeline` on the `.trace` to see CPU and GPU activity side by side with auto-detected phases
3. Check the `summary.bottleneck` field — it tells you if the app is CPU_BOUND, GPU_BOUND, or BALANCED
4. For GPU-bound workloads: capture a `.gputrace` and use `profiler_gpu_scheduling` to find if encoder consolidation would help
5. For CPU-bound workloads: use `profiler_top_functions` to find the CPU hotspot (is it command encoding? shader compilation? data preparation?)

### "Should I combine my GPU operations?"
1. Capture a `.gputrace` with shader profiling data (see [Getting shader performance counters](#getting-shader-performance-counters))
2. `profiler_gpu_scheduling` — check `summary.inter_encoder_gap_pct`. If >5%, combining encoders will help
3. Check `encoder_gaps` for specific gaps and whether they're `COMBINABLE` (shared kernels) or `COMBINABLE_DIFFERENT_KERNELS` (different work but no CPU readback needed)
4. Check `fusion_candidates` for sequences of many small dispatches that could be batched into fewer, larger ones
5. Follow the `recommendations` — they're prioritized by potential savings

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
