# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyobjc-core",
#     "pyobjc-framework-Cocoa",
# ]
# ///
"""Extract timeline and resource usage from an Xcode .gputrace file.

Uses Apple's private GPUTools frameworks (via pyobjc) to read the Metal API
function stream and reconstruct which compute kernels are dispatched, with
what threadgroup sizes, and which buffers are bound.

Requirements:
  - macOS with Xcode installed (uses private frameworks from Xcode.app)
  - DYLD_FRAMEWORK_PATH is set automatically via re-exec if needed

Usage:
    uv run tools/gputrace_timeline.py /path/to/capture.gputrace

The function index → Metal API mapping was reverse-engineered from a compute-only
baspacho sparse linear algebra workload. Indices may change across Xcode versions.

See memory/gputrace-format.md for detailed format documentation.
"""
from __future__ import annotations

import ctypes
import logging
import os
import re
import struct
import sys
import warnings
from collections import defaultdict
from typing import Any

import objc  # type: ignore[import-untyped]
from Foundation import NSBundle, NSURL  # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)

# ---------------------------------------------------------------------------
# Apple private framework loading
# ---------------------------------------------------------------------------

SHARED_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"
_FRAMEWORK_NAMES = [
    "GPUToolsCore",
    "GPUTools",
    "GPUToolsPlatform",
    "GLToolsCore",
    "GPUToolsServices",
]


def _ensure_dyld_framework_path() -> None:
    """Re-exec with DYLD_FRAMEWORK_PATH if not set.

    dyld reads this variable at process startup to resolve @rpath references,
    so it must be set before any GPU framework is loaded. When missing, we
    set it and os.execv() to restart the process.
    """
    if os.environ.get("DYLD_FRAMEWORK_PATH") != SHARED_FW:
        os.environ["DYLD_FRAMEWORK_PATH"] = SHARED_FW
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _load_frameworks() -> None:
    """Load Apple private GPU frameworks in dependency order."""
    for name in _FRAMEWORK_NAMES:
        bundle = NSBundle.bundleWithPath_(f"{SHARED_FW}/{name}.framework")
        if bundle is not None:
            bundle.load()
    # System private framework
    sys_bundle = NSBundle.bundleWithPath_(
        "/System/Library/PrivateFrameworks/GPUToolsCapture.framework"
    )
    if sys_bundle is not None:
        sys_bundle.load()


_load_frameworks()

DYCaptureArchive = objc.lookUpClass("DYCaptureArchive")
DYFunctionTracer = objc.lookUpClass("DYFunctionTracer")

# ---------------------------------------------------------------------------
# Function index → Metal API name mapping
# ---------------------------------------------------------------------------
# These are negative signed int32 indices into the interpose function table.
# Determined by correlating argument patterns from a compute workload.
# May change across Xcode versions.

FUNC_NAMES: dict[int, str] = {
    -16383: "endEncoding (blit, GPUTools tracking)",
    -16377: "blit.copy(from:to:)",
    -16376: "endEncoding (blit)",
    -16371: "setPurgeableState: (GPUTools-inserted)",
    -16370: "endEncoding (GPUTools blit encoder)",
    -16367: "setPurgeableState: (user)",
    -16363: "MTLCommandBuffer.commit",
    -16361: "MTLCommandBuffer.waitUntilCompleted",
    -16356: "addCompletedHandler:",
    -16355: "MTLCommandBuffer.computeCommandEncoder",
    -16354: "makeBlitCommandEncoder",
    -16352: "MTLCommandQueue.commandBuffer",
    -16343: "MTLDevice.makeBuffer(length:options:)",
    -16338: "setComputePipelineState:",
    -16337: "setBytes:length:atIndex:",
    -16336: "setBuffer:offset:atIndex:",
    -16327: "dispatchThreadgroups:threadsPerThreadgroup:",
    -16325: "endEncoding (compute)",
    -16316: "makeComputePipelineState(function:)",
    -16314: "makeBuffer(length:) variant",
    -16313: "makeBuffer(bytes:length:options:)",
    -16305: "makeLibrary(source:options:)",
    -16299: "newComputePipelineStateWithFunction:error:",
    -16290: "makeFunction(name:)",
    -16227: "setBytes:length:atIndex: (inline)",
    -16078: "dispatchThreads:threadsPerThreadgroup:",
    -16067: "supportsFamily:",
    -16009: "memoryBarrierWithScope:",
    -16008: "memoryBarrier(resources:)",
    -15996: "makeSharedEvent()",
    -15990: "encodeSignalEvent / event.notify()",
    -15973: "event cleanup",
    -15736: "newLibraryWithURL:",
    -15422: "MTLSharedEvent.notifyListener:atValue:",
    -10228: "internal.bufferDidModify",
    -10223: "internal.bufferContents",
    -10203: "internal.resourceID",
    -10191: "internal.bufferGPUAddress",
    -10186: "internal.bufferCompleted",
}

# Function indices that need trace-text argument parsing.
_TRACE_INDICES = frozenset({
    -16290, -16299, -16352, -16355, -16338,
    -16337, -16336, -16327, -16078, -16009,
})


def _parse_hex_addrs(trace: str) -> list[int]:
    """Extract hex addresses (0x...) from DYFunctionTracer trace text."""
    return [int(m.group(1), 16) for m in re.finditer(r"0x([0-9a-fA-F]+)l?", trace)]


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------


def read_gputrace(path: str) -> dict[str, Any] | None:
    """Read a .gputrace file and extract timeline events.

    Returns a dict with keys:
      metadata       - capture session metadata
      total_functions - total Metal API calls in the stream
      events         - list of timeline events (set_pipeline, dispatch)
      kernels        - {function_addr: kernel_name}
      pipelines      - {pipeline_addr: kernel_name}
      command_buffers - list of command buffer groups with their dispatches
    """
    url = NSURL.fileURLWithPath_(path)
    archive = DYCaptureArchive.alloc().initWithURL_options_error_(url, 0, None)
    if archive is None:
        log.error("Failed to open: %s", path)
        return None

    # Capture metadata
    metadata: dict[str, Any] = {}
    for key in [
        "DYCaptureSession.graphics_api",
        "DYCaptureEngine.captured_frames_count",
        "DYCaptureSession.nativePointerSize",
    ]:
        val = archive.metadataValueForKey_(key)
        if val is not None:
            metadata[key] = val

    # Open unsorted-capture for timeline (original call) order
    capture_file = archive.openFileWithFilename_error_("unsorted-capture", None)
    fstream = capture_file.openFunctionStream_(None)

    tracer = DYFunctionTracer.alloc().init()
    tracer.setCompact_(True)
    tracer.setNativePointerSize_(8)

    # --- State tracking ---
    kernels: dict[int, str] = {}      # function_addr → kernel_name
    pipelines: dict[int, str] = {}    # pipeline_addr → kernel_name
    last_created_pipeline: str | None = None
    current_pipeline_name: str | None = None
    buffers_bound: dict[int, int] = {}

    # --- Results ---
    events: list[dict[str, Any]] = []
    command_buffers: list[dict[str, Any]] = []
    current_cb_dispatches: list[dict[str, Any]] = []
    compute_encoders: list[dict[str, Any]] = []
    current_encoder_dispatches: list[dict[str, Any]] = []
    encoder_counter: int = 0
    current_encoder_idx: int = -1
    current_cb_idx: int = -1

    func_idx = 0
    while True:
        func_ptr = fstream.readFunction()
        if func_ptr is None:
            break

        ptr_addr = func_ptr.pointerAsInteger
        raw_header = ctypes.string_at(ptr_addr, 8)
        idx = struct.unpack_from("<i", raw_header, 0)[0]

        if idx not in FUNC_NAMES:
            func_idx += 1
            continue

        # Decode trace text only for indices that need argument parsing
        trace: str | None = None
        if idx in _TRACE_INDICES:
            trace = str(tracer.traceFunction_error_(func_ptr, None))

        # ---- Kernel / pipeline creation ----

        if idx == -16290:  # MTLLibrary.newFunctionWithName:
            if trace and "=" in trace and '"' in trace:
                ret_str = trace.split("=")[0].strip()
                kernel_name = trace.split('"')[1]
                if ret_str.startswith("0x"):
                    func_addr = int(ret_str.rstrip("l"), 16)
                    kernels[func_addr] = kernel_name

        elif idx == -16299:  # newComputePipelineStateWithFunction:error:
            if trace and "=" in trace:
                ret_str = trace.split("=")[0].strip()
                if ret_str.startswith("0x"):
                    pipeline_addr = int(ret_str.rstrip("l"), 16)
                    args_part = trace.split("(", 1)[1] if "(" in trace else trace
                    for a in _parse_hex_addrs(args_part):
                        if a in kernels:
                            pipelines[pipeline_addr] = kernels[a]
                            last_created_pipeline = pipelines[pipeline_addr]
                            break
                    else:
                        pipelines[pipeline_addr] = f"pipeline_0x{pipeline_addr:x}"
                        last_created_pipeline = pipelines[pipeline_addr]

        elif idx == -15996:  # makeSharedEvent()
            # This is an MTLSharedEvent, not a pipeline descriptor.
            # Skip processing—shared events don't affect pipeline dispatch mapping.
            pass

        # ---- Command buffer lifecycle ----

        elif idx == -16352:  # MTLCommandQueue.commandBuffer
            if trace and "=" in trace:
                ret_str = trace.split("=")[0].strip()
                if ret_str.startswith("0x"):
                    current_cb_dispatches = []
                    current_cb_idx = len(command_buffers)

        elif idx == -16363:  # MTLCommandBuffer.commit
            # Close any open encoder before closing the command buffer
            if current_encoder_dispatches:
                compute_encoders.append({
                    "encoder_idx": current_encoder_idx,
                    "command_buffer_idx": current_cb_idx,
                    "dispatches": list(current_encoder_dispatches),
                })
                current_encoder_dispatches = []
                current_encoder_idx = -1
            if current_cb_dispatches:
                command_buffers.append({
                    "func_idx": func_idx,
                    "dispatches": list(current_cb_dispatches),
                })
            current_cb_dispatches = []

        # ---- Encoder lifecycle ----

        elif idx == -16355:  # MTLCommandBuffer.computeCommandEncoder
            # Close any previous open encoder
            if current_encoder_dispatches:
                compute_encoders.append({
                    "encoder_idx": current_encoder_idx,
                    "command_buffer_idx": current_cb_idx,
                    "dispatches": list(current_encoder_dispatches),
                })
                current_encoder_dispatches = []
            current_encoder_idx = encoder_counter
            encoder_counter += 1
            # Don't reset pipeline — it carries over because encoder creation
            # isn't always explicit in the unsorted-capture stream.
            buffers_bound = {}

        elif idx in (-16325, -16370):  # endEncoding
            if current_encoder_dispatches:
                compute_encoders.append({
                    "encoder_idx": current_encoder_idx,
                    "command_buffer_idx": current_cb_idx,
                    "dispatches": list(current_encoder_dispatches),
                })
                current_encoder_dispatches = []
                current_encoder_idx = -1

        # ---- Pipeline state setting ----

        elif idx == -16338:  # setComputePipelineState: (explicit)
            # The MTSP record doesn't store the pipeline address in its
            # Argument data. Use last_created_pipeline as a heuristic when
            # this call appears right after a pipeline creation.
            if last_created_pipeline:
                current_pipeline_name = last_created_pipeline
                events.append({
                    "type": "set_pipeline",
                    "kernel": current_pipeline_name,
                    "index": func_idx,
                })
            last_created_pipeline = None  # consumed

        elif idx == -16337:  # setBytes:length:atIndex:
            # KEY INSIGHT: the pipeline state address often appears as a hex
            # argument in setBytes calls at the start of a dispatch group.
            # The encoder/receiver is rendered as decimal ("31488869312ull"),
            # while the pipeline address is hex ("0x1050d4e50l").
            if trace:
                for data_addr in _parse_hex_addrs(trace):
                    if data_addr in pipelines:
                        current_pipeline_name = pipelines[data_addr]
                        events.append({
                            "type": "set_pipeline",
                            "kernel": current_pipeline_name,
                            "pipeline_addr": data_addr,
                            "index": func_idx,
                        })
                        break
                    # Pre-capture pipeline: address isn't in the
                    # pipelines dict but looks like a pipeline pointer
                    # (not the encoder's decimal address).  Label it
                    # the same way Xcode does.  Don't add to pipelines
                    # dict — a later newComputePipelineState may
                    # register the real name for this address.
                    if data_addr > 0xFFFF:
                        name = f"Compute Pipeline 0x{data_addr:x}"
                        current_pipeline_name = name
                        events.append({
                            "type": "set_pipeline",
                            "kernel": name,
                            "pipeline_addr": data_addr,
                            "index": func_idx,
                        })
                        break

        # ---- Buffer bindings ----

        elif idx == -16336:  # setBuffer:offset:atIndex:
            # trace: (null)(encoder_decimal, 0xBUFFER, 0ul, INDEXul)
            if trace:
                addrs = _parse_hex_addrs(trace)
                if addrs:
                    buffer_addr = addrs[0]
                    m = re.search(r"(\d+)ul\)$", trace)
                    buf_index = int(m.group(1)) if m else -1
                    buffers_bound[buf_index] = buffer_addr

        # ---- Memory barriers ----

        elif idx == -16009:  # memoryBarrierWithScope:
            event: dict[str, Any] = {
                "type": "barrier",
                "scope": "buffers",
                "index": func_idx,
                "encoder_idx": current_encoder_idx,
            }
            events.append(event)

        # ---- Dispatch ----

        elif idx in (-16327, -16078):
            # Auto-create encoder if none is active (trace may omit
            # explicit computeCommandEncoder calls)
            if current_encoder_idx == -1:
                current_encoder_idx = encoder_counter
                encoder_counter += 1
            dispatch_type = "threadgroups" if idx == -16327 else "threads"

            threadgroups: tuple[int, ...] | None = None
            threads_per: tuple[int, ...] | None = None

            if trace and idx == -16327:
                struct_matches = re.findall(
                    r"\{(\d+)ul,\s*(\d+)ul,\s*(\d+)ul\}", trace
                )
                if len(struct_matches) >= 2:
                    threadgroups = tuple(int(x) for x in struct_matches[0])
                    threads_per = tuple(int(x) for x in struct_matches[1])
            elif trace and idx == -16078:
                struct_matches = re.findall(
                    r"\{(\d+)ul,\s*(\d+)ul,\s*(\d+)ul\}", trace
                )
                if len(struct_matches) >= 2:
                    threadgroups = tuple(int(x) for x in struct_matches[0])
                    threads_per = tuple(int(x) for x in struct_matches[1])

            event: dict[str, Any] = {
                "type": "dispatch",
                "dispatch_type": dispatch_type,
                "kernel": current_pipeline_name or "unknown",
                "index": func_idx,
                "buffers_bound": dict(buffers_bound),
                "encoder_idx": current_encoder_idx,
            }
            if threadgroups:
                event["threadgroups"] = threadgroups
            if threads_per:
                event["threads_per_threadgroup"] = threads_per

            events.append(event)
            current_cb_dispatches.append(event)
            current_encoder_dispatches.append(event)
            buffers_bound = {}

        func_idx += 1

    return {
        "metadata": metadata,
        "total_functions": func_idx,
        "events": events,
        "kernels": kernels,
        "pipelines": pipelines,
        "command_buffers": command_buffers,
        "compute_encoders": compute_encoders,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    _ensure_dyld_framework_path()
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/baspacho_ffi.gputrace"
    result = read_gputrace(path)

    if result is None:
        sys.exit(1)

    print(f"=== GPU Trace Timeline: {path} ===")
    print(f"Total Metal API calls: {result['total_functions']}")
    print(f"Metadata: {result['metadata']}")

    print(f"\nKernels found: {len(result['kernels'])}")
    for _addr, name in sorted(result["kernels"].items(), key=lambda x: x[1]):
        print(f"  {name}")

    print(f"\nPipelines found: {len(result['pipelines'])}")
    for addr, name in sorted(result["pipelines"].items(), key=lambda x: x[1]):
        print(f"  0x{addr:x} -> {name}")

    dispatch_events = [e for e in result["events"] if e["type"] == "dispatch"]
    barrier_events = [e for e in result["events"] if e["type"] == "barrier"]
    print(f"\nDispatch calls: {len(dispatch_events)}")
    print(f"Memory barriers: {len(barrier_events)}")

    kernel_counts: dict[str, int] = defaultdict(int)
    for evt in dispatch_events:
        kernel_counts[evt["kernel"]] += 1

    print("\nDispatches per kernel:")
    for kernel, count in sorted(kernel_counts.items(), key=lambda x: -x[1]):
        print(f"  {kernel}: {count}")

    print("\nFirst 20 dispatches:")
    for evt in dispatch_events[:20]:
        tg = evt.get("threadgroups", "?")
        tpt = evt.get("threads_per_threadgroup", "")
        bufs = len(evt.get("buffers_bound", {}))
        print(
            f"  [{evt['index']:5d}] {evt['kernel']:45s} "
            f"tg={tg} tpt={tpt} bufs={bufs}"
        )

    print(f"\nCompute encoders: {len(result['compute_encoders'])}")
    for i, enc in enumerate(result["compute_encoders"][:10]):
        print(
            f"  Encoder#{enc['encoder_idx']}: {len(enc['dispatches'])} dispatches "
            f"(CB#{enc['command_buffer_idx']})"
        )

    print(f"\nCommand buffers: {len(result['command_buffers'])}")
    for i, cb in enumerate(result["command_buffers"][:10]):
        print(
            f"  CB#{i}: {len(cb['dispatches'])} dispatches "
            f"(func #{cb['func_idx']})"
        )
        cb_kernels: dict[str, int] = defaultdict(int)
        for d in cb["dispatches"]:
            cb_kernels[d["kernel"]] += 1
        for k, c in sorted(cb_kernels.items(), key=lambda x: -x[1]):
            print(f"    {k}: {c}")


if __name__ == "__main__":
    main()
