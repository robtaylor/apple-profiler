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
import glob as globmod
import json
import logging
import os
import re
import struct
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import objc  # type: ignore[import-untyped]
from Foundation import NSURL, NSBundle  # type: ignore[import-untyped]

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
    compute_encoders: list[dict[str, Any]] = []
    current_encoder_dispatches: list[dict[str, Any]] = []
    encoder_counter: int = 0
    current_encoder_idx: int = -1
    current_encoder_addr: str = ""
    # Track multiple concurrent CBs (unsorted stream interleaves them).
    # Keyed by CB address (off16 from -16352/-16355/-16363).
    active_cbs: dict[str, dict[str, Any]] = {}
    current_cb_addr: str = ""  # points to the currently-active CB key

    func_idx = 0
    while True:
        func_ptr = fstream.readFunction()
        if func_ptr is None:
            break

        ptr_addr = func_ptr.pointerAsInteger
        raw_header = ctypes.string_at(ptr_addr, 24)
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
            # -16352 fires per-encoder, not just per-CB.
            # off16 = CB address, return value = encoder address.
            cb_addr_16352 = struct.unpack_from("<Q", raw_header, 16)[0]
            cb_addr_str = f"0x{cb_addr_16352:x}" if cb_addr_16352 else ""
            if cb_addr_str:
                current_cb_addr = cb_addr_str
                if cb_addr_str not in active_cbs:
                    active_cbs[cb_addr_str] = {
                        "cb_idx": len(command_buffers) + len(active_cbs),
                        "addr": cb_addr_str,
                        "dispatches": [],
                    }
            # Capture encoder address from return value
            if trace and "=" in trace:
                enc_ret = trace.split("=")[0].strip().rstrip("l")
                if enc_ret.startswith("0x"):
                    current_encoder_addr = enc_ret

        elif idx == -16363:  # MTLCommandBuffer.commit
            # off16 = CB address being committed
            commit_addr = struct.unpack_from("<Q", raw_header, 16)[0]
            commit_key = f"0x{commit_addr:x}" if commit_addr else current_cb_addr
            # Resolve the CB from active_cbs
            cb_info = active_cbs.pop(commit_key, None)
            if cb_info is None and current_cb_addr:
                cb_info = active_cbs.pop(current_cb_addr, None)
            # Close any open encoder belonging to this CB
            if current_encoder_dispatches:
                cb_idx = cb_info["cb_idx"] if cb_info else -1
                compute_encoders.append({
                    "encoder_idx": current_encoder_idx,
                    "command_buffer_idx": cb_idx,
                    "addr": current_encoder_addr,
                    "dispatches": list(current_encoder_dispatches),
                })
                current_encoder_dispatches = []
                current_encoder_idx = -1
                current_encoder_addr = ""
            if cb_info and cb_info["dispatches"]:
                command_buffers.append({
                    "func_idx": func_idx,
                    "addr": cb_info["addr"],
                    "dispatches": cb_info["dispatches"],
                })
            if commit_key == current_cb_addr:
                current_cb_addr = ""

        # ---- Encoder lifecycle ----

        elif idx == -16355:  # MTLCommandBuffer.computeCommandEncoder
            # Close any previous open encoder
            if current_encoder_dispatches:
                cb_idx = active_cbs[current_cb_addr]["cb_idx"] if current_cb_addr in active_cbs else -1
                compute_encoders.append({
                    "encoder_idx": current_encoder_idx,
                    "command_buffer_idx": cb_idx,
                    "addr": current_encoder_addr,
                    "dispatches": list(current_encoder_dispatches),
                })
                current_encoder_dispatches = []
            current_encoder_idx = encoder_counter
            encoder_counter += 1
            # off16 = CB address (confirms/sets current CB)
            # -16355 has no return value — encoder addr comes from
            # the preceding -16352's return value (already captured).
            cb_addr_16355 = struct.unpack_from("<Q", raw_header, 16)[0]
            if cb_addr_16355:
                cb_addr = f"0x{cb_addr_16355:x}"
                current_cb_addr = cb_addr
                if cb_addr not in active_cbs:
                    active_cbs[cb_addr] = {
                        "cb_idx": len(command_buffers) + len(active_cbs),
                        "addr": cb_addr,
                        "dispatches": [],
                    }
            # Don't reset pipeline — it carries over because encoder creation
            # isn't always explicit in the unsorted-capture stream.
            buffers_bound = {}

        elif idx in (-16325, -16370):  # endEncoding
            if current_encoder_dispatches:
                cb_idx = active_cbs[current_cb_addr]["cb_idx"] if current_cb_addr in active_cbs else -1
                compute_encoders.append({
                    "encoder_idx": current_encoder_idx,
                    "command_buffer_idx": cb_idx,
                    "addr": current_encoder_addr,
                    "dispatches": list(current_encoder_dispatches),
                })
                current_encoder_dispatches = []
                current_encoder_idx = -1
                current_encoder_addr = ""

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
            if current_cb_addr in active_cbs:
                active_cbs[current_cb_addr]["dispatches"].append(event)
            current_encoder_dispatches.append(event)
            buffers_bound = {}

        func_idx += 1

    # Flush any open encoder
    if current_encoder_dispatches:
        cb_idx = active_cbs[current_cb_addr]["cb_idx"] if current_cb_addr in active_cbs else -1
        compute_encoders.append({
            "encoder_idx": current_encoder_idx,
            "command_buffer_idx": cb_idx,
            "addr": current_encoder_addr,
            "dispatches": list(current_encoder_dispatches),
        })

    # Flush uncommitted CBs (stream may end before final commits)
    for cb_addr, cb_info in active_cbs.items():
        if cb_info["dispatches"]:
            command_buffers.append({
                "func_idx": -1,  # no commit event
                "addr": cb_info["addr"],
                "dispatches": cb_info["dispatches"],
            })

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
# GPU performance counter extraction
# ---------------------------------------------------------------------------

# Additional framework path for shader profiler
_GPU_DEBUGGER_PLUGIN = (
    "/Applications/Xcode.app/Contents/PlugIns/GPUDebugger.ideplugin"
    "/Contents/Frameworks"
)

_PROFILER_FRAMEWORK_NAMES = [
    "GPUToolsShaderProfiler",
]


def _load_profiler_frameworks() -> bool:
    """Load GTShaderProfiler + dependencies for counter extraction.

    Returns True if all frameworks loaded successfully.
    """
    # Shared framework dependencies
    for name in _PROFILER_FRAMEWORK_NAMES:
        bundle = NSBundle.bundleWithPath_(f"{SHARED_FW}/{name}.framework")
        if bundle is not None:
            bundle.load()

    # GTShaderProfiler from GPUDebugger plugin
    gt_path = f"{_GPU_DEBUGGER_PLUGIN}/GTShaderProfiler.framework"
    bundle = NSBundle.bundleWithPath_(gt_path)
    if bundle is None:
        log.warning("GTShaderProfiler.framework not found at %s", gt_path)
        return False
    bundle.load()
    return True


_XCODE_PROFILING_DIR = "/private/tmp/com.apple.gputools.profiling"


def _find_stream_data(gputrace_path: str) -> str | None:
    """Find streamData for a gputrace, checking bundle and Xcode temp dir.

    Searches:
      1. Inside the gputrace bundle: <bundle>/*.gpuprofiler_raw/streamData
      2. Xcode profiling temp dir: /private/tmp/com.apple.gputools.profiling/
         using the gputrace stem name to match (e.g. sprux_ffi_stream.gpuprofiler_raw)
    """
    # 1. Inside the gputrace bundle
    pattern = os.path.join(gputrace_path, "*.gpuprofiler_raw", "streamData")
    matches = globmod.glob(pattern)
    if matches:
        return matches[0]

    # 2. Xcode profiling temp dir — match by gputrace stem name
    stem = os.path.splitext(os.path.basename(gputrace_path))[0]
    candidates = [
        f"{stem}_stream.gpuprofiler_raw",  # e.g. sprux_ffi_stream.gpuprofiler_raw
        f"{stem}.gpuprofiler_raw",
    ]
    for candidate in candidates:
        sd = os.path.join(_XCODE_PROFILING_DIR, candidate, "streamData")
        if os.path.exists(sd):
            return sd

    # 3. Glob the temp dir for any match containing the stem
    if os.path.isdir(_XCODE_PROFILING_DIR):
        pattern = os.path.join(
            _XCODE_PROFILING_DIR, f"*{stem}*.gpuprofiler_raw", "streamData",
        )
        matches = globmod.glob(pattern)
        if matches:
            return matches[0]

    return None


_JXA_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "xcode_gputrace_automation.js")


def _run_jxa(action: str, *args: str) -> dict[str, Any] | str | None:
    """Run the JXA automation script with an action and args, parse JSON result."""
    cmd = ["osascript", "-l", "JavaScript", _JXA_SCRIPT_PATH, action, *args]
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, timeout=15,
        )
        text = out.decode().strip()
        if text.startswith("{"):
            return json.loads(text)
        return text
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        out_text = ""
        if hasattr(e, "output") and e.output:
            out_text = e.output.decode()
        log.warning("JXA %s failed: %s %s", action, e, out_text)
        return None


def _snapshot_stream_data() -> dict[str, float]:
    """Snapshot current streamData files and their mtimes in the profiling dir."""
    profiling_dir = Path(_XCODE_PROFILING_DIR)
    mtimes: dict[str, float] = {}
    if profiling_dir.exists():
        for d in profiling_dir.glob("*.gpuprofiler_raw"):
            sd = d / "streamData"
            if sd.exists():
                mtimes[str(sd)] = sd.stat().st_mtime
    return mtimes


def _check_new_stream_data(before: dict[str, float]) -> str | None:
    """Check for new or updated streamData file since the snapshot."""
    profiling_dir = Path(_XCODE_PROFILING_DIR)
    if not profiling_dir.exists():
        return None
    for d in profiling_dir.glob("*.gpuprofiler_raw"):
        sd = d / "streamData"
        if not sd.exists() or sd.stat().st_size == 0:
            continue
        sd_str = str(sd)
        cur_mtime = sd.stat().st_mtime
        is_new = sd_str not in before
        is_updated = sd_str in before and cur_mtime > before[sd_str]
        if is_new or is_updated:
            return sd_str
    return None


def _replay_gputrace(gputrace_path: str, timeout: int = 120) -> str | None:
    """Open gputrace in Xcode, click Replay, wait for profiling to complete.

    Full lifecycle: close existing window → open fresh → enable profiling →
    click Replay → monitor Activity View + filesystem → close window.

    Returns path to streamData if successful, None otherwise.
    """
    abs_path = os.path.abspath(gputrace_path)
    stem = os.path.splitext(os.path.basename(gputrace_path))[0]

    # 1. Close existing gputrace window if open
    close_result = _run_jxa("close-window", stem)
    if isinstance(close_result, dict) and close_result.get("closed"):
        log.info("Closed existing gputrace window for %s", stem)
        time.sleep(0.5)

    # 2. Open gputrace in Xcode
    log.info("Opening %s in Xcode for replay...", abs_path)
    try:
        subprocess.run(
            ["open", "-g", "-a", "Xcode", abs_path], check=True, timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("Failed to open gputrace in Xcode: %s", e)
        return None

    # 3. Wait for window to appear and click Replay (poll up to 15s)
    replayed = False
    for attempt in range(30):
        time.sleep(0.5)
        result = _run_jxa("ensure-replay", stem)
        if isinstance(result, dict) and result.get("replayed"):
            profiled = result.get("profiled", False)
            log.info(
                "Replay started (profiling %s)",
                "enabled" if profiled else "NOT confirmed",
            )
            replayed = True
            break
        # Log the specific error for debugging
        if isinstance(result, dict) and result.get("error") and attempt == 29:
            log.warning("ensure-replay error: %s", result["error"])

    if not replayed:
        log.warning("Failed to find and click Replay within 15s")
        return None

    # 4. Snapshot existing streamData files
    before = _snapshot_stream_data()

    # 5. Poll for completion: Activity View + filesystem
    # When Xcode finishes profiling, the Activity View changes to
    # "Debugging GPU Workload" — at that point we close the window and
    # then wait for streamData to appear on the filesystem.
    window_closed = False
    for i in range(timeout):
        time.sleep(1)

        # Check filesystem for new/updated streamData
        new_sd = _check_new_stream_data(before)
        if new_sd:
            sd_path = Path(new_sd)
            log.info(
                "Profiling complete after %ds: %s (%d bytes)",
                i + 1, new_sd, sd_path.stat().st_size,
            )
            if not window_closed:
                _run_jxa("close-window", stem)
            return new_sd

        # Check Activity View for profiling completion
        status = _run_jxa("poll-activity")
        if isinstance(status, dict):
            status_text = status.get("status", "")
            # "Debugging GPU Workload" means replay+profiling finished
            # and Xcode entered the GPU debugger — close the window
            if not window_closed and "Debugging GPU" in status_text:
                log.info(
                    "Xcode entered GPU debugger after %ds — closing window",
                    i + 1,
                )
                _run_jxa("close-window", stem)
                window_closed = True
            elif i % 10 == 9:
                log.info("  ... %ds — Xcode: %s", i + 1, status_text)

    log.warning("Profiling did not complete within %ds", timeout)
    if not window_closed:
        _run_jxa("close-window", stem)
    return None


# Known MIO timeline counter names (discovered via framework binary strings).
# These are the human-readable names accepted by GTMioTimelineCounters.counterForName:.
_MIO_COUNTER_NAMES: list[str] = [
    "AF Bandwidth",
    "AF Peak Bandwidth",
    "AF Peak Read Bandwidth",
    "AF Peak Write Bandwidth",
    "AF Read Bandwidth",
    "AF Write Bandwidth",
    "CompressionRatioTextureMemoryRead",
    "L2 Bandwidth",
    "L2 Cache Limiter",
    "L2 Cache Utilization",
    "MMU Limiter",
    "MMU Utilization",
    "Texture Cache Limiter",
    "Texture Cache Utilization",
    "Texture Read Limiter",
    "Texture Read Utilization",
    "Texture Write Limiter",
    "Texture Write Utilization",
    "TextureFilteringLimiter",
]

# LLVM helper path required by GTShaderProfilerStreamDataProcessor
_LLVM_HELPER_PATH = (
    "/Applications/Xcode.app/Contents/Developer/Platforms/"
    "MacOSX.platform/Developer/Library/GPUToolsPlatform/PlugIns/GTLLVMHelper"
)


def _extract_mio_counter(
    counter: Any,
    sample_count: int,
) -> tuple[list[int], list[float]]:
    """Extract timestamps and values from a GTMioCounterData object.

    The timestamps() and values() methods return raw C pointers (uint64*
    and double*) that pyobjc cannot bridge. We use ctypes objc_msgSend
    to get the pointers and memmove to copy the data.
    """
    libobjc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.dylib")
    sel_registerName = libobjc.sel_registerName
    sel_registerName.restype = ctypes.c_void_p
    sel_registerName.argtypes = [ctypes.c_char_p]

    # Pointer-returning objc_msgSend
    objc_msgSend_ptr = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    )(("objc_msgSend", libobjc))

    counter_ptr = objc.pyobjc_id(counter)

    # Timestamps: uint64 array
    sel_ts = sel_registerName(b"timestamps")
    ts_raw = objc_msgSend_ptr(counter_ptr, sel_ts)
    assert ts_raw, "timestamps() returned NULL"

    ts_buf = (ctypes.c_uint64 * sample_count)()
    ctypes.memmove(ts_buf, ts_raw, sample_count * 8)
    timestamps = [int(ts_buf[i]) for i in range(sample_count)]

    # Values: double array
    sel_vals = sel_registerName(b"values")
    vals_raw = objc_msgSend_ptr(counter_ptr, sel_vals)
    assert vals_raw, "values() returned NULL"

    vals_buf = (ctypes.c_double * sample_count)()
    ctypes.memmove(vals_buf, vals_raw, sample_count * 8)
    values = [float(vals_buf[i]) for i in range(sample_count)]

    return timestamps, values


def read_gputrace_counters(
    gputrace_path: str,
    *,
    replay: bool = False,
    replay_timeout: int = 120,
) -> dict[str, Any] | None:
    """Extract derived GPU performance counter samples from streamData.

    Uses GTShaderProfilerStreamDataProcessor → mioData() → overlapping
    timeline → counterForName: to extract per-counter time-series data.

    Searches for streamData in the gputrace bundle and Xcode's temp
    profiling directory. If not found and replay=True, opens the
    gputrace in Xcode and clicks Replay to trigger shader profiling.

    Requires GTShaderProfiler.framework (loaded from GPUDebugger.ideplugin).
    Returns None if streamData is missing or frameworks unavailable.

    Returns a dict with keys:
      counter_names  - list of counter name strings
      num_samples    - number of periodic samples
      timestamps_ns  - list of uint64 timestamps (MIO timeline ticks)
      samples        - list of lists: samples[i][j] = float64 value for
                       sample i, counter j
    """
    stream_path = _find_stream_data(gputrace_path)

    if stream_path is not None:
        log.info("Found existing streamData: %s", stream_path)

    if stream_path is None and replay:
        log.info("No existing streamData — triggering Xcode replay...")
        stream_path = _replay_gputrace(gputrace_path, timeout=replay_timeout)

    if stream_path is None:
        log.info(
            "No streamData found. Use --replay to trigger Xcode replay, "
            "or replay manually in Xcode with shader profiling enabled.",
        )
        return None

    # Load profiler frameworks
    if not _load_profiler_frameworks():
        return None

    # Import ObjC classes
    from Foundation import NSData, NSKeyedUnarchiver, NSSet  # type: ignore[import-untyped]

    try:
        GTShaderProfilerStreamData = objc.lookUpClass("GTShaderProfilerStreamData")
        GTMutableShaderProfilerStreamData = objc.lookUpClass(
            "GTMutableShaderProfilerStreamData",
        )
        GTShaderProfilerStreamDataProcessor = objc.lookUpClass(
            "GTShaderProfilerStreamDataProcessor",
        )
    except objc.nosuchclass_error as e:
        log.warning("Required ObjC class not found: %s", e)
        return None

    # Read and unarchive streamData
    data = NSData.dataWithContentsOfFile_(stream_path)
    if data is None:
        log.warning("Failed to read streamData at %s", stream_path)
        return None

    allowlist = NSSet.setWithArray_([
        GTShaderProfilerStreamData,
        GTMutableShaderProfilerStreamData,
        objc.lookUpClass("NSMutableArray"),
        objc.lookUpClass("NSMutableDictionary"),
        objc.lookUpClass("NSMutableData"),
        objc.lookUpClass("NSArray"),
        objc.lookUpClass("NSDictionary"),
        objc.lookUpClass("NSData"),
        objc.lookUpClass("NSString"),
        objc.lookUpClass("NSNumber"),
    ])

    result = NSKeyedUnarchiver.unarchivedObjectOfClasses_fromData_error_(
        allowlist, data, None,
    )
    sd = result[0] if isinstance(result, tuple) else result
    if sd is None:
        log.warning("Failed to unarchive streamData")
        return None

    # Process via GTShaderProfilerStreamDataProcessor → MIO data path
    if not os.path.exists(_LLVM_HELPER_PATH):
        log.warning("LLVM helper not found at %s", _LLVM_HELPER_PATH)
        return None

    log.info("Processing streamData via MIO pipeline (this may take a moment)...")
    processor = GTShaderProfilerStreamDataProcessor.alloc() \
        .initWithStreamData_llvmHelperPath_(sd, _LLVM_HELPER_PATH)

    # Suppress noisy GTLLVMHelper stderr/stdout (LLVM warnings, GPU core info)
    saved_stderr = os.dup(2)
    saved_stdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.dup2(devnull, 1)
    try:
        processor.processStreamData()
        processor.waitUntilFinished()
    finally:
        os.dup2(saved_stderr, 2)
        os.dup2(saved_stdout, 1)
        os.close(saved_stderr)
        os.close(saved_stdout)
        os.close(devnull)

    mio = processor.mioData()
    if mio is None:
        log.warning("MIO data is None — processing may have failed")
        return None

    mio.loadTimeline()
    timeline = mio.overlappingTimeline()
    if timeline is None:
        log.warning("No overlapping timeline in MIO data")
        return None

    if timeline.profiledState() != 2:
        if replay:
            log.info(
                "streamData has profiledState=%d (not profiled) "
                "— triggering Xcode replay...",
                timeline.profiledState(),
            )
            new_path = _replay_gputrace(gputrace_path, timeout=replay_timeout)
            if new_path is not None:
                return read_gputrace_counters(
                    gputrace_path, replay=False,
                    replay_timeout=replay_timeout,
                )
        log.info(
            "profiledState=%d — no counter data. "
            "Replay with shader profiling enabled.",
            timeline.profiledState(),
        )
        return None

    tc = timeline.timelineCounters()
    if tc is None:
        log.warning("No timeline counters available")
        return None

    # Discover available counters by probing known names
    counter_names: list[str] = []
    counter_objects: list[Any] = []
    for name in _MIO_COUNTER_NAMES:
        try:
            c = tc.counterForName_(name)
            if c is not None and c.sampleCount() > 0:
                counter_names.append(name)
                counter_objects.append(c)
        except Exception:
            pass

    if not counter_names:
        log.info("No counters found with known names")
        return None

    # All counters should have the same sample count
    num_samples = counter_objects[0].sampleCount()

    # Extract timestamps from the first counter (shared across all)
    timestamps, _ = _extract_mio_counter(counter_objects[0], num_samples)

    # Extract values for each counter
    samples: list[list[float]] = [[] for _ in range(num_samples)]
    for ci, c_obj in enumerate(counter_objects):
        assert c_obj.sampleCount() == num_samples, (
            f"Counter {counter_names[ci]} has {c_obj.sampleCount()} samples, "
            f"expected {num_samples}"
        )
        _, values = _extract_mio_counter(c_obj, num_samples)
        for s in range(num_samples):
            samples[s].append(values[s])

    # Extract Active Cores / Total Cores from costTimeline (constant values)
    cost_tl = mio.costTimeline()
    if cost_tl is not None:
        sampled_cores = int(cost_tl.sampledCores())
        total_cores = int(cost_tl.totalCores())
        if sampled_cores > 0:
            counter_names.append("Active Cores")
            for s in range(num_samples):
                samples[s].append(float(sampled_cores))
            log.info("Active Cores: %d (of %d total)", sampled_cores, total_cores)
        if total_cores > 0:
            counter_names.append("Total Cores")
            for s in range(num_samples):
                samples[s].append(float(total_cores))

    log.info(
        "Extracted %d counters x %d samples from MIO timeline",
        len(counter_names), num_samples,
    )

    return {
        "counter_names": counter_names,
        "num_samples": num_samples,
        "timestamps_ns": timestamps,
        "samples": samples,
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
        addr = enc.get('addr', '')
        addr_str = f" {addr}" if addr else ""
        print(
            f"  Encoder#{enc['encoder_idx']}{addr_str}: {len(enc['dispatches'])} dispatches "
            f"(CB#{enc['command_buffer_idx']})"
        )

    print(f"\nCommand buffers: {len(result['command_buffers'])}")
    for i, cb in enumerate(result["command_buffers"][:10]):
        addr = cb.get('addr', '')
        addr_str = f" {addr}" if addr else ""
        print(
            f"  CB#{i}{addr_str}: {len(cb['dispatches'])} dispatches"
        )
        cb_kernels: dict[str, int] = defaultdict(int)
        for d in cb["dispatches"]:
            cb_kernels[d["kernel"]] += 1
        for k, c in sorted(cb_kernels.items(), key=lambda x: -x[1]):
            print(f"    {k}: {c}")


if __name__ == "__main__":
    main()
