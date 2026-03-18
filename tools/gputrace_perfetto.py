# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyobjc-core",
#     "pyobjc-framework-Cocoa",
# ]
# ///
"""Export GPU trace timeline as Chrome Trace Event JSON for Perfetto.

Reads a .gputrace file via gputrace_timeline.read_gputrace() and outputs
a Chrome Trace Event Format JSON file that can be loaded in ui.perfetto.dev.

Since there are no wall-clock timestamps in the trace (only func_idx ordering),
timestamps are synthesized: each dispatch/barrier occupies a 1µs slot at
ts = func_idx. This preserves ordering and produces a readable timeline.

Two grouping modes:

  --group-by pipeline  (default)
    Process = compute pipeline (kernel name). Each unique kernel gets its own
    track group; dispatches appear chronologically within. Barriers appear on
    a dedicated "Barriers" track.

  --group-by cb
    Process = command buffer index, Thread = encoder index.
    Shows the hardware submission structure.

Usage:
    uv run tools/gputrace_perfetto.py <path.gputrace> [-o output.json] [--open]
    uv run tools/gputrace_perfetto.py <path.gputrace> --group-by cb
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def timeline_to_perfetto(
    data: dict[str, Any],
    group_by: str = "pipeline",
) -> dict[str, list[dict[str, Any]]]:
    """Convert read_gputrace() output to Chrome Trace Event format.

    Args:
        data: Output from read_gputrace().
        group_by: "pipeline" groups tracks by kernel name (default),
                  "cb" groups by command buffer / encoder.

    Returns a dict with key "traceEvents" containing the event list,
    suitable for JSON serialization and loading in Perfetto/chrome://tracing.
    """
    if group_by == "pipeline":
        return _group_by_pipeline(data)
    elif group_by == "cb":
        return _group_by_cb(data)
    else:
        raise ValueError(f"Unknown group_by mode: {group_by!r}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_tuple(t: tuple[int, ...] | list[int]) -> str:
    if len(t) == 3:
        return f"{t[0]}x{t[1]}x{t[2]}"
    return str(t)


def _dispatch_args(event: dict[str, Any]) -> dict[str, Any]:
    """Build the args dict for a dispatch event."""
    func_idx = event.get("index", 0)
    args: dict[str, Any] = {"func_idx": func_idx}

    tg = event.get("threadgroups")
    if tg:
        args["threadgroups"] = _format_tuple(tg)

    tpt = event.get("threads_per_threadgroup")
    if tpt:
        args["threads_per_threadgroup"] = _format_tuple(tpt)

    bufs = event.get("buffers_bound", {})
    if bufs:
        args["buffers_bound"] = len(bufs)

    dispatch_type = event.get("dispatch_type", "")
    if dispatch_type:
        args["dispatch_type"] = dispatch_type

    return args


def _update_range(
    ranges: dict[int, tuple[int, int]], key: int, value: int,
) -> None:
    """Update min/max range for a key."""
    if key in ranges:
        cur_min, cur_max = ranges[key]
        ranges[key] = (min(cur_min, value), max(cur_max, value))
    else:
        ranges[key] = (value, value)


def _update_range_str(
    ranges: dict[str, tuple[int, int]], key: str, value: int,
) -> None:
    if key in ranges:
        cur_min, cur_max = ranges[key]
        ranges[key] = (min(cur_min, value), max(cur_max, value))
    else:
        ranges[key] = (value, value)


# ---------------------------------------------------------------------------
# Pipeline-grouped mode
# ---------------------------------------------------------------------------


def _group_by_pipeline(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group tracks by compute pipeline (kernel name).

    Each unique kernel gets a process (pid). All dispatches of that kernel
    appear chronologically on a single thread. Barriers go to a dedicated
    "Barriers" process.
    """
    events: list[dict[str, Any]] = []

    # Build encoder → CB mapping
    encoder_to_cb: dict[int, int] = {}
    for enc in data.get("compute_encoders", []):
        encoder_to_cb[enc["encoder_idx"]] = enc.get("command_buffer_idx", -1)

    # Collect CB addresses
    cb_addrs: dict[int, str] = {}
    for cb_idx, cb in enumerate(data.get("command_buffers", [])):
        addr = cb.get("addr", "")
        if addr:
            cb_addrs[cb_idx] = addr

    # Assign a stable pid to each kernel (by first appearance order)
    kernel_to_pid: dict[str, int] = {}
    next_pid = 0

    def _get_kernel_pid(kernel: str) -> int:
        nonlocal next_pid
        if kernel not in kernel_to_pid:
            kernel_to_pid[kernel] = next_pid
            next_pid += 1
        return kernel_to_pid[kernel]

    # Reserved pid for barriers
    BARRIER_PID = -1

    # Track func_idx ranges per kernel for wrapper spans
    kernel_func_range: dict[str, tuple[int, int]] = {}

    for event in data.get("events", []):
        etype = event.get("type")
        func_idx = event.get("index", 0)
        enc_idx = event.get("encoder_idx", 0)
        cb_idx = encoder_to_cb.get(enc_idx, 0)

        if etype == "dispatch":
            kernel = event.get("kernel", "unknown")
            pid = _get_kernel_pid(kernel)
            args = _dispatch_args(event)
            args["cb"] = cb_idx
            cb_addr = cb_addrs.get(cb_idx, "")
            if cb_addr:
                args["cb_addr"] = cb_addr
            args["encoder"] = enc_idx

            events.append({
                "ph": "X",
                "name": kernel,
                "cat": "dispatch",
                "pid": pid,
                "tid": 0,
                "ts": func_idx,
                "dur": 1,
                "args": args,
            })

            _update_range_str(kernel_func_range, kernel, func_idx)

        elif etype == "barrier":
            scope = event.get("scope", "buffers")
            events.append({
                "ph": "i",
                "name": f"barrier ({scope})",
                "cat": "barrier",
                "pid": BARRIER_PID,
                "tid": enc_idx,
                "ts": func_idx,
                "s": "t",
                "args": {
                    "scope": scope,
                    "encoder": enc_idx,
                    "cb": cb_idx,
                },
            })

    # Process name metadata for each kernel
    for kernel, pid in kernel_to_pid.items():
        events.append({
            "ph": "M",
            "name": "process_name",
            "pid": pid,
            "tid": 0,
            "args": {"name": kernel},
        })

    # Process sort order: by first func_idx (earliest first)
    for kernel, pid in kernel_to_pid.items():
        fmin = kernel_func_range.get(kernel, (0, 0))[0]
        events.append({
            "ph": "M",
            "name": "process_sort_index",
            "pid": pid,
            "tid": 0,
            "args": {"sort_index": fmin},
        })

    # Barrier process metadata (if any barriers exist)
    has_barriers = any(
        e.get("type") == "barrier" for e in data.get("events", [])
    )
    if has_barriers:
        events.append({
            "ph": "M",
            "name": "process_name",
            "pid": BARRIER_PID,
            "tid": 0,
            "args": {"name": "Barriers"},
        })
        # Sort barriers to the end
        events.append({
            "ph": "M",
            "name": "process_sort_index",
            "pid": BARRIER_PID,
            "tid": 0,
            "args": {"sort_index": 999999},
        })

    return {"traceEvents": events}


# ---------------------------------------------------------------------------
# CB-grouped mode (original)
# ---------------------------------------------------------------------------


def _group_by_cb(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group tracks by command buffer / encoder (original layout).

    Process (pid) = CB index, Thread (tid) = encoder index.
    """
    events: list[dict[str, Any]] = []

    # Build encoder → CB mapping from compute_encoders
    encoder_to_cb: dict[int, int] = {}
    for enc in data.get("compute_encoders", []):
        encoder_to_cb[enc["encoder_idx"]] = enc.get("command_buffer_idx", -1)

    # Collect CB addresses
    cb_addrs: dict[int, str] = {}
    for cb_idx, cb in enumerate(data.get("command_buffers", [])):
        addr = cb.get("addr", "")
        if addr:
            cb_addrs[cb_idx] = addr

    # Collect encoder addresses
    enc_addrs: dict[int, str] = {}
    for enc in data.get("compute_encoders", []):
        addr = enc.get("addr", "")
        if addr:
            enc_addrs[enc["encoder_idx"]] = addr

    # Track func_idx ranges per encoder and per CB for wrapper spans
    encoder_func_range: dict[int, tuple[int, int]] = {}
    cb_func_range: dict[int, tuple[int, int]] = {}

    for event in data.get("events", []):
        etype = event.get("type")
        func_idx = event.get("index", 0)
        enc_idx = event.get("encoder_idx", 0)
        cb_idx = encoder_to_cb.get(enc_idx, 0)

        pid = cb_idx
        tid = enc_idx

        if etype == "dispatch":
            kernel = event.get("kernel", "unknown")
            args = _dispatch_args(event)

            events.append({
                "ph": "X",
                "name": kernel,
                "cat": "dispatch",
                "pid": pid,
                "tid": tid,
                "ts": func_idx,
                "dur": 1,
                "args": args,
            })

            _update_range(encoder_func_range, enc_idx, func_idx)
            _update_range(cb_func_range, cb_idx, func_idx)

        elif etype == "barrier":
            scope = event.get("scope", "buffers")
            events.append({
                "ph": "i",
                "name": f"barrier ({scope})",
                "cat": "barrier",
                "pid": pid,
                "tid": tid,
                "ts": func_idx,
                "s": "t",
                "args": {"scope": scope},
            })

            _update_range(encoder_func_range, enc_idx, func_idx)
            _update_range(cb_func_range, cb_idx, func_idx)

    # Encoder wrapper spans (B/E pairs)
    for enc_idx, (fmin, fmax) in sorted(encoder_func_range.items()):
        cb_idx = encoder_to_cb.get(enc_idx, 0)
        addr = enc_addrs.get(enc_idx, "")
        label = f"Encoder #{enc_idx}"
        if addr:
            label += f" ({addr})"

        events.append({
            "ph": "B", "name": label, "cat": "encoder",
            "pid": cb_idx, "tid": enc_idx, "ts": fmin,
        })
        events.append({
            "ph": "E", "name": label, "cat": "encoder",
            "pid": cb_idx, "tid": enc_idx, "ts": fmax + 1,
        })

    # CB overview spans on a dedicated tid
    CB_OVERVIEW_TID = -1
    for cb_idx, (fmin, fmax) in sorted(cb_func_range.items()):
        addr = cb_addrs.get(cb_idx, "")
        label = f"CB #{cb_idx}"
        if addr:
            label += f" ({addr})"

        events.append({
            "ph": "B", "name": label, "cat": "command_buffer",
            "pid": cb_idx, "tid": CB_OVERVIEW_TID, "ts": fmin,
        })
        events.append({
            "ph": "E", "name": label, "cat": "command_buffer",
            "pid": cb_idx, "tid": CB_OVERVIEW_TID, "ts": fmax + 1,
        })

    # Process name metadata
    for cb_idx in cb_func_range:
        addr = cb_addrs.get(cb_idx, "")
        name = f"CB #{cb_idx}"
        if addr:
            name += f" ({addr})"
        events.append({
            "ph": "M", "name": "process_name",
            "pid": cb_idx, "tid": 0, "args": {"name": name},
        })

    # Thread name metadata
    for cb_idx in cb_func_range:
        events.append({
            "ph": "M", "name": "thread_name",
            "pid": cb_idx, "tid": CB_OVERVIEW_TID,
            "args": {"name": "CB Overview"},
        })

    for enc_idx in encoder_func_range:
        cb_idx = encoder_to_cb.get(enc_idx, 0)
        addr = enc_addrs.get(enc_idx, "")
        name = f"Encoder #{enc_idx}"
        if addr:
            name += f" ({addr})"
        events.append({
            "ph": "M", "name": "thread_name",
            "pid": cb_idx, "tid": enc_idx,
            "args": {"name": name},
        })

    return {"traceEvents": events}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GPU trace timeline as Chrome Trace Event JSON for Perfetto."
    )
    parser.add_argument("gputrace", help="Path to .gputrace file")
    parser.add_argument(
        "-o", "--output",
        help="Output JSON path (default: <input_stem>_perfetto.json)",
    )
    parser.add_argument(
        "--group-by", choices=["pipeline", "cb"], default="pipeline",
        help="Track grouping: 'pipeline' (default) groups by kernel name, "
             "'cb' groups by command buffer/encoder.",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open ui.perfetto.dev in browser after export",
    )
    args = parser.parse_args()

    # Import gputrace_timeline (needs Apple frameworks + DYLD_FRAMEWORK_PATH)
    tools_dir = str(Path(__file__).parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    from gputrace_timeline import _ensure_dyld_framework_path, read_gputrace

    _ensure_dyld_framework_path()

    trace_data = read_gputrace(args.gputrace)
    if trace_data is None:
        log.error("Failed to read %s", args.gputrace)
        sys.exit(1)

    perfetto = timeline_to_perfetto(trace_data, group_by=args.group_by)

    output_path = args.output
    if output_path is None:
        stem = Path(args.gputrace).stem
        output_path = f"{stem}_perfetto.json"

    with open(output_path, "w") as f:
        json.dump(perfetto, f, indent=2)

    num_events = len(perfetto["traceEvents"])
    log.info("Wrote %d events to %s", num_events, output_path)
    log.info("Open https://ui.perfetto.dev and drag in the file to view.")

    if args.open:
        webbrowser.open("https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
