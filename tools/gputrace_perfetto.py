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

Track mapping:
  - Process (pid) = command buffer index → one track group per CB
  - Thread (tid) = encoder index → one sub-track per encoder

Usage:
    uv run tools/gputrace_perfetto.py <path.gputrace> [-o output.json] [--open]
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


def timeline_to_perfetto(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert read_gputrace() output to Chrome Trace Event format.

    Returns a dict with key "traceEvents" containing the event list,
    suitable for JSON serialization and loading in Perfetto/chrome://tracing.
    """
    events: list[dict[str, Any]] = []

    # Build encoder → CB mapping from compute_encoders
    encoder_to_cb: dict[int, int] = {}
    encoder_dispatches: dict[int, list[dict[str, Any]]] = {}
    for enc in data.get("compute_encoders", []):
        enc_idx = enc["encoder_idx"]
        encoder_to_cb[enc_idx] = enc.get("command_buffer_idx", -1)
        encoder_dispatches[enc_idx] = enc.get("dispatches", [])

    # Collect CB addresses for metadata labels
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
    encoder_func_range: dict[int, tuple[int, int]] = {}  # enc_idx → (min, max)
    cb_func_range: dict[int, tuple[int, int]] = {}  # cb_idx → (min, max)

    # First pass: emit dispatch and barrier events, track ranges
    for event in data.get("events", []):
        etype = event.get("type")
        func_idx = event.get("index", 0)
        enc_idx = event.get("encoder_idx", 0)
        cb_idx = encoder_to_cb.get(enc_idx, 0)

        # Use cb_idx as pid, enc_idx as tid
        pid = cb_idx
        tid = enc_idx

        if etype == "dispatch":
            kernel = event.get("kernel", "unknown")
            args: dict[str, Any] = {"func_idx": func_idx}

            tg = event.get("threadgroups")
            if tg:
                args["threadgroups"] = f"{tg[0]}x{tg[1]}x{tg[2]}" if len(tg) == 3 else str(tg)

            tpt = event.get("threads_per_threadgroup")
            if tpt:
                args["threads_per_threadgroup"] = (
                    f"{tpt[0]}x{tpt[1]}x{tpt[2]}" if len(tpt) == 3 else str(tpt)
                )

            bufs = event.get("buffers_bound", {})
            if bufs:
                args["buffers_bound"] = len(bufs)

            dispatch_type = event.get("dispatch_type", "")
            if dispatch_type:
                args["dispatch_type"] = dispatch_type

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

            # Update ranges
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
                "s": "t",  # thread-scoped instant event
                "args": {"scope": scope},
            })

            _update_range(encoder_func_range, enc_idx, func_idx)
            _update_range(cb_func_range, cb_idx, func_idx)

        elif etype == "set_pipeline":
            # Pipeline sets as instant events on the encoder track
            kernel = event.get("kernel", "unknown")
            # We need encoder context — use the last known encoder
            # set_pipeline events don't have encoder_idx, so we skip them
            # unless we can determine the encoder from context
            pass

    # Second pass: emit encoder and CB wrapper spans (B/E pairs)
    for enc_idx, (fmin, fmax) in sorted(encoder_func_range.items()):
        cb_idx = encoder_to_cb.get(enc_idx, 0)
        addr = enc_addrs.get(enc_idx, "")
        label = f"Encoder #{enc_idx}"
        if addr:
            label += f" ({addr})"

        events.append({
            "ph": "B",
            "name": label,
            "cat": "encoder",
            "pid": cb_idx,
            "tid": enc_idx,
            "ts": fmin,
        })
        events.append({
            "ph": "E",
            "name": label,
            "cat": "encoder",
            "pid": cb_idx,
            "tid": enc_idx,
            "ts": fmax + 1,
        })

    # CB spans use a dedicated tid to avoid overlapping with encoder events.
    # Use tid = -1 (rendered as a separate row within the CB process).
    CB_OVERVIEW_TID = -1
    for cb_idx, (fmin, fmax) in sorted(cb_func_range.items()):
        addr = cb_addrs.get(cb_idx, "")
        label = f"CB #{cb_idx}"
        if addr:
            label += f" ({addr})"

        events.append({
            "ph": "B",
            "name": label,
            "cat": "command_buffer",
            "pid": cb_idx,
            "tid": CB_OVERVIEW_TID,
            "ts": fmin,
        })
        events.append({
            "ph": "E",
            "name": label,
            "cat": "command_buffer",
            "pid": cb_idx,
            "tid": CB_OVERVIEW_TID,
            "ts": fmax + 1,
        })

    # Emit metadata events for process and thread names
    seen_pids: set[int] = set()
    for cb_idx in cb_func_range:
        if cb_idx not in seen_pids:
            addr = cb_addrs.get(cb_idx, "")
            name = f"CB #{cb_idx}"
            if addr:
                name += f" ({addr})"
            events.append({
                "ph": "M",
                "name": "process_name",
                "pid": cb_idx,
                "tid": 0,
                "args": {"name": name},
            })
            seen_pids.add(cb_idx)

    # Thread name metadata for CB overview track
    for cb_idx in cb_func_range:
        events.append({
            "ph": "M",
            "name": "thread_name",
            "pid": cb_idx,
            "tid": CB_OVERVIEW_TID,
            "args": {"name": "CB Overview"},
        })

    seen_tids: set[tuple[int, int]] = set()
    for enc_idx in encoder_func_range:
        cb_idx = encoder_to_cb.get(enc_idx, 0)
        key = (cb_idx, enc_idx)
        if key not in seen_tids:
            addr = enc_addrs.get(enc_idx, "")
            name = f"Encoder #{enc_idx}"
            if addr:
                name += f" ({addr})"
            events.append({
                "ph": "M",
                "name": "thread_name",
                "pid": cb_idx,
                "tid": enc_idx,
                "args": {"name": name},
            })
            seen_tids.add(key)

    return {"traceEvents": events}


def _update_range(ranges: dict[int, tuple[int, int]], key: int, value: int) -> None:
    """Update min/max range for a key."""
    if key in ranges:
        cur_min, cur_max = ranges[key]
        ranges[key] = (min(cur_min, value), max(cur_max, value))
    else:
        ranges[key] = (value, value)


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
        "--open", action="store_true",
        help="Open ui.perfetto.dev in browser after export",
    )
    args = parser.parse_args()

    # Import gputrace_timeline (needs Apple frameworks + DYLD_FRAMEWORK_PATH)
    # Add tools dir to path so we can import sibling module
    tools_dir = str(Path(__file__).parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    from gputrace_timeline import _ensure_dyld_framework_path, read_gputrace

    _ensure_dyld_framework_path()

    trace_data = read_gputrace(args.gputrace)
    if trace_data is None:
        log.error("Failed to read %s", args.gputrace)
        sys.exit(1)

    perfetto = timeline_to_perfetto(trace_data)

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
