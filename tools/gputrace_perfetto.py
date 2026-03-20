# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyobjc-core",
#     "pyobjc-framework-Cocoa",
#     "perfetto",
# ]
# ///
"""Export GPU trace timeline as Perfetto-compatible trace files.

Reads a .gputrace file via gputrace_timeline.read_gputrace() and outputs
either Chrome Trace Event JSON or Perfetto protobuf (.pftrace) format,
both loadable in ui.perfetto.dev.

Since there are no wall-clock timestamps in the trace (only func_idx ordering),
timestamps are synthesized: each dispatch/barrier occupies a 1µs slot at
ts = func_idx. This preserves ordering and produces a readable timeline.

Two grouping modes:

  --group-by pipeline  (default)
    JSON: Process = kernel name, dispatches chronological, barriers on dedicated track.
    pftrace: GpuRenderStageEvent for dispatches (Perfetto GPU UI), TrackEvent for
    barriers/encoder spans.

  --group-by cb
    JSON: Process = CB index, Thread = encoder index.
    pftrace: All TrackEvent — process per CB, child tracks per encoder.

Two output formats:

  --format json  (default)
    Chrome Trace Event Format JSON.

  --format pftrace
    Perfetto protobuf binary. GpuRenderStageEvent gets dedicated GPU UI treatment
    in Perfetto (pipeline mode only).

Usage:
    uv run tools/gputrace_perfetto.py <path.gputrace> [-o output.json] [--open]
    uv run tools/gputrace_perfetto.py <path.gputrace> --group-by cb
    uv run tools/gputrace_perfetto.py <path.gputrace> --format pftrace
    uv run tools/gputrace_perfetto.py <path.gputrace> --format pftrace --group-by cb
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


def _build_dispatch_duration_map(events: list[dict[str, Any]]) -> dict[int, int]:
    """Map each dispatch's func_idx to its duration (units = func_idx delta).

    Each dispatch lasts until the next dispatch in the global stream
    (barriers are skipped — they're sync markers within a dispatch's span).
    The last dispatch gets a duration of 1.
    """
    dispatch_indices = sorted(
        ev.get("index", 0) for ev in events if ev.get("type") == "dispatch"
    )
    dur: dict[int, int] = {}
    for i, idx in enumerate(dispatch_indices):
        dur[idx] = dispatch_indices[i + 1] - idx if i + 1 < len(dispatch_indices) else 1
    return dur


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
                "tid": 0,
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
# Perfetto protobuf (.pftrace) export
# ---------------------------------------------------------------------------


def timeline_to_pftrace(
    data: dict[str, Any],
    group_by: str = "pipeline",
    counters: dict[str, Any] | None = None,
) -> bytes:
    """Convert read_gputrace() output to Perfetto protobuf (.pftrace) format.

    Args:
        data: Output from read_gputrace().
        group_by: "pipeline" uses GpuRenderStageEvent for dispatches (GPU UI),
                  "cb" uses all TrackEvent (process per CB).
        counters: Optional output from read_gputrace_counters(). When provided,
                  GpuCounterEvent packets are appended for GPU utilization tracks.

    Returns serialized protobuf bytes suitable for writing to a .pftrace file.
    """
    if group_by == "pipeline":
        trace = _pftrace_pipeline(data)
    elif group_by == "cb":
        trace = _pftrace_cb(data)
    else:
        raise ValueError(f"Unknown group_by mode: {group_by!r}")

    if counters is not None:
        from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as pb
        _add_gpu_counters(trace, pb, counters)

    return trace.SerializeToString()


# Pipeline mode sequence ID
_GPU_SEQ = 1


def _pftrace_pipeline(data: dict[str, Any]) -> Any:
    """Pipeline-grouped pftrace: GpuRenderStageEvent dispatches + TrackEvent barriers/spans.

    Returns a pb.Trace() object (not yet serialized).
    """
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as pb

    trace = pb.Trace()

    # -- Mappings -----------------------------------------------------------
    encoder_to_cb: dict[int, int] = {}
    for enc in data.get("compute_encoders", []):
        encoder_to_cb[enc["encoder_idx"]] = enc.get("command_buffer_idx", 0)

    cb_addrs: dict[int, str] = {}
    for cb_idx, cb in enumerate(data.get("command_buffers", [])):
        if cb.get("addr"):
            cb_addrs[cb_idx] = cb["addr"]

    # Collect unique kernels in first-appearance order for hw_queue interning
    kernel_order: list[str] = []
    kernel_to_iid: dict[str, int] = {}
    for ev in data.get("events", []):
        if ev.get("type") == "dispatch":
            k = ev.get("kernel", "unknown")
            if k not in kernel_to_iid:
                kernel_to_iid[k] = len(kernel_order)
                kernel_order.append(k)

    # -- 1. Interned GPU specifications (via InternedData) ------------------
    STAGE_IID = 100

    intern_pkt = trace.packet.add()
    intern_pkt.trusted_packet_sequence_id = _GPU_SEQ
    intern_pkt.timestamp = 0
    intern_pkt.sequence_flags = 1  # SEQ_INCREMENTAL_STATE_CLEARED

    for kernel_name in kernel_order:
        iid = kernel_to_iid[kernel_name] + 1  # 1-based
        spec = intern_pkt.interned_data.gpu_specifications.add()
        spec.iid = iid
        spec.name = kernel_name
        spec.description = f"Compute kernel: {kernel_name}"
        spec.category = pb.InternedGpuRenderStageSpecification.COMPUTE

    # Stage specification
    stage_spec = intern_pkt.interned_data.gpu_specifications.add()
    stage_spec.iid = STAGE_IID
    stage_spec.name = "Compute Dispatch"
    stage_spec.description = "Metal compute pipeline dispatch"
    stage_spec.category = pb.InternedGpuRenderStageSpecification.COMPUTE

    # -- 2. GpuRenderStageEvent packets (dispatches) -----------------------
    # Duration: each event lasts until the next event (Xcode-style).
    dur_map = _build_dispatch_duration_map(data.get("events", []))

    event_id = 0
    for ev in data.get("events", []):
        if ev.get("type") != "dispatch":
            continue

        kernel = ev.get("kernel", "unknown")
        func_idx = ev.get("index", 0)
        enc_idx = ev.get("encoder_idx", 0)
        cb_idx = encoder_to_cb.get(enc_idx, 0)

        pkt = trace.packet.add()
        pkt.timestamp = func_idx * 1000  # ns
        pkt.trusted_packet_sequence_id = _GPU_SEQ

        gpu = pkt.gpu_render_stage_event
        gpu.event_id = event_id
        event_id += 1
        gpu.hw_queue_iid = kernel_to_iid[kernel] + 1  # 1-based
        gpu.stage_iid = STAGE_IID
        gpu.duration = dur_map.get(func_idx, 1) * 1000  # ns
        gpu.submission_id = cb_idx

        # command_buffer_handle is uint64 — parse hex addr
        cb_addr = cb_addrs.get(cb_idx, "0x0")
        gpu.command_buffer_handle = int(cb_addr, 16)

        # Extra data
        tg = ev.get("threadgroups")
        if tg:
            ed = gpu.extra_data.add()
            ed.name = "threadgroups"
            ed.value = _format_tuple(tg)

        tpt = ev.get("threads_per_threadgroup")
        if tpt:
            ed = gpu.extra_data.add()
            ed.name = "threads_per_threadgroup"
            ed.value = _format_tuple(tpt)

        bufs = ev.get("buffers_bound")
        if bufs:
            ed = gpu.extra_data.add()
            ed.name = "buffers_bound"
            ed.value = str(len(bufs))

        dispatch_type = ev.get("dispatch_type", "")
        if dispatch_type:
            ed = gpu.extra_data.add()
            ed.name = "dispatch_type"
            ed.value = dispatch_type

        ed = gpu.extra_data.add()
        ed.name = "encoder"
        ed.value = str(enc_idx)

        ed = gpu.extra_data.add()
        ed.name = "cb"
        ed.value = str(cb_idx)

    # Pipeline mode: GpuRenderStageEvent carries all context (encoder/CB in
    # extra_data). No TrackEvent tracks needed — they just add noise.

    return trace


# CB-grouped pftrace UUIDs
_CB_PROCESS_BASE = 1000    # CB N -> 1000+N
_CB_OVERVIEW_BASE = 2000   # CB overview N -> 2000+N
_CB_ENCODER_BASE = 3000    # encoder N -> 3000+N
_CB_PFTRACE_SEQ = 1


def _pftrace_cb(data: dict[str, Any]) -> Any:
    """CB-grouped pftrace: all TrackEvent — process per CB, child tracks per encoder.

    Returns a pb.Trace() object (not yet serialized).
    """
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as pb

    trace = pb.Trace()

    # -- Mappings -----------------------------------------------------------
    encoder_to_cb: dict[int, int] = {}
    for enc in data.get("compute_encoders", []):
        encoder_to_cb[enc["encoder_idx"]] = enc.get("command_buffer_idx", 0)

    cb_addrs: dict[int, str] = {}
    for cb_idx, cb in enumerate(data.get("command_buffers", [])):
        if cb.get("addr"):
            cb_addrs[cb_idx] = cb["addr"]

    enc_addrs: dict[int, str] = {}
    for enc in data.get("compute_encoders", []):
        if enc.get("addr"):
            enc_addrs[enc["encoder_idx"]] = enc["addr"]

    # Track func_idx ranges per encoder and CB
    encoder_func_range: dict[int, tuple[int, int]] = {}
    cb_func_range: dict[int, tuple[int, int]] = {}

    for ev in data.get("events", []):
        func_idx = ev.get("index", 0)
        enc_idx = ev.get("encoder_idx", 0)
        cb_idx = encoder_to_cb.get(enc_idx, 0)

        if enc_idx in encoder_func_range:
            lo, hi = encoder_func_range[enc_idx]
            encoder_func_range[enc_idx] = (min(lo, func_idx), max(hi, func_idx))
        else:
            encoder_func_range[enc_idx] = (func_idx, func_idx)

        if cb_idx in cb_func_range:
            lo, hi = cb_func_range[cb_idx]
            cb_func_range[cb_idx] = (min(lo, func_idx), max(hi, func_idx))
        else:
            cb_func_range[cb_idx] = (func_idx, func_idx)

    # -- 1. TrackDescriptor: process per CB, child tracks per encoder -------
    for cb_idx in sorted(cb_func_range):
        addr = cb_addrs.get(cb_idx, "")
        name = f"CB #{cb_idx}"
        if addr:
            name += f" ({addr})"

        proc_pkt = trace.packet.add()
        proc_pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
        proc_pkt.track_descriptor.uuid = _CB_PROCESS_BASE + cb_idx
        proc_pkt.track_descriptor.name = name
        proc_pkt.track_descriptor.process.pid = cb_idx
        proc_pkt.track_descriptor.process.process_name = name

        # CB overview child track
        overview_pkt = trace.packet.add()
        overview_pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
        overview_pkt.track_descriptor.uuid = _CB_OVERVIEW_BASE + cb_idx
        overview_pkt.track_descriptor.parent_uuid = _CB_PROCESS_BASE + cb_idx
        overview_pkt.track_descriptor.name = "CB Overview"

    for enc in data.get("compute_encoders", []):
        enc_idx = enc["encoder_idx"]
        if enc_idx not in encoder_func_range:
            continue
        cb_idx = enc.get("command_buffer_idx", 0)
        addr = enc.get("addr", "")
        label = f"Encoder #{enc_idx}"
        if addr:
            label += f" ({addr})"

        enc_pkt = trace.packet.add()
        enc_pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
        enc_pkt.track_descriptor.uuid = _CB_ENCODER_BASE + enc_idx
        enc_pkt.track_descriptor.parent_uuid = _CB_PROCESS_BASE + cb_idx
        enc_pkt.track_descriptor.name = label

    # -- 2. Dispatch slices + barrier instants on encoder tracks ------------
    # Duration: each event lasts until the next event (Xcode-style).
    dur_map = _build_dispatch_duration_map(data.get("events", []))

    first_event = True
    for ev in data.get("events", []):
        etype = ev.get("type")
        func_idx = ev.get("index", 0)
        enc_idx = ev.get("encoder_idx", 0)
        enc_uuid = _CB_ENCODER_BASE + enc_idx

        if etype == "dispatch":
            kernel = ev.get("kernel", "unknown")
            dur = dur_map.get(func_idx, 1)

            # SLICE_BEGIN
            begin_pkt = trace.packet.add()
            begin_pkt.timestamp = func_idx * 1000
            begin_pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
            if first_event:
                begin_pkt.sequence_flags = 1  # SEQ_INCREMENTAL_STATE_CLEARED
                first_event = False

            begin_te = begin_pkt.track_event
            begin_te.type = pb.TrackEvent.TYPE_SLICE_BEGIN
            begin_te.track_uuid = enc_uuid
            begin_te.name = kernel

            # Debug annotations for metadata
            da = begin_te.debug_annotations.add()
            da.name = "func_idx"
            da.int_value = func_idx

            tg = ev.get("threadgroups")
            if tg:
                da = begin_te.debug_annotations.add()
                da.name = "threadgroups"
                da.string_value = _format_tuple(tg)

            tpt = ev.get("threads_per_threadgroup")
            if tpt:
                da = begin_te.debug_annotations.add()
                da.name = "threads_per_threadgroup"
                da.string_value = _format_tuple(tpt)

            bufs = ev.get("buffers_bound")
            if bufs:
                da = begin_te.debug_annotations.add()
                da.name = "buffers_bound"
                da.int_value = len(bufs)

            dispatch_type = ev.get("dispatch_type", "")
            if dispatch_type:
                da = begin_te.debug_annotations.add()
                da.name = "dispatch_type"
                da.string_value = dispatch_type

            # SLICE_END
            end_pkt = trace.packet.add()
            end_pkt.timestamp = (func_idx + dur) * 1000
            end_pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
            end_te = end_pkt.track_event
            end_te.type = pb.TrackEvent.TYPE_SLICE_END
            end_te.track_uuid = enc_uuid

        elif etype == "barrier":
            scope = ev.get("scope", "buffers")

            pkt = trace.packet.add()
            pkt.timestamp = func_idx * 1000
            pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
            if first_event:
                pkt.sequence_flags = 1
                first_event = False

            te = pkt.track_event
            te.type = pb.TrackEvent.TYPE_INSTANT
            te.track_uuid = enc_uuid
            te.name = f"barrier ({scope})"

            da = te.debug_annotations.add()
            da.name = "scope"
            da.string_value = scope

    # -- 3. Encoder wrapper spans -------------------------------------------
    for enc in data.get("compute_encoders", []):
        enc_idx = enc["encoder_idx"]
        if enc_idx not in encoder_func_range:
            continue

        lo, hi = encoder_func_range[enc_idx]
        enc_uuid = _CB_ENCODER_BASE + enc_idx
        addr = enc.get("addr", "")
        label = f"Encoder #{enc_idx}"
        if addr:
            label += f" ({addr})"

        begin_pkt = trace.packet.add()
        begin_pkt.timestamp = lo * 1000
        begin_pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
        if first_event:
            begin_pkt.sequence_flags = 1
            first_event = False
        begin_te = begin_pkt.track_event
        begin_te.type = pb.TrackEvent.TYPE_SLICE_BEGIN
        begin_te.track_uuid = enc_uuid
        begin_te.name = label

        end_pkt = trace.packet.add()
        end_pkt.timestamp = (hi + 1) * 1000
        end_pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
        end_te = end_pkt.track_event
        end_te.type = pb.TrackEvent.TYPE_SLICE_END
        end_te.track_uuid = enc_uuid

    # -- 4. CB overview spans -----------------------------------------------
    for cb_idx in sorted(cb_func_range):
        lo, hi = cb_func_range[cb_idx]
        overview_uuid = _CB_OVERVIEW_BASE + cb_idx
        addr = cb_addrs.get(cb_idx, "")
        label = f"CB #{cb_idx}"
        if addr:
            label += f" ({addr})"

        begin_pkt = trace.packet.add()
        begin_pkt.timestamp = lo * 1000
        begin_pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
        if first_event:
            begin_pkt.sequence_flags = 1
            first_event = False
        begin_te = begin_pkt.track_event
        begin_te.type = pb.TrackEvent.TYPE_SLICE_BEGIN
        begin_te.track_uuid = overview_uuid
        begin_te.name = label

        end_pkt = trace.packet.add()
        end_pkt.timestamp = (hi + 1) * 1000
        end_pkt.trusted_packet_sequence_id = _CB_PFTRACE_SEQ
        end_te = end_pkt.track_event
        end_te.type = pb.TrackEvent.TYPE_SLICE_END
        end_te.track_uuid = overview_uuid

    return trace


# ---------------------------------------------------------------------------
# GPU performance counter tracks
# ---------------------------------------------------------------------------

_COUNTER_SEQ = 3

# Counter name → display group for Perfetto track grouping.
# Counters in the same group appear under a collapsible parent track.
# Names must match the keys in _COUNTER_CATEGORY_ORDER in gputrace_timeline.py.
_COUNTER_GROUP_MAP: dict[str, str] = {
    # GPU activity & cores
    "GT Active Core Count": "GPU Activity",
    "Raytracing Active GT": "GPU Activity",
    # Occupancy
    "Total Occupancy": "Occupancy",
    "Compute Occupancy": "Occupancy",
    "Fragment Occupancy": "Occupancy",
    "Vertex Occupancy": "Occupancy",
    "Occupancy Manager Target": "Occupancy",
    "Total Simdgroups Inflight Per Shader Core": "Occupancy",
    "Compute Simdgroups Inflight Per Shader Core": "Occupancy",
    "Fragment Simdgroups Inflight Per Shader Core": "Occupancy",
    "Vertex Simdgroups Inflight Per Shader Core": "Occupancy",
    "Occupancy Management L1 Eviction Rate": "Occupancy",
    # Memory bandwidth
    "AF Bandwidth": "Memory Bandwidth",
    "AF Read Bandwidth": "Memory Bandwidth",
    "AF Write Bandwidth": "Memory Bandwidth",
    "AF Peak Bandwidth": "Memory Bandwidth",
    "AF Peak Read Bandwidth": "Memory Bandwidth",
    "AF Peak Write Bandwidth": "Memory Bandwidth",
    "L2 Bandwidth": "Memory Bandwidth",
    # Shader core utilization & instruction throughput
    "Shader Core Utilization": "Shader Utilization",
    "Shader Core Limiter": "Shader Utilization",
    "ALU Utilization": "Shader Utilization",
    "F16 Utilization": "Shader Utilization",
    "F16 Limiter": "Shader Utilization",
    "F32 Utilization": "Shader Utilization",
    "F32 Limiter": "Shader Utilization",
    "IC Utilization": "Shader Utilization",
    "IC Limiter": "Shader Utilization",
    "SCIB Utilization": "Shader Utilization",
    "SCIB Limiter": "Shader Utilization",
    "Control Flow Utilization": "Shader Utilization",
    "Control Flow Limiter": "Shader Utilization",
    "Instruction Dispatch Utilization": "Shader Utilization",
    "Instruction Dispatch Limiter": "Shader Utilization",
    "Instruction Issue Utilization": "Shader Utilization",
    "Instruction Issue Limiter": "Shader Utilization",
    "Address Generation Utilization": "Shader Utilization",
    "Address Generation Limiter": "Shader Utilization",
    # Shader launch
    "Compute Shader Launch Utilization": "Shader Launch",
    "Compute Shader Launch Limiter": "Shader Launch",
    "Fragment Shader Launch Utilization": "Shader Launch",
    "Fragment Shader Launch Limiter": "Shader Launch",
    "Vertex Shader Launch Utilization": "Shader Launch",
    "Vertex Shader Launch Limiter": "Shader Launch",
    # L1 cache bandwidth
    "L1 Load Bandwidth": "L1 Cache",
    "L1 Store Bandwidth": "L1 Cache",
    "L1 Cache Utilization": "L1 Cache",
    "L1 Cache Limiter": "L1 Cache",
    "Buffer L1 Load Bandwidth": "L1 Cache",
    "Buffer L1 Store Bandwidth": "L1 Cache",
    "Buffer L1 Load Ratio": "L1 Cache",
    "Buffer L1 Store Ratio": "L1 Cache",
    "Buffer L1 Miss Rate": "L1 Cache",
    "Imageblock L1 Load Bandwidth": "L1 Cache",
    "Imageblock L1 Store Bandwidth": "L1 Cache",
    "Imageblock L1 Load Ratio": "L1 Cache",
    "Imageblock L1 Store Ratio": "L1 Cache",
    "Threadgroup Memory L1 Load Bandwidth": "L1 Cache",
    "Threadgroup Memory L1 Store Bandwidth": "L1 Cache",
    "Threadgroup L1 Load Ratio": "L1 Cache",
    "Threadgroup L1 Store Ratio": "L1 Cache",
    "Stack L1 Load Bandwidth": "L1 Cache",
    "Stack L1 Store Bandwidth": "L1 Cache",
    "Stack L1 Load Ratio": "L1 Cache",
    "Stack L1 Store Ratio": "L1 Cache",
    "GPR L1 Load Bandwidth": "L1 Cache",
    "GPR L1 Store Bandwidth": "L1 Cache",
    "GPR L1 Read Ratio": "L1 Cache",
    "GPR L1 Write Ratio": "L1 Cache",
    "Other L1 Load Bandwidth": "L1 Cache",
    "Other L1 Store Bandwidth": "L1 Cache",
    "Other L1 Loads Ratio": "L1 Cache",
    "Other L1 Stores Ratio": "L1 Cache",
    # L1 residency
    "L1 Total Occupancy": "L1 Residency",
    "L1 Total Bytes Occupancy": "L1 Residency",
    "L1 Buffer Occupancy": "L1 Residency",
    "L1 Buffer Bytes Occupancy": "L1 Residency",
    "L1 Imageblock Occupancy": "L1 Residency",
    "L1 Imageblock Bytes Occupancy": "L1 Residency",
    "L1 Threadgroup Occupancy": "L1 Residency",
    "L1 Threadgroup Bytes Occupancy": "L1 Residency",
    "L1 GPR Occupancy": "L1 Residency",
    "L1 GPR Bytes Occupancy": "L1 Residency",
    "L1 Stack Occupancy": "L1 Residency",
    "L1 Stack Bytes Occupancy": "L1 Residency",
    "L1 Other Occupancy": "L1 Residency",
    "L1 Other Bytes Occupancy": "L1 Residency",
    "L1 Raytracing Scratch Occupancy": "L1 Residency",
    "L1 Raytracing Scratch Bytes Occupancy": "L1 Residency",
    # L2 / texture / MMU
    "L2 Cache Utilization": "L2 / Texture / MMU",
    "L2 Cache Limiter": "L2 / Texture / MMU",
    "Texture Cache Utilization": "L2 / Texture / MMU",
    "Texture Cache Limiter": "L2 / Texture / MMU",
    "Texture Read Utilization": "L2 / Texture / MMU",
    "Texture Read Limiter": "L2 / Texture / MMU",
    "Texture Write Utilization": "L2 / Texture / MMU",
    "Texture Write Limiter": "L2 / Texture / MMU",
    "TextureFilteringLimiter": "L2 / Texture / MMU",
    "CompressionRatioTextureMemoryRead": "L2 / Texture / MMU",
    "MMU Utilization": "L2 / Texture / MMU",
    "MMU Limiter": "L2 / Texture / MMU",
    # Raytracing
    "Raytracing Active": "Raytracing",
    "Ray Occupancy": "Raytracing",
    "Leaf Test Occupancy": "Raytracing",
    "Ray T Leaf Test": "Raytracing",
    "Raytracing Node Test": "Raytracing",
    "Intersect Ray Threads": "Raytracing",
    "Raytracing Scratch L1 Load Bandwidth": "Raytracing",
    "Raytracing Scratch L1 Store Bandwidth": "Raytracing",
    "Raytracing Scratch L1 Load Ratio": "Raytracing",
    "Raytracing Scratch L1 Store Ratio": "Raytracing",
}

# Group display order in the Perfetto UI (lower = higher in the list)
_GROUP_ORDER: list[str] = [
    "GPU Activity",
    "Occupancy",
    "Memory Bandwidth",
    "Shader Utilization",
    "Shader Launch",
    "L1 Cache",
    "L1 Residency",
    "L2 / Texture / MMU",
    "Raytracing",
]

# Base UUID offset for counter tracks (must not collide with other track UUIDs)
_COUNTER_UUID_BASE = 0x6700_0000


def _counter_unit(name: str) -> str:
    """Return a Perfetto unit_name string for a counter based on its name."""
    if any(
        kw in name
        for kw in ("Utilization", "Limiter", "Percent", "Rate",
                    "Occupancy", "Ratio", "Miss Rate", "Compression")
    ):
        return "percent"
    if any(kw in name for kw in ("Bytes", "Bandwidth")):
        return "bytes"
    return ""


def _add_gpu_counters(trace: Any, pb: Any, counters: dict[str, Any]) -> None:
    """Append grouped counter tracks to a Trace protobuf.

    Creates collapsible track groups in the Perfetto UI using
    TrackDescriptor parent/child relationships with TYPE_COUNTER
    TrackEvents.

    Args:
        trace: pb.Trace() object to append to.
        pb: The perfetto_trace_pb2 module.
        counters: Output from read_gputrace_counters().
    """
    counter_names: list[str] = counters["counter_names"]
    num_samples: int = counters["num_samples"]
    timestamps_ns: list[int] = counters["timestamps_ns"]
    samples: list[list[float]] = counters["samples"]
    num_counters = len(counter_names)

    # Identify non-zero counter indices (at least one sample > 0)
    nonzero_indices: list[int] = []
    for c in range(num_counters):
        if any(samples[s][c] != 0.0 for s in range(num_samples)):
            nonzero_indices.append(c)

    if not nonzero_indices:
        return

    # Assign each counter to a group
    group_rank = {g: i for i, g in enumerate(_GROUP_ORDER)}
    ungrouped_name = "Other Counters"

    # Collect which groups are actually used
    used_groups: dict[str, list[int]] = {}  # group_name → [counter indices]
    for c in nonzero_indices:
        name = counter_names[c]
        group = _COUNTER_GROUP_MAP.get(name, ungrouped_name)
        used_groups.setdefault(group, []).append(c)

    # Sort groups: known order first, then "Other Counters" last
    sorted_groups = sorted(
        used_groups.keys(),
        key=lambda g: (group_rank.get(g, len(_GROUP_ORDER)), g),
    )

    # Allocate UUIDs for groups and counters
    group_uuids: dict[str, int] = {}
    counter_uuids: dict[int, int] = {}
    next_uuid = _COUNTER_UUID_BASE

    for group_name in sorted_groups:
        group_uuids[group_name] = next_uuid
        next_uuid += 1
        for c in used_groups[group_name]:
            counter_uuids[c] = next_uuid
            next_uuid += 1

    # Top-level "GPU Counters" track that groups everything
    root_uuid = _COUNTER_UUID_BASE - 1
    root_pkt = trace.packet.add()
    root_pkt.timestamp = 0
    root_pkt.trusted_packet_sequence_id = _COUNTER_SEQ
    root_td = root_pkt.track_descriptor
    root_td.uuid = root_uuid
    root_td.name = "GPU Counters"
    root_td.child_ordering = pb.TrackDescriptor.EXPLICIT

    # Emit TrackDescriptor packets for groups and counters
    for group_rank_idx, group_name in enumerate(sorted_groups):
        # Parent group track
        pkt = trace.packet.add()
        pkt.timestamp = 0
        pkt.trusted_packet_sequence_id = _COUNTER_SEQ
        td = pkt.track_descriptor
        td.uuid = group_uuids[group_name]
        td.parent_uuid = root_uuid
        td.name = group_name
        td.sibling_order_rank = group_rank_idx
        td.child_ordering = pb.TrackDescriptor.EXPLICIT

        # Child counter tracks
        for child_rank, c in enumerate(used_groups[group_name]):
            name = counter_names[c]
            pkt = trace.packet.add()
            pkt.timestamp = 0
            pkt.trusted_packet_sequence_id = _COUNTER_SEQ
            td = pkt.track_descriptor
            td.uuid = counter_uuids[c]
            td.parent_uuid = group_uuids[group_name]
            td.name = name
            td.sibling_order_rank = child_rank
            unit = _counter_unit(name)
            if unit:
                td.counter.unit_name = unit

    # Emit counter values as TrackEvent TYPE_COUNTER packets
    for s in range(num_samples):
        ts = timestamps_ns[s]
        for c in nonzero_indices:
            pkt = trace.packet.add()
            pkt.timestamp = ts
            pkt.trusted_packet_sequence_id = _COUNTER_SEQ
            pkt.track_event.type = pb.TrackEvent.TYPE_COUNTER
            pkt.track_event.track_uuid = counter_uuids[c]
            pkt.track_event.double_counter_value = samples[s][c]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GPU trace timeline as Perfetto-compatible trace file."
    )
    parser.add_argument("gputrace", help="Path to .gputrace file")
    parser.add_argument(
        "-o", "--output",
        help="Output path (default: <input_stem>_perfetto.<ext>)",
    )
    parser.add_argument(
        "--group-by", choices=["pipeline", "cb"], default="pipeline",
        help="Track grouping: 'pipeline' (default) groups by kernel name, "
             "'cb' groups by command buffer/encoder.",
    )
    parser.add_argument(
        "--format", choices=["json", "pftrace"], default="json",
        help="Output format: 'json' (default) for Chrome Trace Event JSON, "
             "'pftrace' for Perfetto protobuf binary.",
    )
    parser.add_argument(
        "--counters", action="store_true",
        help="Include GPU performance counter tracks (requires shader profiling data)",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="If no counter data found, open gputrace in Xcode and click "
             "Replay to trigger shader profiling (requires accessibility permissions)",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open ui.perfetto.dev in browser after export",
    )
    args = parser.parse_args()

    import os

    # gputrace_timeline loads Apple GPU frameworks at import time, which
    # requires DYLD_FRAMEWORK_PATH to be set. Re-exec if missing.
    _shared_fw = "/Applications/Xcode.app/Contents/SharedFrameworks"
    if os.environ.get("DYLD_FRAMEWORK_PATH") != _shared_fw:
        os.environ["DYLD_FRAMEWORK_PATH"] = _shared_fw
        os.execv(sys.executable, [sys.executable] + sys.argv)

    tools_dir = str(Path(__file__).parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    from gputrace_timeline import read_gputrace

    trace_data = read_gputrace(args.gputrace)
    if trace_data is None:
        log.error("Failed to read %s", args.gputrace)
        sys.exit(1)

    # Extract GPU performance counters if requested
    counter_data: dict[str, Any] | None = None
    if args.counters:
        if args.format != "pftrace":
            log.warning("--counters is only supported with --format pftrace; ignoring")
        else:
            from gputrace_timeline import read_gputrace_counters
            counter_data = read_gputrace_counters(
                args.gputrace, replay=args.replay,
            )
            if counter_data is not None:
                # Filter to non-zero counters for summary
                num_counters = len(counter_data["counter_names"])
                num_nonzero = sum(
                    1 for c in range(num_counters)
                    if any(
                        counter_data["samples"][s][c] != 0.0
                        for s in range(counter_data["num_samples"])
                    )
                )
                log.info(
                    "GPU counters: %d total, %d non-zero, %d samples",
                    num_counters, num_nonzero, counter_data["num_samples"],
                )
            else:
                log.warning("No GPU counter data found (shader profiling not enabled?)")

    ext = ".pftrace" if args.format == "pftrace" else ".json"
    output_path = args.output
    if output_path is None:
        stem = Path(args.gputrace).stem
        output_path = f"{stem}_perfetto{ext}"

    if args.format == "pftrace":
        trace_bytes = timeline_to_pftrace(
            trace_data, group_by=args.group_by, counters=counter_data,
        )
        with open(output_path, "wb") as f:
            f.write(trace_bytes)
        log.info("Wrote %d bytes to %s", len(trace_bytes), output_path)
    else:
        perfetto = timeline_to_perfetto(trace_data, group_by=args.group_by)
        with open(output_path, "w") as f:
            json.dump(perfetto, f, indent=2)
        num_events = len(perfetto["traceEvents"])
        log.info("Wrote %d events to %s", num_events, output_path)

    log.info("Open https://ui.perfetto.dev and drag in the file to view.")

    if args.open:
        webbrowser.open("https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
