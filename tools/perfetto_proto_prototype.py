# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "perfetto",
# ]
# ///
"""Prototype: GpuRenderStageEvent + TrackEvent in Perfetto protobuf format.

Tests whether GpuRenderStageEvent renders with special GPU UI treatment
in Perfetto, and whether mixing it with TrackEvent (for barriers/flows) works.

Uses synthetic trace data (same fixture as tests/test_perfetto.py) — no
Apple frameworks needed.

Usage:
    uv run tools/perfetto_proto_prototype.py [-o output.pftrace]
    uv run tools/perfetto_proto_prototype.py --open
"""
from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as pb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic trace data (matches _COMPLEX_DATA from tests/test_perfetto.py)
# ---------------------------------------------------------------------------

TRACE_DATA = {
    "events": [
        # CB0, Encoder 0
        {"type": "dispatch", "kernel": "fill", "index": 1, "encoder_idx": 0,
         "threadgroups": (8, 1, 1), "threads_per_threadgroup": (64, 1, 1)},
        {"type": "barrier", "scope": "buffers", "index": 2, "encoder_idx": 0},
        {"type": "dispatch", "kernel": "reduce", "index": 3, "encoder_idx": 0},
        # CB0, Encoder 1
        {"type": "dispatch", "kernel": "scatter", "index": 6, "encoder_idx": 1},
        # CB1, Encoder 2
        {"type": "dispatch", "kernel": "matmul", "index": 10, "encoder_idx": 2,
         "buffers_bound": {0: 0x100, 1: 0x200, 2: 0x300},
         "dispatch_type": "threads"},
        {"type": "dispatch", "kernel": "softmax", "index": 12, "encoder_idx": 2},
    ],
    "command_buffers": [
        {"func_idx": 8, "addr": "0xcb0"},
        {"func_idx": 15, "addr": "0xcb1"},
    ],
    "compute_encoders": [
        {"encoder_idx": 0, "command_buffer_idx": 0, "addr": "0xe0"},
        {"encoder_idx": 1, "command_buffer_idx": 0, "addr": "0xe1"},
        {"encoder_idx": 2, "command_buffer_idx": 1, "addr": "0xe2"},
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_tuple(t: tuple[int, ...]) -> str:
    if len(t) == 3:
        return f"{t[0]}x{t[1]}x{t[2]}"
    return str(t)


# ---------------------------------------------------------------------------
# Trace builder
# ---------------------------------------------------------------------------

# Stable UUIDs for tracks
_PROCESS_UUID = 1
_BARRIER_TRACK_UUID = 100
_ENCODER_TRACK_BASE = 200  # encoder N -> 200+N

# Sequence IDs (one per "writer" — GPU events vs TrackEvents)
_GPU_SEQ = 1
_TRACK_SEQ = 2


def build_trace(data: dict) -> bytes:
    """Build a Perfetto .pftrace from synthetic trace data."""
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
    # Use the newer InternedData.gpu_specifications approach rather than
    # the deprecated GpuRenderStageEvent.specifications embedded field.
    #
    # Each hw_queue gets an InternedGpuRenderStageSpecification with
    # category=COMPUTE. The stage ("Compute Dispatch") also gets one.
    #
    # Convention: hw_queue iids start at 1; stage iid = 100.
    STAGE_IID = 100

    intern_pkt = trace.packet.add()
    intern_pkt.trusted_packet_sequence_id = _GPU_SEQ
    intern_pkt.timestamp = 0
    # Mark start of incremental state
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
        gpu.duration = 1000  # 1µs in ns
        gpu.submission_id = cb_idx

        # command_buffer_handle is uint64 — parse hex addr
        cb_addr = cb_addrs.get(cb_idx, "0x0")
        gpu.command_buffer_handle = int(cb_addr, 16)

        # Extra data
        tg = ev.get("threadgroups")
        if tg:
            ed = gpu.extra_data.add()
            ed.name = "threadgroups"
            ed.value = _fmt_tuple(tg)

        tpt = ev.get("threads_per_threadgroup")
        if tpt:
            ed = gpu.extra_data.add()
            ed.name = "threads_per_threadgroup"
            ed.value = _fmt_tuple(tpt)

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

    # -- 3. TrackDescriptor packets (process + barrier/encoder tracks) ------

    # Process track (groups all TrackEvent tracks)
    proc_td = trace.packet.add()
    proc_td.trusted_packet_sequence_id = _TRACK_SEQ
    proc_td.track_descriptor.uuid = _PROCESS_UUID
    proc_td.track_descriptor.name = "GPU Compute (TrackEvent)"
    proc_td.track_descriptor.process.pid = 1
    proc_td.track_descriptor.process.process_name = "GPU Compute (TrackEvent)"

    # Barrier track
    barrier_td = trace.packet.add()
    barrier_td.trusted_packet_sequence_id = _TRACK_SEQ
    barrier_td.track_descriptor.uuid = _BARRIER_TRACK_UUID
    barrier_td.track_descriptor.parent_uuid = _PROCESS_UUID
    barrier_td.track_descriptor.name = "Barriers"

    # Encoder tracks
    enc_func_ranges: dict[int, tuple[int, int]] = {}
    for ev in data.get("events", []):
        func_idx = ev.get("index", 0)
        enc_idx = ev.get("encoder_idx", 0)
        if enc_idx in enc_func_ranges:
            lo, hi = enc_func_ranges[enc_idx]
            enc_func_ranges[enc_idx] = (min(lo, func_idx), max(hi, func_idx))
        else:
            enc_func_ranges[enc_idx] = (func_idx, func_idx)

    for enc in data.get("compute_encoders", []):
        enc_idx = enc["encoder_idx"]
        cb_idx = enc.get("command_buffer_idx", 0)
        addr = enc.get("addr", "")

        enc_td = trace.packet.add()
        enc_td.trusted_packet_sequence_id = _TRACK_SEQ
        enc_uuid = _ENCODER_TRACK_BASE + enc_idx
        enc_td.track_descriptor.uuid = enc_uuid
        enc_td.track_descriptor.parent_uuid = _PROCESS_UUID
        label = f"Encoder #{enc_idx}"
        if addr:
            label += f" ({addr})"
        label += f" [CB {cb_idx}]"
        enc_td.track_descriptor.name = label

    # -- 4. TrackEvent — barriers (TYPE_INSTANT) ----------------------------

    # First TrackEvent packet needs incremental state cleared
    first_track_event = True

    for ev in data.get("events", []):
        if ev.get("type") != "barrier":
            continue

        scope = ev.get("scope", "buffers")
        func_idx = ev.get("index", 0)
        enc_idx = ev.get("encoder_idx", 0)
        cb_idx = encoder_to_cb.get(enc_idx, 0)

        pkt = trace.packet.add()
        pkt.timestamp = func_idx * 1000
        pkt.trusted_packet_sequence_id = _TRACK_SEQ
        if first_track_event:
            pkt.sequence_flags = 1  # SEQ_INCREMENTAL_STATE_CLEARED
            first_track_event = False

        te = pkt.track_event
        te.type = pb.TrackEvent.TYPE_INSTANT
        te.track_uuid = _BARRIER_TRACK_UUID
        te.name = f"barrier ({scope})"

        da_scope = te.debug_annotations.add()
        da_scope.name = "scope"
        da_scope.string_value = scope

        da_enc = te.debug_annotations.add()
        da_enc.name = "encoder"
        da_enc.int_value = enc_idx

        da_cb = te.debug_annotations.add()
        da_cb.name = "cb"
        da_cb.int_value = cb_idx

    # -- 5. TrackEvent — encoder spans (SLICE_BEGIN / SLICE_END) ------------

    for enc in data.get("compute_encoders", []):
        enc_idx = enc["encoder_idx"]
        if enc_idx not in enc_func_ranges:
            continue

        lo, hi = enc_func_ranges[enc_idx]
        enc_uuid = _ENCODER_TRACK_BASE + enc_idx
        cb_idx = enc.get("command_buffer_idx", 0)
        addr = enc.get("addr", "")
        label = f"Encoder #{enc_idx}"
        if addr:
            label += f" ({addr})"

        # SLICE_BEGIN
        begin_pkt = trace.packet.add()
        begin_pkt.timestamp = lo * 1000
        begin_pkt.trusted_packet_sequence_id = _TRACK_SEQ
        if first_track_event:
            begin_pkt.sequence_flags = 1
            first_track_event = False
        begin_te = begin_pkt.track_event
        begin_te.type = pb.TrackEvent.TYPE_SLICE_BEGIN
        begin_te.track_uuid = enc_uuid
        begin_te.name = label

        da = begin_te.debug_annotations.add()
        da.name = "cb"
        da.int_value = cb_idx

        # SLICE_END
        end_pkt = trace.packet.add()
        end_pkt.timestamp = (hi + 1) * 1000  # exclusive end
        end_pkt.trusted_packet_sequence_id = _TRACK_SEQ
        end_te = end_pkt.track_event
        end_te.type = pb.TrackEvent.TYPE_SLICE_END
        end_te.track_uuid = enc_uuid

    return trace.SerializeToString()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototype: GpuRenderStageEvent + TrackEvent in .pftrace format.",
    )
    parser.add_argument(
        "-o", "--output",
        default="/tmp/claude/gpu_proto_test.pftrace",
        help="Output .pftrace path (default: /tmp/claude/gpu_proto_test.pftrace)",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open ui.perfetto.dev in browser after export",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    trace_bytes = build_trace(TRACE_DATA)
    output.write_bytes(trace_bytes)

    log.info("Wrote %d bytes to %s", len(trace_bytes), output)
    log.info("Contents:")
    log.info("  - %d GpuRenderStageEvent packets (dispatches)",
             sum(1 for e in TRACE_DATA["events"] if e["type"] == "dispatch"))
    log.info("  - %d TrackEvent packets (barriers)",
             sum(1 for e in TRACE_DATA["events"] if e["type"] == "barrier"))
    log.info("  - %d TrackEvent packets (encoder spans)",
             len(TRACE_DATA["compute_encoders"]) * 2)
    log.info("  - %d unique kernels interned as GPU hw_queue specs",
             len({e["kernel"] for e in TRACE_DATA["events"] if e.get("kernel")}))
    log.info("")
    log.info("Open https://ui.perfetto.dev and drag in the file to view.")
    log.info("Check:")
    log.info("  1. Do GpuRenderStageEvent packets get a 'GPU' section?")
    log.info("  2. Do hw_queue/stage names from interned specs appear?")
    log.info("  3. Do TrackEvent barriers render alongside GPU events?")
    log.info("  4. Does extra_data show up in event details panel?")

    if args.open:
        webbrowser.open("https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
