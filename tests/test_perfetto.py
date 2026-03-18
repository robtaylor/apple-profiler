"""Tests for gputrace_perfetto Perfetto/Chrome Trace Event export.

These tests exercise the timeline_to_perfetto conversion using synthetic
trace data (same format as read_gputrace() output) without requiring
Apple frameworks or .gputrace files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add tools dir to import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from gputrace_perfetto import timeline_to_perfetto  # noqa: E402


def _make_trace_data(
    events: list[dict] | None = None,
    command_buffers: list[dict] | None = None,
    compute_encoders: list[dict] | None = None,
) -> dict:
    """Build a synthetic read_gputrace() result dict."""
    return {
        "metadata": {},
        "total_functions": 100,
        "events": events or [],
        "kernels": {},
        "pipelines": {},
        "command_buffers": command_buffers or [],
        "compute_encoders": compute_encoders or [],
    }


class TestBasicOutput:
    """Test output structure and validity."""

    def test_empty_trace(self):
        result = timeline_to_perfetto(_make_trace_data())
        assert "traceEvents" in result
        assert isinstance(result["traceEvents"], list)
        assert len(result["traceEvents"]) == 0

    def test_output_is_json_serializable(self):
        data = _make_trace_data(
            events=[
                {
                    "type": "dispatch",
                    "kernel": "my_kernel",
                    "index": 10,
                    "encoder_idx": 0,
                    "buffers_bound": {0: 0x100},
                    "threadgroups": (4, 1, 1),
                    "threads_per_threadgroup": (256, 1, 1),
                    "dispatch_type": "threadgroups",
                },
            ],
            command_buffers=[{"func_idx": 15, "addr": "0xabc", "dispatches": [{"index": 10}]}],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "addr": "0xdef",
                 "dispatches": [{"index": 10}]}
            ],
        )
        result = timeline_to_perfetto(data)
        # Should not raise
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert "traceEvents" in parsed


class TestDispatchEvents:
    """Test dispatch → X (complete) events."""

    def test_single_dispatch(self):
        data = _make_trace_data(
            events=[
                {
                    "type": "dispatch",
                    "kernel": "matmul",
                    "index": 5,
                    "encoder_idx": 0,
                    "buffers_bound": {0: 0x100, 1: 0x200},
                    "threadgroups": (4, 1, 1),
                    "threads_per_threadgroup": (256, 1, 1),
                    "dispatch_type": "threadgroups",
                },
            ],
            command_buffers=[{"func_idx": 10, "addr": "0xabc", "dispatches": [{"index": 5}]}],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 5}]}
            ],
        )
        result = timeline_to_perfetto(data)
        trace_events = result["traceEvents"]

        # Find the dispatch event
        x_events = [e for e in trace_events if e["ph"] == "X"]
        assert len(x_events) == 1

        evt = x_events[0]
        assert evt["name"] == "matmul"
        assert evt["cat"] == "dispatch"
        assert evt["ts"] == 5
        assert evt["dur"] == 1
        assert evt["pid"] == 0  # CB index
        assert evt["tid"] == 0  # encoder index
        assert evt["args"]["func_idx"] == 5
        assert evt["args"]["threadgroups"] == "4x1x1"
        assert evt["args"]["threads_per_threadgroup"] == "256x1x1"
        assert evt["args"]["buffers_bound"] == 2
        assert evt["args"]["dispatch_type"] == "threadgroups"

    def test_multiple_dispatches_ordering(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 3, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 7, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k3", "index": 12, "encoder_idx": 0},
            ],
            command_buffers=[
                {"func_idx": 20, "dispatches": [{"index": 3}, {"index": 7}, {"index": 12}]}
            ],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0,
                 "dispatches": [{"index": 3}, {"index": 7}, {"index": 12}]}
            ],
        )
        result = timeline_to_perfetto(data)
        x_events = [e for e in result["traceEvents"] if e["ph"] == "X"]
        assert len(x_events) == 3

        # Timestamps should match func_idx
        timestamps = [e["ts"] for e in x_events]
        assert timestamps == [3, 7, 12]

        # All on same pid/tid
        assert all(e["pid"] == 0 for e in x_events)
        assert all(e["tid"] == 0 for e in x_events)

    def test_dispatch_without_optional_fields(self):
        """Dispatches without threadgroups/buffers should still work."""
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "simple", "index": 1, "encoder_idx": 0},
            ],
            command_buffers=[{"func_idx": 5, "dispatches": [{"index": 1}]}],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 1}]}
            ],
        )
        result = timeline_to_perfetto(data)
        x_events = [e for e in result["traceEvents"] if e["ph"] == "X"]
        assert len(x_events) == 1
        assert x_events[0]["name"] == "simple"
        assert "threadgroups" not in x_events[0]["args"]
        assert "threads_per_threadgroup" not in x_events[0]["args"]


class TestBarrierEvents:
    """Test barrier → i (instant) events."""

    def test_barrier_as_instant_event(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 1, "encoder_idx": 0},
                {"type": "barrier", "scope": "buffers", "index": 2, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 3, "encoder_idx": 0},
            ],
            command_buffers=[
                {"func_idx": 10, "dispatches": [{"index": 1}, {"index": 3}]}
            ],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0,
                 "dispatches": [{"index": 1}, {"index": 3}]}
            ],
        )
        result = timeline_to_perfetto(data)
        i_events = [e for e in result["traceEvents"] if e["ph"] == "i"]
        assert len(i_events) == 1

        evt = i_events[0]
        assert evt["name"] == "barrier (buffers)"
        assert evt["cat"] == "barrier"
        assert evt["ts"] == 2
        assert evt["s"] == "t"  # thread-scoped
        assert evt["args"]["scope"] == "buffers"
        assert evt["pid"] == 0
        assert evt["tid"] == 0

    def test_barrier_resource_scope(self):
        data = _make_trace_data(
            events=[
                {"type": "barrier", "scope": "resources", "index": 5, "encoder_idx": 0},
            ],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": []}
            ],
        )
        result = timeline_to_perfetto(data)
        i_events = [e for e in result["traceEvents"] if e["ph"] == "i"]
        assert len(i_events) == 1
        assert i_events[0]["name"] == "barrier (resources)"


class TestEncoderSpans:
    """Test encoder B/E wrapper events."""

    def test_encoder_wraps_dispatches(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 5, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 10, "encoder_idx": 0},
            ],
            command_buffers=[
                {"func_idx": 15, "dispatches": [{"index": 5}, {"index": 10}]}
            ],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "addr": "0xfoo",
                 "dispatches": [{"index": 5}, {"index": 10}]}
            ],
        )
        result = timeline_to_perfetto(data)

        enc_b = [e for e in result["traceEvents"] if e["ph"] == "B" and e["cat"] == "encoder"]
        enc_e = [e for e in result["traceEvents"] if e["ph"] == "E" and e["cat"] == "encoder"]
        assert len(enc_b) == 1
        assert len(enc_e) == 1

        assert enc_b[0]["ts"] == 5  # min func_idx
        assert enc_e[0]["ts"] == 11  # max func_idx + 1
        assert "Encoder #0" in enc_b[0]["name"]
        assert "(0xfoo)" in enc_b[0]["name"]

    def test_multiple_encoders(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 2, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 8, "encoder_idx": 1},
            ],
            command_buffers=[
                {"func_idx": 15, "dispatches": [{"index": 2}, {"index": 8}]}
            ],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 2}]},
                {"encoder_idx": 1, "command_buffer_idx": 0, "dispatches": [{"index": 8}]},
            ],
        )
        result = timeline_to_perfetto(data)

        enc_b = [e for e in result["traceEvents"] if e["ph"] == "B" and e["cat"] == "encoder"]
        assert len(enc_b) == 2


class TestCBSpans:
    """Test command buffer B/E wrapper events."""

    def test_cb_wraps_all_encoders(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 3, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 15, "encoder_idx": 1},
            ],
            command_buffers=[
                {"func_idx": 20, "addr": "0xcb0", "dispatches": [{"index": 3}, {"index": 15}]}
            ],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 3}]},
                {"encoder_idx": 1, "command_buffer_idx": 0, "dispatches": [{"index": 15}]},
            ],
        )
        result = timeline_to_perfetto(data)

        cb_b = [e for e in result["traceEvents"] if e["ph"] == "B" and e["cat"] == "command_buffer"]
        cb_e = [e for e in result["traceEvents"] if e["ph"] == "E" and e["cat"] == "command_buffer"]
        assert len(cb_b) == 1
        assert len(cb_e) == 1

        assert cb_b[0]["ts"] == 3
        assert cb_e[0]["ts"] == 16  # max + 1
        assert "CB #0" in cb_b[0]["name"]
        assert "(0xcb0)" in cb_b[0]["name"]

    def test_cb_uses_dedicated_tid(self):
        """CB overview spans should use tid=-1 to avoid overlap with encoder tracks."""
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 1, "encoder_idx": 0},
            ],
            command_buffers=[{"func_idx": 5, "dispatches": [{"index": 1}]}],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 1}]}
            ],
        )
        result = timeline_to_perfetto(data)

        cb_b = [e for e in result["traceEvents"] if e["ph"] == "B" and e["cat"] == "command_buffer"]
        assert len(cb_b) == 1
        assert cb_b[0]["tid"] == -1


class TestMultipleCBs:
    """Test traces with multiple command buffers."""

    def test_two_cbs_separate_processes(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 1, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 5, "encoder_idx": 1},
            ],
            command_buffers=[
                {"func_idx": 10, "addr": "0xaa", "dispatches": [{"index": 1}]},
                {"func_idx": 12, "addr": "0xbb", "dispatches": [{"index": 5}]},
            ],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 1}]},
                {"encoder_idx": 1, "command_buffer_idx": 1, "dispatches": [{"index": 5}]},
            ],
        )
        result = timeline_to_perfetto(data)
        x_events = [e for e in result["traceEvents"] if e["ph"] == "X"]
        assert len(x_events) == 2

        # Different pids (different CBs)
        pids = {e["pid"] for e in x_events}
        assert pids == {0, 1}


class TestMetadataEvents:
    """Test process/thread name metadata events."""

    def test_process_name_metadata(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 1, "encoder_idx": 0},
            ],
            command_buffers=[{"func_idx": 5, "addr": "0xabc", "dispatches": [{"index": 1}]}],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 1}]}
            ],
        )
        result = timeline_to_perfetto(data)

        proc_meta = [
            e for e in result["traceEvents"]
            if e["ph"] == "M" and e["name"] == "process_name"
        ]
        assert len(proc_meta) == 1
        assert "CB #0" in proc_meta[0]["args"]["name"]
        assert "(0xabc)" in proc_meta[0]["args"]["name"]

    def test_thread_name_metadata(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 1, "encoder_idx": 0},
            ],
            command_buffers=[{"func_idx": 5, "dispatches": [{"index": 1}]}],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "addr": "0xenc",
                 "dispatches": [{"index": 1}]}
            ],
        )
        result = timeline_to_perfetto(data)

        thread_meta = [
            e for e in result["traceEvents"]
            if e["ph"] == "M" and e["name"] == "thread_name" and e.get("tid", 0) != -1
        ]
        assert len(thread_meta) >= 1
        enc_meta = [m for m in thread_meta if m["tid"] == 0]
        assert len(enc_meta) == 1
        assert "Encoder #0" in enc_meta[0]["args"]["name"]
        assert "(0xenc)" in enc_meta[0]["args"]["name"]


class TestComplexTrace:
    """Integration-style test with a multi-CB, multi-encoder trace."""

    def test_realistic_trace(self):
        """Simulate: CB0 has 2 encoders, CB1 has 1 encoder, with barriers."""
        data = _make_trace_data(
            events=[
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
            command_buffers=[
                {"func_idx": 8, "addr": "0xcb0",
                 "dispatches": [{"index": 1}, {"index": 3}, {"index": 6}]},
                {"func_idx": 15, "addr": "0xcb1",
                 "dispatches": [{"index": 10}, {"index": 12}]},
            ],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0, "addr": "0xe0",
                 "dispatches": [{"index": 1}, {"index": 3}]},
                {"encoder_idx": 1, "command_buffer_idx": 0, "addr": "0xe1",
                 "dispatches": [{"index": 6}]},
                {"encoder_idx": 2, "command_buffer_idx": 1, "addr": "0xe2",
                 "dispatches": [{"index": 10}, {"index": 12}]},
            ],
        )
        result = timeline_to_perfetto(data)
        events = result["traceEvents"]

        # Count event types
        x_events = [e for e in events if e["ph"] == "X"]
        i_events = [e for e in events if e["ph"] == "i"]
        b_events = [e for e in events if e["ph"] == "B"]
        e_events = [e for e in events if e["ph"] == "E"]
        m_events = [e for e in events if e["ph"] == "M"]

        assert len(x_events) == 5  # fill, reduce, scatter, matmul, softmax
        assert len(i_events) == 1  # 1 barrier
        assert len(b_events) == 5  # 3 encoders + 2 CBs
        assert len(e_events) == 5

        # CB0 dispatches (encoders 0,1) should have pid=0
        cb0_dispatches = [e for e in x_events if e["pid"] == 0]
        assert len(cb0_dispatches) == 3  # fill, reduce, scatter (enc0 + enc1)

        # CB1 dispatches (encoder 2) should have pid=1
        cb1_dispatches = [e for e in x_events if e["pid"] == 1]
        assert len(cb1_dispatches) == 2  # matmul, softmax

        # Verify kernel names
        kernel_names = [e["name"] for e in x_events]
        assert "fill" in kernel_names
        assert "reduce" in kernel_names
        assert "scatter" in kernel_names
        assert "matmul" in kernel_names
        assert "softmax" in kernel_names

        # Metadata events
        proc_names = [e for e in m_events if e["name"] == "process_name"]
        assert len(proc_names) == 2  # 2 CBs

        # Verify overall JSON is valid
        serialized = json.dumps(result)
        assert len(serialized) > 0
