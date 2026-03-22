"""Tests for gputrace_perfetto Perfetto/Chrome Trace Event export.

These tests exercise the timeline_to_perfetto conversion using synthetic
trace data (same format as read_gputrace() output) without requiring
Apple frameworks or .gputrace files.
"""
from __future__ import annotations

import json

import pytest

# Import from the package tools module
from apple_profiler.tools.gputrace_perfetto import timeline_to_perfetto  # noqa: E402


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


# A reusable multi-CB, multi-encoder trace for integration tests.
_COMPLEX_DATA = _make_trace_data(
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


class TestBasicOutput:
    """Test output structure and validity."""

    def test_empty_trace_pipeline(self):
        result = timeline_to_perfetto(_make_trace_data(), group_by="pipeline")
        assert "traceEvents" in result
        assert len(result["traceEvents"]) == 0

    def test_empty_trace_cb(self):
        result = timeline_to_perfetto(_make_trace_data(), group_by="cb")
        assert "traceEvents" in result
        assert len(result["traceEvents"]) == 0

    def test_invalid_group_by(self):
        with pytest.raises(ValueError, match="Unknown group_by"):
            timeline_to_perfetto(_make_trace_data(), group_by="nope")

    @pytest.mark.parametrize("mode", ["pipeline", "cb"])
    def test_output_is_json_serializable(self, mode: str):
        data = _make_trace_data(
            events=[{
                "type": "dispatch", "kernel": "my_kernel", "index": 10,
                "encoder_idx": 0, "buffers_bound": {0: 0x100},
                "threadgroups": (4, 1, 1),
                "threads_per_threadgroup": (256, 1, 1),
                "dispatch_type": "threadgroups",
            }],
            command_buffers=[{
                "func_idx": 15, "addr": "0xabc",
                "dispatches": [{"index": 10}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "addr": "0xdef", "dispatches": [{"index": 10}],
            }],
        )
        result = timeline_to_perfetto(data, group_by=mode)
        parsed = json.loads(json.dumps(result))
        assert "traceEvents" in parsed


# -----------------------------------------------------------------------
# Pipeline-grouped mode (default)
# -----------------------------------------------------------------------


class TestPipelineDispatches:
    """Dispatch events in pipeline-grouped mode."""

    def test_single_dispatch_kernel_as_process(self):
        data = _make_trace_data(
            events=[{
                "type": "dispatch", "kernel": "matmul", "index": 5,
                "encoder_idx": 0, "buffers_bound": {0: 0x100, 1: 0x200},
                "threadgroups": (4, 1, 1),
                "threads_per_threadgroup": (256, 1, 1),
                "dispatch_type": "threadgroups",
            }],
            command_buffers=[{
                "func_idx": 10, "addr": "0xabc",
                "dispatches": [{"index": 5}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 5}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="pipeline")
        x_events = [e for e in result["traceEvents"] if e["ph"] == "X"]
        assert len(x_events) == 1

        evt = x_events[0]
        assert evt["name"] == "matmul"
        assert evt["ts"] == 5
        assert evt["dur"] == 1
        assert evt["tid"] == 0  # single track per kernel
        assert evt["args"]["func_idx"] == 5
        assert evt["args"]["threadgroups"] == "4x1x1"
        assert evt["args"]["buffers_bound"] == 2
        assert evt["args"]["cb"] == 0
        assert evt["args"]["cb_addr"] == "0xabc"
        assert evt["args"]["encoder"] == 0

    def test_same_kernel_same_process(self):
        """Two dispatches of the same kernel share a process."""
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k", "index": 1,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k", "index": 5,
                 "encoder_idx": 0},
            ],
            command_buffers=[{
                "func_idx": 10,
                "dispatches": [{"index": 1}, {"index": 5}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 1}, {"index": 5}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="pipeline")
        x_events = [e for e in result["traceEvents"] if e["ph"] == "X"]
        assert len(x_events) == 2
        assert x_events[0]["pid"] == x_events[1]["pid"]

    def test_different_kernels_different_processes(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "kA", "index": 1,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "kB", "index": 5,
                 "encoder_idx": 0},
            ],
            command_buffers=[{
                "func_idx": 10,
                "dispatches": [{"index": 1}, {"index": 5}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 1}, {"index": 5}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="pipeline")
        x_events = [e for e in result["traceEvents"] if e["ph"] == "X"]
        assert x_events[0]["pid"] != x_events[1]["pid"]


class TestPipelineBarriers:
    """Barriers in pipeline-grouped mode."""

    def test_barrier_on_dedicated_process(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k", "index": 1,
                 "encoder_idx": 0},
                {"type": "barrier", "scope": "buffers", "index": 2,
                 "encoder_idx": 0},
            ],
            command_buffers=[{
                "func_idx": 10,
                "dispatches": [{"index": 1}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 1}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="pipeline")
        i_events = [e for e in result["traceEvents"] if e["ph"] == "i"]
        assert len(i_events) == 1

        evt = i_events[0]
        assert evt["pid"] == -1  # dedicated barrier process
        assert evt["args"]["scope"] == "buffers"
        assert evt["args"]["encoder"] == 0
        assert evt["args"]["cb"] == 0


class TestPipelineMetadata:
    """Metadata events in pipeline-grouped mode."""

    def test_process_name_is_kernel(self):
        data = _make_trace_data(
            events=[{
                "type": "dispatch", "kernel": "my_kernel", "index": 1,
                "encoder_idx": 0,
            }],
            command_buffers=[{
                "func_idx": 5, "dispatches": [{"index": 1}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 1}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="pipeline")
        proc_meta = [
            e for e in result["traceEvents"]
            if e["ph"] == "M" and e["name"] == "process_name"
        ]
        assert len(proc_meta) == 1
        assert proc_meta[0]["args"]["name"] == "my_kernel"

    def test_process_sort_order_by_first_appearance(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "late", "index": 10,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "early", "index": 2,
                 "encoder_idx": 0},
            ],
            command_buffers=[{
                "func_idx": 15,
                "dispatches": [{"index": 10}, {"index": 2}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 10}, {"index": 2}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="pipeline")
        sort_meta = [
            e for e in result["traceEvents"]
            if e["ph"] == "M" and e["name"] == "process_sort_index"
        ]
        by_pid = {e["pid"]: e["args"]["sort_index"] for e in sort_meta}
        # "early" has lower func_idx, should sort first
        x_events = [e for e in result["traceEvents"] if e["ph"] == "X"]
        early_pid = next(e["pid"] for e in x_events if e["name"] == "early")
        late_pid = next(e["pid"] for e in x_events if e["name"] == "late")
        assert by_pid[early_pid] < by_pid[late_pid]

    def test_barrier_process_metadata(self):
        data = _make_trace_data(
            events=[
                {"type": "barrier", "scope": "buffers", "index": 1,
                 "encoder_idx": 0},
            ],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [],
            }],
        )
        result = timeline_to_perfetto(data, group_by="pipeline")
        proc_meta = [
            e for e in result["traceEvents"]
            if e["ph"] == "M" and e["name"] == "process_name"
        ]
        assert len(proc_meta) == 1
        assert proc_meta[0]["args"]["name"] == "Barriers"
        assert proc_meta[0]["pid"] == -1


class TestPipelineComplex:
    """Integration test for pipeline-grouped mode."""

    def test_realistic_trace(self):
        result = timeline_to_perfetto(_COMPLEX_DATA, group_by="pipeline")
        events = result["traceEvents"]

        x_events = [e for e in events if e["ph"] == "X"]
        i_events = [e for e in events if e["ph"] == "i"]
        m_events = [e for e in events if e["ph"] == "M"]

        # 5 dispatches
        assert len(x_events) == 5
        # 1 barrier
        assert len(i_events) == 1

        # 5 unique kernels → 5 processes + barrier process = 6 process names
        proc_names = [
            e for e in m_events if e["name"] == "process_name"
        ]
        assert len(proc_names) == 6  # fill, reduce, scatter, matmul, softmax, Barriers

        # Each kernel has its own pid
        kernel_pids = {e["name"]: e["pid"] for e in x_events}
        assert len(set(kernel_pids.values())) == 5

        # No B/E spans in pipeline mode
        assert not any(e["ph"] in ("B", "E") for e in events)

        # Valid JSON
        assert len(json.dumps(result)) > 0


# -----------------------------------------------------------------------
# CB-grouped mode
# -----------------------------------------------------------------------


class TestCBDispatchEvents:
    """Dispatch events in CB-grouped mode."""

    def test_single_dispatch(self):
        data = _make_trace_data(
            events=[{
                "type": "dispatch", "kernel": "matmul", "index": 5,
                "encoder_idx": 0, "buffers_bound": {0: 0x100, 1: 0x200},
                "threadgroups": (4, 1, 1),
                "threads_per_threadgroup": (256, 1, 1),
                "dispatch_type": "threadgroups",
            }],
            command_buffers=[{
                "func_idx": 10, "addr": "0xabc",
                "dispatches": [{"index": 5}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 5}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="cb")
        x_events = [e for e in result["traceEvents"] if e["ph"] == "X"]
        assert len(x_events) == 1

        evt = x_events[0]
        assert evt["name"] == "matmul"
        assert evt["pid"] == 0  # CB index
        assert evt["tid"] == 0  # encoder index
        assert evt["ts"] == 5
        assert evt["args"]["threadgroups"] == "4x1x1"

    def test_multiple_dispatches_ordering(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 3,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 7,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k3", "index": 12,
                 "encoder_idx": 0},
            ],
            command_buffers=[{
                "func_idx": 20,
                "dispatches": [{"index": 3}, {"index": 7}, {"index": 12}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 3}, {"index": 7}, {"index": 12}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="cb")
        x_events = [e for e in result["traceEvents"] if e["ph"] == "X"]
        assert [e["ts"] for e in x_events] == [3, 7, 12]
        assert all(e["pid"] == 0 for e in x_events)


class TestCBBarrierEvents:
    """Barrier events in CB-grouped mode."""

    def test_barrier_as_instant_event(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 1,
                 "encoder_idx": 0},
                {"type": "barrier", "scope": "buffers", "index": 2,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 3,
                 "encoder_idx": 0},
            ],
            command_buffers=[{
                "func_idx": 10,
                "dispatches": [{"index": 1}, {"index": 3}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 1}, {"index": 3}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="cb")
        i_events = [e for e in result["traceEvents"] if e["ph"] == "i"]
        assert len(i_events) == 1

        evt = i_events[0]
        assert evt["name"] == "barrier (buffers)"
        assert evt["ts"] == 2
        assert evt["s"] == "t"
        assert evt["pid"] == 0  # same CB
        assert evt["tid"] == 0  # same encoder


class TestCBEncoderSpans:
    """Encoder B/E spans in CB-grouped mode."""

    def test_encoder_wraps_dispatches(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 5,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 10,
                 "encoder_idx": 0},
            ],
            command_buffers=[{
                "func_idx": 15,
                "dispatches": [{"index": 5}, {"index": 10}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0, "addr": "0xfoo",
                "dispatches": [{"index": 5}, {"index": 10}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="cb")

        enc_b = [
            e for e in result["traceEvents"]
            if e["ph"] == "B" and e["cat"] == "encoder"
        ]
        enc_e = [
            e for e in result["traceEvents"]
            if e["ph"] == "E" and e["cat"] == "encoder"
        ]
        assert len(enc_b) == 1
        assert len(enc_e) == 1
        assert enc_b[0]["ts"] == 5
        assert enc_e[0]["ts"] == 11
        assert "Encoder #0" in enc_b[0]["name"]
        assert "(0xfoo)" in enc_b[0]["name"]


class TestCBSpans:
    """CB wrapper spans in CB-grouped mode."""

    def test_cb_wraps_all_encoders(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 3,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 15,
                 "encoder_idx": 1},
            ],
            command_buffers=[{
                "func_idx": 20, "addr": "0xcb0",
                "dispatches": [{"index": 3}, {"index": 15}],
            }],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0,
                 "dispatches": [{"index": 3}]},
                {"encoder_idx": 1, "command_buffer_idx": 0,
                 "dispatches": [{"index": 15}]},
            ],
        )
        result = timeline_to_perfetto(data, group_by="cb")

        cb_b = [
            e for e in result["traceEvents"]
            if e["ph"] == "B" and e["cat"] == "command_buffer"
        ]
        cb_e = [
            e for e in result["traceEvents"]
            if e["ph"] == "E" and e["cat"] == "command_buffer"
        ]
        assert len(cb_b) == 1
        assert cb_b[0]["ts"] == 3
        assert cb_e[0]["ts"] == 16
        assert cb_b[0]["tid"] == -1  # dedicated CB overview tid
        assert "CB #0" in cb_b[0]["name"]


class TestCBMultipleCBs:
    """Multiple CBs in CB-grouped mode."""

    def test_two_cbs_separate_processes(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 1,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 5,
                 "encoder_idx": 1},
            ],
            command_buffers=[
                {"func_idx": 10, "addr": "0xaa",
                 "dispatches": [{"index": 1}]},
                {"func_idx": 12, "addr": "0xbb",
                 "dispatches": [{"index": 5}]},
            ],
            compute_encoders=[
                {"encoder_idx": 0, "command_buffer_idx": 0,
                 "dispatches": [{"index": 1}]},
                {"encoder_idx": 1, "command_buffer_idx": 1,
                 "dispatches": [{"index": 5}]},
            ],
        )
        result = timeline_to_perfetto(data, group_by="cb")
        x_events = [
            e for e in result["traceEvents"] if e["ph"] == "X"
        ]
        pids = {e["pid"] for e in x_events}
        assert pids == {0, 1}


class TestCBMetadata:
    """Metadata in CB-grouped mode."""

    def test_process_name_is_cb(self):
        data = _make_trace_data(
            events=[{
                "type": "dispatch", "kernel": "k1", "index": 1,
                "encoder_idx": 0,
            }],
            command_buffers=[{
                "func_idx": 5, "addr": "0xabc",
                "dispatches": [{"index": 1}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 1}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="cb")
        proc_meta = [
            e for e in result["traceEvents"]
            if e["ph"] == "M" and e["name"] == "process_name"
        ]
        assert len(proc_meta) == 1
        assert "CB #0" in proc_meta[0]["args"]["name"]
        assert "(0xabc)" in proc_meta[0]["args"]["name"]

    def test_thread_name_is_encoder(self):
        data = _make_trace_data(
            events=[{
                "type": "dispatch", "kernel": "k1", "index": 1,
                "encoder_idx": 0,
            }],
            command_buffers=[{
                "func_idx": 5, "dispatches": [{"index": 1}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0, "addr": "0xenc",
                "dispatches": [{"index": 1}],
            }],
        )
        result = timeline_to_perfetto(data, group_by="cb")
        thread_meta = [
            e for e in result["traceEvents"]
            if e["ph"] == "M" and e["name"] == "thread_name"
            and e.get("tid", 0) != -1
        ]
        enc_meta = [m for m in thread_meta if m["tid"] == 0]
        assert len(enc_meta) == 1
        assert "Encoder #0" in enc_meta[0]["args"]["name"]


class TestCBComplex:
    """Integration test for CB-grouped mode."""

    def test_realistic_trace(self):
        result = timeline_to_perfetto(_COMPLEX_DATA, group_by="cb")
        events = result["traceEvents"]

        x_events = [e for e in events if e["ph"] == "X"]
        i_events = [e for e in events if e["ph"] == "i"]
        b_events = [e for e in events if e["ph"] == "B"]
        e_events = [e for e in events if e["ph"] == "E"]
        m_events = [e for e in events if e["ph"] == "M"]

        assert len(x_events) == 5
        assert len(i_events) == 1
        assert len(b_events) == 5  # 3 encoders + 2 CBs
        assert len(e_events) == 5

        # CB0 dispatches
        cb0 = [e for e in x_events if e["pid"] == 0]
        assert len(cb0) == 3
        # CB1 dispatches
        cb1 = [e for e in x_events if e["pid"] == 1]
        assert len(cb1) == 2

        # 2 CB process names
        proc_names = [
            e for e in m_events if e["name"] == "process_name"
        ]
        assert len(proc_names) == 2

        assert len(json.dumps(result)) > 0


# -----------------------------------------------------------------------
# Default mode
# -----------------------------------------------------------------------


class TestDefaultMode:
    """Verify that default group_by is 'pipeline'."""

    def test_default_is_pipeline(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "kA", "index": 1,
                 "encoder_idx": 0},
                {"type": "dispatch", "kernel": "kB", "index": 5,
                 "encoder_idx": 0},
            ],
            command_buffers=[{
                "func_idx": 10,
                "dispatches": [{"index": 1}, {"index": 5}],
            }],
            compute_encoders=[{
                "encoder_idx": 0, "command_buffer_idx": 0,
                "dispatches": [{"index": 1}, {"index": 5}],
            }],
        )
        # Default call (no group_by) should match explicit pipeline
        default = timeline_to_perfetto(data)
        explicit = timeline_to_perfetto(data, group_by="pipeline")
        assert default == explicit
