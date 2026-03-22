"""Tests for gputrace_perfetto Perfetto protobuf (.pftrace) export.

These tests exercise the timeline_to_pftrace conversion using synthetic
trace data (same format as read_gputrace() output) without requiring
Apple frameworks or .gputrace files.
"""
from __future__ import annotations

import pytest
from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as pb

# Import from the package tools module
from apple_profiler.tools.gputrace_perfetto import timeline_to_pftrace  # noqa: E402


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
        {"func_idx": 8, "addr": "0xcb0"},
        {"func_idx": 15, "addr": "0xcb1"},
    ],
    compute_encoders=[
        {"encoder_idx": 0, "command_buffer_idx": 0, "addr": "0xe0"},
        {"encoder_idx": 1, "command_buffer_idx": 0, "addr": "0xe1"},
        {"encoder_idx": 2, "command_buffer_idx": 1, "addr": "0xe2"},
    ],
)


def _parse_trace(raw: bytes) -> pb.Trace:
    """Parse raw protobuf bytes into a Trace message."""
    trace = pb.Trace()
    trace.ParseFromString(raw)
    return trace


def _gpu_events(trace: pb.Trace) -> list:
    """Extract packets that have gpu_render_stage_event."""
    return [p for p in trace.packet if p.HasField("gpu_render_stage_event")]


def _track_events(trace: pb.Trace) -> list:
    """Extract packets that have track_event."""
    return [p for p in trace.packet if p.HasField("track_event")]


def _track_descriptors(trace: pb.Trace) -> list:
    """Extract packets that have track_descriptor."""
    return [p for p in trace.packet if p.HasField("track_descriptor")]


def _interned_data_pkts(trace: pb.Trace) -> list:
    """Extract packets that have interned_data."""
    return [p for p in trace.packet if p.HasField("interned_data")]


# -----------------------------------------------------------------------
# Basic output
# -----------------------------------------------------------------------


class TestPftraceBasicOutput:
    """Test output structure and validity."""

    def test_empty_trace_pipeline(self):
        raw = timeline_to_pftrace(_make_trace_data(), group_by="pipeline")
        assert isinstance(raw, bytes)
        trace = _parse_trace(raw)
        # Should have interned data + track descriptors but no GPU/track events with dispatches
        gpu = _gpu_events(trace)
        assert len(gpu) == 0

    def test_empty_trace_cb(self):
        raw = timeline_to_pftrace(_make_trace_data(), group_by="cb")
        assert isinstance(raw, bytes)
        trace = _parse_trace(raw)
        te = _track_events(trace)
        assert len(te) == 0

    def test_invalid_group_by(self):
        with pytest.raises(ValueError, match="Unknown group_by"):
            timeline_to_pftrace(_make_trace_data(), group_by="nope")

    @pytest.mark.parametrize("mode", ["pipeline", "cb"])
    def test_valid_protobuf(self, mode: str):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by=mode)
        trace = _parse_trace(raw)
        assert len(trace.packet) > 0

    @pytest.mark.parametrize("mode", ["pipeline", "cb"])
    def test_roundtrip_serialization(self, mode: str):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by=mode)
        trace1 = _parse_trace(raw)
        raw2 = trace1.SerializeToString()
        trace2 = _parse_trace(raw2)
        assert len(trace2.packet) == len(trace1.packet)


# -----------------------------------------------------------------------
# Pipeline mode — GpuRenderStageEvent dispatches
# -----------------------------------------------------------------------


class TestPftracePipelineDispatches:
    """GpuRenderStageEvent packets in pipeline mode."""

    def test_dispatch_count(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        gpu = _gpu_events(trace)
        assert len(gpu) == 5  # 5 dispatches

    def test_dispatch_hw_queue_iid(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "fill", "index": 1, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "reduce", "index": 3, "encoder_idx": 0},
            ],
            command_buffers=[{"func_idx": 5}],
            compute_encoders=[{"encoder_idx": 0, "command_buffer_idx": 0}],
        )
        raw = timeline_to_pftrace(data, group_by="pipeline")
        trace = _parse_trace(raw)
        gpu = _gpu_events(trace)

        # Different kernels get different hw_queue_iid
        iids = {p.gpu_render_stage_event.hw_queue_iid for p in gpu}
        assert len(iids) == 2

    def test_dispatch_stage_iid(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        gpu = _gpu_events(trace)
        # All dispatches share the same stage_iid (100)
        stage_iids = {p.gpu_render_stage_event.stage_iid for p in gpu}
        assert stage_iids == {100}

    def test_dispatch_duration_fills_to_next_dispatch(self):
        """Duration extends to the next dispatch, skipping barriers."""
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 1, "encoder_idx": 0},
                {"type": "barrier", "scope": "buffers", "index": 2, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 5, "encoder_idx": 0},
                {"type": "barrier", "scope": "buffers", "index": 6, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k3", "index": 10, "encoder_idx": 0},
            ],
            command_buffers=[{"func_idx": 15}],
            compute_encoders=[{"encoder_idx": 0, "command_buffer_idx": 0}],
        )
        raw = timeline_to_pftrace(data, group_by="pipeline")
        trace = _parse_trace(raw)
        gpu = _gpu_events(trace)
        durations = sorted(
            (p.timestamp, p.gpu_render_stage_event.duration) for p in gpu
        )
        # idx=1 → next dispatch=5 (skip barrier@2) → dur=4*1000=4000ns
        assert durations[0] == (1000, 4000)
        # idx=5 → next dispatch=10 (skip barrier@6) → dur=5*1000=5000ns
        assert durations[1] == (5000, 5000)
        # idx=10 → last → dur=1*1000=1000ns
        assert durations[2] == (10000, 1000)

    def test_dispatch_timestamp(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k", "index": 7, "encoder_idx": 0},
            ],
            command_buffers=[{"func_idx": 10}],
            compute_encoders=[{"encoder_idx": 0, "command_buffer_idx": 0}],
        )
        raw = timeline_to_pftrace(data, group_by="pipeline")
        trace = _parse_trace(raw)
        gpu = _gpu_events(trace)
        assert gpu[0].timestamp == 7000  # index * 1000

    def test_dispatch_extra_data(self):
        data = _make_trace_data(
            events=[{
                "type": "dispatch", "kernel": "k", "index": 1, "encoder_idx": 0,
                "threadgroups": (8, 1, 1), "threads_per_threadgroup": (64, 1, 1),
                "buffers_bound": {0: 0x100, 1: 0x200},
                "dispatch_type": "threads",
            }],
            command_buffers=[{"func_idx": 5, "addr": "0xcb0"}],
            compute_encoders=[{"encoder_idx": 0, "command_buffer_idx": 0}],
        )
        raw = timeline_to_pftrace(data, group_by="pipeline")
        trace = _parse_trace(raw)
        gpu = _gpu_events(trace)
        ed = {e.name: e.value for e in gpu[0].gpu_render_stage_event.extra_data}
        assert ed["threadgroups"] == "8x1x1"
        assert ed["threads_per_threadgroup"] == "64x1x1"
        assert ed["buffers_bound"] == "2"
        assert ed["dispatch_type"] == "threads"
        assert ed["encoder"] == "0"
        assert ed["cb"] == "0"

    def test_dispatch_submission_id(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        gpu = _gpu_events(trace)
        # CB0 dispatches: fill(1), reduce(3), scatter(6) → submission_id=0
        # CB1 dispatches: matmul(10), softmax(12) → submission_id=1
        sub_ids = [(p.timestamp, p.gpu_render_stage_event.submission_id) for p in gpu]
        assert (1000, 0) in sub_ids  # fill
        assert (10000, 1) in sub_ids  # matmul

    def test_dispatch_command_buffer_handle(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        gpu = _gpu_events(trace)
        # CB0 addr = 0xcb0, CB1 addr = 0xcb1
        handles = {p.gpu_render_stage_event.command_buffer_handle for p in gpu}
        assert 0xcb0 in handles
        assert 0xcb1 in handles


# -----------------------------------------------------------------------
# Pipeline mode — barriers (TrackEvent TYPE_INSTANT)
# -----------------------------------------------------------------------


class TestPftracePipelineNoTrackEvents:
    """Pipeline mode uses only GpuRenderStageEvent — no TrackEvent tracks."""

    def test_no_track_events(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        assert len(_track_events(trace)) == 0

    def test_no_track_descriptors(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        assert len(_track_descriptors(trace)) == 0


# -----------------------------------------------------------------------
# Pipeline mode — interning
# -----------------------------------------------------------------------


class TestPftracePipelineInterning:
    """InternedData gpu_specifications match kernels."""

    def test_interned_kernel_names(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        intern_pkts = _interned_data_pkts(trace)
        assert len(intern_pkts) >= 1

        specs = list(intern_pkts[0].interned_data.gpu_specifications)
        spec_names = {s.name for s in specs}
        # 5 unique kernels + "Compute Dispatch" stage
        expected = {"fill", "reduce", "scatter", "matmul", "softmax", "Compute Dispatch"}
        assert spec_names == expected

    def test_interned_category(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        intern_pkts = _interned_data_pkts(trace)
        specs = list(intern_pkts[0].interned_data.gpu_specifications)
        for s in specs:
            assert s.category == pb.InternedGpuRenderStageSpecification.COMPUTE

    def test_hw_queue_iids_match_interning(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)

        intern_pkts = _interned_data_pkts(trace)
        specs = list(intern_pkts[0].interned_data.gpu_specifications)
        valid_iids = {s.iid for s in specs}

        gpu = _gpu_events(trace)
        for p in gpu:
            assert p.gpu_render_stage_event.hw_queue_iid in valid_iids
            assert p.gpu_render_stage_event.stage_iid in valid_iids


# -----------------------------------------------------------------------
# Pipeline mode — complex integration
# -----------------------------------------------------------------------


class TestPftracePipelineComplex:
    """Integration test with _COMPLEX_DATA in pipeline mode."""

    def test_packet_count(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        # 1 intern packet + 5 GPU event packets = 6
        assert len(trace.packet) == 6

    def test_only_gpu_and_intern_packets(self):
        """Pipeline mode has only GpuRenderStageEvent + InternedData packets."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline")
        trace = _parse_trace(raw)
        assert len(_gpu_events(trace)) == 5
        assert len(_interned_data_pkts(trace)) == 1
        assert len(_track_events(trace)) == 0
        assert len(_track_descriptors(trace)) == 0


# -----------------------------------------------------------------------
# CB mode — dispatches as TrackEvent slices
# -----------------------------------------------------------------------


class TestPftraceCBDispatches:
    """TrackEvent slices for dispatches in CB mode (no GpuRenderStageEvent)."""

    def test_no_gpu_events(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        gpu = _gpu_events(trace)
        assert len(gpu) == 0

    def test_dispatch_slice_count(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        begins = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN
        ]
        # 5 dispatches + 3 encoder spans + 2 CB overview spans = 10
        dispatch_begins = [
            b for b in begins
            if not b.track_event.name.startswith(("Encoder", "CB"))
        ]
        assert len(dispatch_begins) == 5

    def test_dispatch_name_is_kernel(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "matmul", "index": 1, "encoder_idx": 0},
            ],
            command_buffers=[{"func_idx": 5}],
            compute_encoders=[{"encoder_idx": 0, "command_buffer_idx": 0}],
        )
        raw = timeline_to_pftrace(data, group_by="cb")
        trace = _parse_trace(raw)
        begins = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN
            and p.track_event.name == "matmul"
        ]
        assert len(begins) == 1

    def test_dispatch_debug_annotations(self):
        data = _make_trace_data(
            events=[{
                "type": "dispatch", "kernel": "k", "index": 5, "encoder_idx": 0,
                "threadgroups": (4, 1, 1), "threads_per_threadgroup": (256, 1, 1),
                "buffers_bound": {0: 0x100, 1: 0x200},
                "dispatch_type": "threads",
            }],
            command_buffers=[{"func_idx": 10}],
            compute_encoders=[{"encoder_idx": 0, "command_buffer_idx": 0}],
        )
        raw = timeline_to_pftrace(data, group_by="cb")
        trace = _parse_trace(raw)
        begins = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN
            and p.track_event.name == "k"
        ]
        annotations = {}
        for da in begins[0].track_event.debug_annotations:
            if da.string_value:
                annotations[da.name] = da.string_value
            else:
                annotations[da.name] = da.int_value
        assert annotations["func_idx"] == 5
        assert annotations["threadgroups"] == "4x1x1"
        assert annotations["threads_per_threadgroup"] == "256x1x1"
        assert annotations["buffers_bound"] == 2
        assert annotations["dispatch_type"] == "threads"

    def test_dispatch_on_encoder_track(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k", "index": 1, "encoder_idx": 2},
            ],
            command_buffers=[{"func_idx": 5}, {"func_idx": 5}],
            compute_encoders=[{"encoder_idx": 2, "command_buffer_idx": 1}],
        )
        raw = timeline_to_pftrace(data, group_by="cb")
        trace = _parse_trace(raw)
        begins = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN
            and p.track_event.name == "k"
        ]
        assert begins[0].track_event.track_uuid == 3002  # _CB_ENCODER_BASE + 2


# -----------------------------------------------------------------------
# CB mode — barriers
# -----------------------------------------------------------------------


class TestPftraceCBBarriers:
    """TYPE_INSTANT for barriers in CB mode."""

    def test_barrier_count(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        instants = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_INSTANT
        ]
        assert len(instants) == 1

    def test_barrier_on_encoder_track(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        instants = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_INSTANT
        ]
        # Barrier is on encoder 0
        assert instants[0].track_event.track_uuid == 3000  # _CB_ENCODER_BASE + 0

    def test_barrier_name(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        instants = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_INSTANT
        ]
        assert instants[0].track_event.name == "barrier (buffers)"


# -----------------------------------------------------------------------
# CB mode — encoder + CB overview spans
# -----------------------------------------------------------------------


class TestPftraceCBSpans:
    """Encoder wrapper and CB overview spans in CB mode."""

    def test_encoder_span_count(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        begins = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN
            and p.track_event.name.startswith("Encoder")
        ]
        assert len(begins) == 3

    def test_cb_overview_span_count(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        begins = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN
            and p.track_event.name.startswith("CB")
        ]
        assert len(begins) == 2

    def test_encoder_span_timestamps(self):
        data = _make_trace_data(
            events=[
                {"type": "dispatch", "kernel": "k1", "index": 5, "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 10, "encoder_idx": 0},
            ],
            command_buffers=[{"func_idx": 15}],
            compute_encoders=[{"encoder_idx": 0, "command_buffer_idx": 0, "addr": "0xe0"}],
        )
        raw = timeline_to_pftrace(data, group_by="cb")
        trace = _parse_trace(raw)
        enc_begins = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN
            and p.track_event.name.startswith("Encoder")
        ]
        # The encoder wrapper span should cover func_idx 5..10 → ts 5000..11000
        assert enc_begins[0].timestamp == 5000
        # Find the matching end: last SLICE_END on same track (after dispatch ends)
        enc_uuid = enc_begins[0].track_event.track_uuid
        enc_ends = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_SLICE_END
            and p.track_event.track_uuid == enc_uuid
        ]
        # Last end should be the wrapper span's end at (10+1)*1000
        assert enc_ends[-1].timestamp == 11000

    def test_cb_overview_timestamps(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        cb_begins = [
            p for p in _track_events(trace)
            if p.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN
            and p.track_event.name.startswith("CB")
        ]
        # CB0 spans func_idx 1-6, CB1 spans 10-12
        cb0_begin = [b for b in cb_begins if "CB #0" in b.track_event.name]
        cb1_begin = [b for b in cb_begins if "CB #1" in b.track_event.name]
        assert cb0_begin[0].timestamp == 1000
        assert cb1_begin[0].timestamp == 10000


# -----------------------------------------------------------------------
# CB mode — track hierarchy
# -----------------------------------------------------------------------


class TestPftraceCBTrackHierarchy:
    """parent_uuid correctness in CB mode."""

    def test_process_tracks(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        tds = _track_descriptors(trace)

        process_tds = [t for t in tds if t.track_descriptor.HasField("process")]
        assert len(process_tds) == 2  # 2 CBs

    def test_encoder_parent_is_cb(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        tds = _track_descriptors(trace)

        # Encoder 0 and 1 belong to CB0, Encoder 2 belongs to CB1
        enc_tds = [
            t for t in tds
            if t.track_descriptor.name.startswith("Encoder")
        ]
        for enc_td in enc_tds:
            parent = enc_td.track_descriptor.parent_uuid
            # Parent should be a CB process UUID (1000+cb_idx)
            assert parent >= 1000
            assert parent < 2000

    def test_overview_parent_is_cb(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        tds = _track_descriptors(trace)

        overview_tds = [
            t for t in tds
            if t.track_descriptor.name == "CB Overview"
        ]
        for ov_td in overview_tds:
            parent = ov_td.track_descriptor.parent_uuid
            assert parent >= 1000
            assert parent < 2000

    def test_encoder_cb_assignment(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        tds = _track_descriptors(trace)

        enc_tds = {
            t.track_descriptor.name: t.track_descriptor.parent_uuid
            for t in tds
            if t.track_descriptor.name.startswith("Encoder")
        }
        # Encoder 0 and 1 -> CB0 (1000), Encoder 2 -> CB1 (1001)
        enc0 = [k for k in enc_tds if "#0" in k][0]
        enc1 = [k for k in enc_tds if "#1" in k][0]
        enc2 = [k for k in enc_tds if "#2" in k][0]
        assert enc_tds[enc0] == 1000  # CB0
        assert enc_tds[enc1] == 1000  # CB0
        assert enc_tds[enc2] == 1001  # CB1


# -----------------------------------------------------------------------
# CB mode — complex integration
# -----------------------------------------------------------------------


class TestPftraceCBComplex:
    """Integration test with _COMPLEX_DATA in CB mode."""

    def test_all_events_present(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)

        te = _track_events(trace)
        begins = [p for p in te if p.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN]
        ends = [p for p in te if p.track_event.type == pb.TrackEvent.TYPE_SLICE_END]
        instants = [p for p in te if p.track_event.type == pb.TrackEvent.TYPE_INSTANT]

        # 5 dispatches + 3 encoder spans + 2 CB overview = 10 begin/end pairs
        assert len(begins) == 10
        assert len(ends) == 10
        assert len(instants) == 1  # 1 barrier

    def test_no_interned_data(self):
        """CB mode doesn't use GpuRenderStageEvent, so no interning needed."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        intern_pkts = _interned_data_pkts(trace)
        assert len(intern_pkts) == 0

    def test_track_descriptor_count(self):
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb")
        trace = _parse_trace(raw)
        tds = _track_descriptors(trace)
        # 2 CB processes + 2 CB overviews + 3 encoder tracks = 7
        assert len(tds) == 7


# -----------------------------------------------------------------------
# GPU performance counter tracks
# -----------------------------------------------------------------------


def _gpu_counter_events(trace: pb.Trace) -> list:
    """Extract packets that have gpu_counter_event."""
    return [p for p in trace.packet if p.HasField("gpu_counter_event")]


def _counter_track_descriptors(trace: pb.Trace) -> list:
    """Extract TrackDescriptor packets that define counter tracks (have counter field)."""
    return [
        p for p in trace.packet
        if p.HasField("track_descriptor") and p.track_descriptor.HasField("counter")
    ]


def _counter_group_descriptors(trace: pb.Trace) -> list:
    """Extract TrackDescriptor packets that are counter group parents."""
    # Group descriptors have parent_uuid set but no counter field
    parent_uuids = {
        p.track_descriptor.parent_uuid
        for p in trace.packet
        if p.HasField("track_descriptor") and p.track_descriptor.parent_uuid != 0
    }
    return [
        p for p in trace.packet
        if p.HasField("track_descriptor")
        and p.track_descriptor.uuid in parent_uuids
        and not p.track_descriptor.HasField("counter")
    ]


def _counter_event_pkts(trace: pb.Trace) -> list:
    """Extract TrackEvent packets of TYPE_COUNTER."""
    return [
        p for p in trace.packet
        if p.HasField("track_event")
        and p.track_event.type == pb.TrackEvent.TYPE_COUNTER
    ]


_COUNTER_DATA = {
    "counter_names": [
        "L2 Cache Utilization", "AF Bandwidth", "L2 Cache Limiter", "AF Peak Bandwidth",
    ],
    "num_samples": 3,
    "timestamps_ns": [1000000, 2000000, 3000000],
    "samples": [
        [0.85, 1234567.0, 0.42, 0.0],  # AF Peak Bandwidth always 0 → filtered
        [0.91, 1234890.0, 0.38, 0.0],
        [0.78, 1235200.0, 0.45, 0.0],
    ],
}


class TestPftraceCounterTracks:
    """GPU performance counter tracks via grouped TrackDescriptor + TrackEvent."""

    def test_counters_none(self):
        """counters=None produces no counter track descriptors or events."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=None)
        trace = _parse_trace(raw)
        assert len(_counter_track_descriptors(trace)) == 0
        assert len(_counter_event_pkts(trace)) == 0

    def test_counter_track_descriptors(self):
        """3 counter track descriptors (AF Peak Bandwidth filtered as all-zero)."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        descs = _counter_track_descriptors(trace)
        assert len(descs) == 3

    def test_counter_spec_names(self):
        """Track names match counter_names excluding zero-only counters."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        descs = _counter_track_descriptors(trace)
        names = {d.track_descriptor.name for d in descs}
        assert names == {"L2 Cache Utilization", "AF Bandwidth", "L2 Cache Limiter"}

    def test_counter_spec_groups(self):
        """Counters are grouped under parent TrackDescriptor tracks."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        groups = _counter_group_descriptors(trace)
        group_names = {g.track_descriptor.name for g in groups}
        # L2 Cache Util/Limiter → "L2 / Texture / MMU", AF Bandwidth → "Memory Bandwidth"
        assert "L2 / Texture / MMU" in group_names
        assert "Memory Bandwidth" in group_names

    def test_counter_spec_units(self):
        """L2 Cache Utilization→percent, AF Bandwidth→bytes, L2 Cache Limiter→percent."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        descs = _counter_track_descriptors(trace)
        units = {
            d.track_descriptor.name: d.track_descriptor.counter.unit_name
            for d in descs
        }
        assert units["L2 Cache Utilization"] == "percent"
        assert units["AF Bandwidth"] == "bytes"
        assert units["L2 Cache Limiter"] == "percent"

    def test_counter_sample_packets(self):
        """3 counters x 3 timestamps = 9 TYPE_COUNTER TrackEvent packets."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        events = _counter_event_pkts(trace)
        assert len(events) == 9  # 3 counters × 3 samples

    def test_counter_sample_timestamps(self):
        """Counter event timestamps match timestamps_ns."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        events = _counter_event_pkts(trace)
        timestamps = sorted({p.timestamp for p in events})
        assert timestamps == [1000000, 2000000, 3000000]

    def test_counter_sample_values(self):
        """double_counter_value matches sample values for first timestamp."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        events = _counter_event_pkts(trace)
        # Build uuid→name map from track descriptors
        uuid_to_name = {
            d.track_descriptor.uuid: d.track_descriptor.name
            for d in _counter_track_descriptors(trace)
        }
        # Get values at first timestamp
        first = [p for p in events if p.timestamp == 1000000]
        assert len(first) == 3
        values = {
            uuid_to_name[p.track_event.track_uuid]: p.track_event.double_counter_value
            for p in first
        }
        assert abs(values["L2 Cache Utilization"] - 0.85) < 1e-5
        assert abs(values["AF Bandwidth"] - 1234567.0) < 1e-1
        assert abs(values["L2 Cache Limiter"] - 0.42) < 1e-5

    def test_counter_zero_filtering(self):
        """All-zero counter (AF Peak Bandwidth) absent from descriptors and events."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        # Check descriptors
        desc_names = {
            d.track_descriptor.name for d in _counter_track_descriptors(trace)
        }
        assert "AF Peak Bandwidth" not in desc_names
        # Check events: no track_uuid for AF Peak Bandwidth
        counter_uuids = {
            d.track_descriptor.uuid
            for d in _counter_track_descriptors(trace)
        }
        for pkt in _counter_event_pkts(trace):
            assert pkt.track_event.track_uuid in counter_uuids

    def test_counters_pipeline_mode(self):
        """Counter tracks appear alongside GpuRenderStageEvent in pipeline output."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="pipeline", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        assert len(_gpu_events(trace)) == 5  # dispatches
        assert len(_counter_track_descriptors(trace)) == 3
        assert len(_counter_event_pkts(trace)) == 9

    def test_counters_cb_mode(self):
        """Counter tracks appear alongside TrackEvent slices in CB output."""
        raw = timeline_to_pftrace(_COMPLEX_DATA, group_by="cb", counters=_COUNTER_DATA)
        trace = _parse_trace(raw)
        assert len(_track_events(trace)) > 0
        assert len(_counter_track_descriptors(trace)) == 3
        assert len(_counter_event_pkts(trace)) == 9
