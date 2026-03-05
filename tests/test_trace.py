"""Tests for the high-level TraceFile API using fixture XML."""

from __future__ import annotations

from pathlib import Path

import pytest

from apple_profiler.trace import TraceFile


@pytest.fixture
def trace_file(
    toc_xml: str,
    cpu_profile_xml: str,
    potential_hangs_xml: str,
    os_signpost_xml: str,
    os_signpost_interval_xml: str,
) -> TraceFile:
    """Create a TraceFile backed by fixture XML."""
    table_map = {
        "cpu-profile": cpu_profile_xml,
        "potential-hangs": potential_hangs_xml,
        "os-signpost": os_signpost_xml,
        "os-signpost-interval": os_signpost_interval_xml,
    }

    def loader(schema: str, kwargs: dict[str, str]) -> str:
        xml = table_map.get(schema)
        if xml is None:
            raise ValueError(f"No fixture for schema: {schema}")
        return xml

    return TraceFile.from_xml(toc_xml, table_loader=loader)


class TestTraceInfo:
    def test_device_info(self, trace_file: TraceFile) -> None:
        info = trace_file.info
        assert info.device_name == "Robert's MacBook Pro"
        assert info.device_model == "MacBook Pro"
        assert info.platform == "macOS"
        assert info.os_version == "15.7.3 (24G419)"

    def test_timing(self, trace_file: TraceFile) -> None:
        info = trace_file.info
        assert info.duration_seconds == pytest.approx(3.398757)
        assert info.start_date == "2026-03-05T17:27:38.879Z"

    def test_recording_info(self, trace_file: TraceFile) -> None:
        info = trace_file.info
        assert info.instruments_version == "26.0 (17C529)"
        assert info.template_name == "Blank"
        assert info.recording_mode == "Deferred"
        assert info.end_reason == "Time limit reached"

    def test_target_process(self, trace_file: TraceFile) -> None:
        info = trace_file.info
        assert info.target_process == "Finder"
        assert info.target_pid == 751


class TestTables:
    def test_list_tables(self, trace_file: TraceFile) -> None:
        tables = trace_file.tables()
        schemas = [t.schema for t in tables]
        assert "cpu-profile" in schemas
        assert "potential-hangs" in schemas
        assert "os-signpost" in schemas

    def test_has_table(self, trace_file: TraceFile) -> None:
        assert trace_file.has_table("cpu-profile")
        assert not trace_file.has_table("nonexistent-table")

    def test_table_attributes(self, trace_file: TraceFile) -> None:
        tables = trace_file.tables()
        cpu = next(t for t in tables if t.schema == "cpu-profile")
        assert cpu.attributes["target-pid"] == "SINGLE"


class TestCpuSamples:
    def test_sample_count(self, trace_file: TraceFile) -> None:
        samples = trace_file.cpu_samples()
        assert len(samples) == 3

    def test_first_sample(self, trace_file: TraceFile) -> None:
        sample = trace_file.cpu_samples()[0]
        assert sample.time_ns == 575214000
        assert sample.weight == 837200
        assert sample.core == "CPU 0 (E Core)"
        assert sample.state == "Running"

    def test_process_extraction(self, trace_file: TraceFile) -> None:
        sample = trace_file.cpu_samples()[0]
        assert sample.process.pid == 751
        assert sample.process.name == "Finder"

    def test_thread_extraction(self, trace_file: TraceFile) -> None:
        sample = trace_file.cpu_samples()[0]
        assert sample.thread.tid == 105161155
        assert "Finder" in sample.thread.name

    def test_backtrace(self, trace_file: TraceFile) -> None:
        sample = trace_file.cpu_samples()[0]
        assert len(sample.backtrace) == 3
        assert sample.backtrace[0].name == "CF_IS_OBJC"
        assert sample.backtrace[0].address == "0x19286220d"
        assert sample.backtrace[0].binary_name == "CoreFoundation"

    def test_ref_resolved_backtrace(self, trace_file: TraceFile) -> None:
        # Row 1 backtrace has frame ref="11" -> CF_IS_OBJC
        sample = trace_file.cpu_samples()[1]
        assert sample.backtrace[0].name == "CF_IS_OBJC"
        assert sample.backtrace[0].binary_name == "CoreFoundation"

    def test_different_thread(self, trace_file: TraceFile) -> None:
        sample = trace_file.cpu_samples()[2]
        assert sample.thread.tid == 105161156
        assert sample.core == "CPU 4 (P Core)"
        assert sample.state == "Blocked"


class TestTopFunctions:
    def test_top_functions(self, trace_file: TraceFile) -> None:
        top = trace_file.top_functions(n=5)
        assert len(top) > 0
        # CF_IS_OBJC appears in rows 0 and 1, with weights 837200 and 123450
        names = [name for name, _ in top]
        assert "CF_IS_OBJC" in names


class TestHangs:
    def test_hang_count(self, trace_file: TraceFile) -> None:
        hangs = trace_file.hangs()
        assert len(hangs) == 2

    def test_severe_hang(self, trace_file: TraceFile) -> None:
        hang = trace_file.hangs()[0]
        assert hang.start_ns == 597113458
        assert hang.duration_ns == 2075111540
        assert hang.hang_type == "Severe Hang"
        assert hang.process.pid == 751

    def test_regular_hang(self, trace_file: TraceFile) -> None:
        hang = trace_file.hangs()[1]
        assert hang.hang_type == "Hang"
        assert hang.process.pid == 500
        assert hang.process.name == "Safari"


class TestSignpostEvents:
    def test_event_count(self, trace_file: TraceFile) -> None:
        events = trace_file.signpost_events()
        assert len(events) == 3

    def test_begin_event(self, trace_file: TraceFile) -> None:
        event = trace_file.signpost_events()[0]
        assert event.event_type == "Begin"
        assert event.name == "NetworkRequest"
        assert event.subsystem == "com.example.app"
        assert event.category == "networking"
        assert event.time_ns == 573703666

    def test_end_event(self, trace_file: TraceFile) -> None:
        event = trace_file.signpost_events()[1]
        assert event.event_type == "End"
        assert event.name == "NetworkRequest"  # resolved via ref

    def test_standalone_event(self, trace_file: TraceFile) -> None:
        event = trace_file.signpost_events()[2]
        assert event.event_type == "Event"
        assert event.name == "CacheHit"
        assert event.scope is None  # was sentinel

    def test_filter_by_name(self, trace_file: TraceFile) -> None:
        events = trace_file.signpost_events(name="CacheHit")
        assert len(events) == 1
        assert events[0].name == "CacheHit"

    def test_filter_by_subsystem(self, trace_file: TraceFile) -> None:
        events = trace_file.signpost_events(subsystem="com.example.app")
        assert len(events) == 3

    def test_filter_by_category(self, trace_file: TraceFile) -> None:
        events = trace_file.signpost_events(category="caching")
        assert len(events) == 1

    def test_filter_no_match(self, trace_file: TraceFile) -> None:
        events = trace_file.signpost_events(subsystem="nonexistent")
        assert len(events) == 0

    def test_message_extraction(self, trace_file: TraceFile) -> None:
        event = trace_file.signpost_events()[0]
        assert "url=https://api.example.com/data" in event.message


class TestSignpostIntervals:
    def test_interval_count(self, trace_file: TraceFile) -> None:
        intervals = trace_file.signpost_intervals()
        assert len(intervals) == 2

    def test_first_interval(self, trace_file: TraceFile) -> None:
        interval = trace_file.signpost_intervals()[0]
        assert interval.start_ns == 573703666
        assert interval.duration_ns == 100000000
        assert interval.name == "NetworkRequest"
        assert interval.subsystem == "com.example.app"
        assert interval.category == "networking"

    def test_interval_messages(self, trace_file: TraceFile) -> None:
        interval = trace_file.signpost_intervals()[0]
        assert "url=" in interval.start_message
        assert "status=200" in interval.end_message

    def test_filter_intervals(self, trace_file: TraceFile) -> None:
        intervals = trace_file.signpost_intervals(name="DatabaseQuery")
        assert len(intervals) == 1
        assert intervals[0].category == "database"

    def test_interval_process(self, trace_file: TraceFile) -> None:
        interval = trace_file.signpost_intervals()[0]
        assert interval.process is not None
        assert interval.process.pid == 394


class TestProcesses:
    def test_processes_from_toc(self, trace_file: TraceFile) -> None:
        procs = trace_file.processes()
        assert len(procs) >= 2
        names = [p.name for p in procs]
        assert "Finder" in names
        assert "kernel.release.t6041" in names


class TestIntegration:
    """Integration test requiring actual xctrace binary."""

    @pytest.mark.integration
    def test_real_trace(self) -> None:
        trace_path = Path("/tmp/claude/finder.trace")
        if not trace_path.exists():
            pytest.skip("No trace file at /tmp/claude/finder.trace")

        t = TraceFile(trace_path)
        info = t.info
        assert info.device_name != ""
        assert info.duration_seconds > 0

        tables = t.tables()
        assert len(tables) > 0

        if t.has_table("cpu-profile"):
            samples = t.cpu_samples()
            assert len(samples) > 0
            assert samples[0].backtrace  # should have frames
