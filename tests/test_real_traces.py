"""Tests using real xctrace XML exports as fixtures.

These test against actual XML output from xctrace export, verifying that
the parser and TraceFile API handle real-world data correctly including:
- Complex id/ref chains across many rows
- Real backtrace depths (20+ frames)
- All column types present in production schemas
- Edge cases in real data (sentinel elements, missing fields)
"""

from __future__ import annotations

from apple_profiler._parser import iter_rows, parse_table_xml
from apple_profiler.trace import TraceFile

# ── Parser tests with real XML ──


class TestRealCpuProfileParser:
    """Test parser against real Finder CPU profile export."""

    def test_row_count(self, real_finder_cpu_profile_xml: str) -> None:
        table = parse_table_xml(real_finder_cpu_profile_xml)
        assert table.schema_name == "cpu-profile"
        assert len(table.rows) == 91

    def test_all_rows_have_correct_column_count(self, real_finder_cpu_profile_xml: str) -> None:
        table = parse_table_xml(real_finder_cpu_profile_xml)
        for i, row in enumerate(table.rows):
            assert len(row) == 7, f"Row {i} has {len(row)} columns, expected 7"

    def test_first_row_values(self, real_finder_cpu_profile_xml: str) -> None:
        table = parse_table_xml(real_finder_cpu_profile_xml)
        row = table.rows[0]
        assert row[0].int_value == 575214000  # time
        assert row[0].fmt == "00:00.575.214"
        assert row[3].value == "CPU 0 (E Core)"  # core
        assert row[4].value == "Running"  # thread-state
        assert row[5].int_value == 837200  # weight

    def test_ref_resolution_across_rows(self, real_finder_cpu_profile_xml: str) -> None:
        """Rows after the first should resolve refs to the same values."""
        table = parse_table_xml(real_finder_cpu_profile_xml)
        # All rows should reference the Finder process
        for i, row in enumerate(table.rows):
            process = row[2]  # process column
            assert process.fmt is not None, f"Row {i} has no process fmt"
            assert "751" in process.fmt, f"Row {i} process doesn't contain pid 751: {process.fmt}"

    def test_backtrace_has_real_depth(self, real_finder_cpu_profile_xml: str) -> None:
        """Real backtraces should have many frames."""
        table = parse_table_xml(real_finder_cpu_profile_xml)
        backtrace = table.rows[0][6]
        frames = backtrace.children_by_tag("frame")
        assert len(frames) >= 10, f"Expected deep backtrace, got {len(frames)} frames"

    def test_binary_refs_resolve_correctly(self, real_finder_cpu_profile_xml: str) -> None:
        """Multiple frames referencing the same binary should all resolve."""
        table = parse_table_xml(real_finder_cpu_profile_xml)
        backtrace = table.rows[0][6]
        frames = backtrace.children_by_tag("frame")
        binaries_seen: set[str] = set()
        for frame in frames:
            binary = frame.child("binary")
            if binary is not None:
                name = binary.attrs.get("name", "")
                assert name != "", f"Binary has empty name: {binary}"
                binaries_seen.add(name)
        assert "CoreFoundation" in binaries_seen

    def test_iter_rows_matches_full_parse(self, real_finder_cpu_profile_xml: str) -> None:
        table = parse_table_xml(real_finder_cpu_profile_xml)
        streamed = list(iter_rows(real_finder_cpu_profile_xml))
        assert len(streamed) == len(table.rows)


class TestRealHangsParser:
    def test_parse_single_hang(self, real_hangs_xml: str) -> None:
        table = parse_table_xml(real_hangs_xml)
        assert table.schema_name == "potential-hangs"
        assert len(table.rows) == 1
        row = table.rows[0]
        assert row[0].int_value == 597113458  # start
        assert row[1].int_value == 2075111540  # duration (~2s)
        assert row[2].value == "Severe Hang"


class TestRealSignpostParser:
    def test_signpost_events_row_count(self, real_signpost_events_xml: str) -> None:
        table = parse_table_xml(real_signpost_events_xml)
        assert table.schema_name == "os-signpost"
        assert len(table.rows) == 50

    def test_signpost_events_schema_columns(self, real_signpost_events_xml: str) -> None:
        table = parse_table_xml(real_signpost_events_xml)
        mnemonics = [c.mnemonic for c in table.columns]
        # Real schema has 13 columns including format-string and emit-location
        assert "time" in mnemonics
        assert "event-type" in mnemonics
        assert "name" in mnemonics
        assert "subsystem" in mnemonics
        assert "category" in mnemonics
        assert "message" in mnemonics
        assert "format-string" in mnemonics
        assert "emit-location" in mnemonics

    def test_signpost_intervals_row_count(self, real_signpost_intervals_xml: str) -> None:
        table = parse_table_xml(real_signpost_intervals_xml)
        assert table.schema_name == "os-signpost-interval"
        assert len(table.rows) == 50

    def test_signpost_intervals_schema_columns(self, real_signpost_intervals_xml: str) -> None:
        table = parse_table_xml(real_signpost_intervals_xml)
        mnemonics = [c.mnemonic for c in table.columns]
        # Real schema has 18 columns
        assert len(mnemonics) == 18
        assert "start" in mnemonics
        assert "duration" in mnemonics
        assert "start-thread" in mnemonics
        assert "end-thread" in mnemonics
        assert "signature" in mnemonics  # extra column not in hand-crafted fixtures

    def test_sentinel_handling_in_signpost_events(self, real_signpost_events_xml: str) -> None:
        """Real signpost data has sentinel elements for missing backtrace/message."""
        table = parse_table_xml(real_signpost_events_xml)
        sentinel_count = 0
        for row in table.rows:
            for elem in row:
                if elem.tag == "sentinel":
                    sentinel_count += 1
        assert sentinel_count > 0, "Expected sentinel elements in real signpost data"


# ── TraceFile tests with real XML ──


class TestRealTraceFileFinder:
    """Test TraceFile API against real Finder trace data."""

    def _make_trace(self, real_finder_toc_xml: str, real_finder_cpu_profile_xml: str) -> TraceFile:
        table_map = {"cpu-profile": real_finder_cpu_profile_xml}

        def loader(schema: str, kwargs: dict[str, str]) -> str:
            xml = table_map.get(schema)
            if xml is None:
                raise ValueError(f"No fixture for schema: {schema}")
            return xml

        return TraceFile.from_xml(real_finder_toc_xml, table_loader=loader)

    def test_info(self, real_finder_toc_xml: str, real_finder_cpu_profile_xml: str) -> None:
        t = self._make_trace(real_finder_toc_xml, real_finder_cpu_profile_xml)
        info = t.info
        assert "MacBook Pro" in info.device_name
        assert info.device_model == "MacBook Pro"
        assert info.platform == "macOS"
        assert "15.7" in info.os_version
        assert info.target_process == "Finder"
        assert info.target_pid == 751
        assert info.duration_seconds > 3.0

    def test_tables(self, real_finder_toc_xml: str, real_finder_cpu_profile_xml: str) -> None:
        t = self._make_trace(real_finder_toc_xml, real_finder_cpu_profile_xml)
        tables = t.tables()
        schemas = [tb.schema for tb in tables]
        assert "cpu-profile" in schemas
        assert "kdebug" in schemas  # real traces have kdebug tables

    def test_cpu_samples(self, real_finder_toc_xml: str, real_finder_cpu_profile_xml: str) -> None:
        t = self._make_trace(real_finder_toc_xml, real_finder_cpu_profile_xml)
        samples = t.cpu_samples()
        assert len(samples) == 91

        # Check first sample
        s0 = samples[0]
        assert s0.time_ns == 575214000
        assert s0.process.pid == 751
        assert s0.process.name == "Finder"
        assert s0.thread.tid == 105161155
        assert "E Core" in s0.core
        assert s0.state == "Running"
        assert s0.weight == 837200

    def test_cpu_sample_backtraces(
        self, real_finder_toc_xml: str, real_finder_cpu_profile_xml: str
    ) -> None:
        t = self._make_trace(real_finder_toc_xml, real_finder_cpu_profile_xml)
        samples = t.cpu_samples()
        # Every sample should have a backtrace
        for i, s in enumerate(samples):
            assert len(s.backtrace) > 0, f"Sample {i} has no backtrace"
            # Every frame should have a name and address
            for j, f in enumerate(s.backtrace):
                assert f.name != "", f"Sample {i} frame {j} has empty name"
                assert f.address != "", f"Sample {i} frame {j} has empty address"

    def test_cpu_sample_binary_info(
        self, real_finder_toc_xml: str, real_finder_cpu_profile_xml: str
    ) -> None:
        t = self._make_trace(real_finder_toc_xml, real_finder_cpu_profile_xml)
        s0 = t.cpu_samples()[0]
        # First frame should have CoreFoundation binary
        assert s0.backtrace[0].name == "CF_IS_OBJC"
        assert s0.backtrace[0].binary_name == "CoreFoundation"
        assert s0.backtrace[0].binary_uuid is not None

    def test_top_functions(
        self, real_finder_toc_xml: str, real_finder_cpu_profile_xml: str
    ) -> None:
        t = self._make_trace(real_finder_toc_xml, real_finder_cpu_profile_xml)
        top = t.top_functions(10)
        assert len(top) == 10
        # All weights should be positive
        for name, weight in top:
            assert weight > 0
            assert name != ""

    def test_processes_from_toc(
        self, real_finder_toc_xml: str, real_finder_cpu_profile_xml: str
    ) -> None:
        t = self._make_trace(real_finder_toc_xml, real_finder_cpu_profile_xml)
        procs = t.processes()
        pids = {p.pid for p in procs}
        assert 751 in pids  # Finder
        assert 0 in pids  # kernel


class TestRealTraceFileHangs:
    def _make_trace(self, real_hangs_toc_xml: str, real_hangs_xml: str) -> TraceFile:
        table_map = {"potential-hangs": real_hangs_xml}

        def loader(schema: str, kwargs: dict[str, str]) -> str:
            xml = table_map.get(schema)
            if xml is None:
                raise ValueError(f"No fixture for schema: {schema}")
            return xml

        return TraceFile.from_xml(real_hangs_toc_xml, table_loader=loader)

    def test_hangs(self, real_hangs_toc_xml: str, real_hangs_xml: str) -> None:
        t = self._make_trace(real_hangs_toc_xml, real_hangs_xml)
        hangs = t.hangs()
        assert len(hangs) == 1
        h = hangs[0]
        assert h.hang_type == "Severe Hang"
        assert h.duration_ns > 2_000_000_000  # > 2 seconds
        assert h.start_ns > 0


class TestRealTraceFileSignposts:
    def _make_trace(
        self,
        real_signpost_toc_xml: str,
        real_signpost_events_xml: str,
        real_signpost_intervals_xml: str,
    ) -> TraceFile:
        table_map = {
            "os-signpost": real_signpost_events_xml,
            "os-signpost-interval": real_signpost_intervals_xml,
        }

        def loader(schema: str, kwargs: dict[str, str]) -> str:
            xml = table_map.get(schema)
            if xml is None:
                raise ValueError(f"No fixture for schema: {schema}")
            return xml

        return TraceFile.from_xml(real_signpost_toc_xml, table_loader=loader)

    def test_signpost_events(
        self,
        real_signpost_toc_xml: str,
        real_signpost_events_xml: str,
        real_signpost_intervals_xml: str,
    ) -> None:
        t = self._make_trace(
            real_signpost_toc_xml, real_signpost_events_xml, real_signpost_intervals_xml
        )
        events = t.signpost_events()
        assert len(events) == 50

        # First event should be a Begin
        e0 = events[0]
        assert e0.event_type == "Begin"
        assert e0.name == "FrameLifetime"
        assert e0.subsystem == "com.apple.SkyLight"
        assert e0.category == "tracing.stalls"
        assert e0.time_ns == 573703666

    def test_signpost_events_filter(
        self,
        real_signpost_toc_xml: str,
        real_signpost_events_xml: str,
        real_signpost_intervals_xml: str,
    ) -> None:
        t = self._make_trace(
            real_signpost_toc_xml, real_signpost_events_xml, real_signpost_intervals_xml
        )
        # Filter by subsystem
        skylight = t.signpost_events(subsystem="com.apple.SkyLight")
        assert len(skylight) > 0
        assert all(e.subsystem == "com.apple.SkyLight" for e in skylight)

        # Filter by name
        frame_events = t.signpost_events(name="FrameLifetime")
        assert len(frame_events) > 0
        assert all(e.name == "FrameLifetime" for e in frame_events)

    def test_signpost_intervals(
        self,
        real_signpost_toc_xml: str,
        real_signpost_events_xml: str,
        real_signpost_intervals_xml: str,
    ) -> None:
        t = self._make_trace(
            real_signpost_toc_xml, real_signpost_events_xml, real_signpost_intervals_xml
        )
        intervals = t.signpost_intervals()
        assert len(intervals) == 50

        # First interval
        i0 = intervals[0]
        assert i0.name == "FrameLifetime"
        assert i0.start_ns == 573703666
        assert i0.duration_ns > 0
        assert i0.subsystem == "com.apple.SkyLight"

    def test_signpost_intervals_filter(
        self,
        real_signpost_toc_xml: str,
        real_signpost_events_xml: str,
        real_signpost_intervals_xml: str,
    ) -> None:
        t = self._make_trace(
            real_signpost_toc_xml, real_signpost_events_xml, real_signpost_intervals_xml
        )
        composite = t.signpost_intervals(name="CompositeLoop")
        assert len(composite) > 0
        assert all(i.name == "CompositeLoop" for i in composite)

    def test_signpost_process_extraction(
        self,
        real_signpost_toc_xml: str,
        real_signpost_events_xml: str,
        real_signpost_intervals_xml: str,
    ) -> None:
        t = self._make_trace(
            real_signpost_toc_xml, real_signpost_events_xml, real_signpost_intervals_xml
        )
        events = t.signpost_events()
        # WindowServer should be the process for SkyLight signposts
        skylight = [e for e in events if e.subsystem == "com.apple.SkyLight"]
        assert len(skylight) > 0
        assert skylight[0].process is not None
        assert skylight[0].process.name == "WindowServer"
        assert skylight[0].process.pid == 394
