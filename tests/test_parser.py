"""Tests for the XML parser with id/ref resolution."""

from __future__ import annotations

from apple_profiler._parser import iter_rows, parse_table_xml


class TestIdRefResolution:
    """Test the core id/ref deduplication resolution."""

    def test_simple_ref_resolves_to_original(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        # Row 0 defines process with id="4", row 1 uses ref="4"
        row0 = table.rows[0]
        row1 = table.rows[1]
        # Both should have the same process fmt
        assert row0[2].fmt == "Finder (751)"  # process column
        assert row1[2].fmt == "Finder (751)"

    def test_nested_ref_in_backtrace(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        # Row 1 has a backtrace with frame ref="11" (CF_IS_OBJC from row 0)
        row1_backtrace = table.rows[1][6]  # stack column
        assert row1_backtrace.tag == "backtrace"
        frame0 = row1_backtrace.children[0]
        assert frame0.tag == "frame"
        assert frame0.attrs.get("name") == "CF_IS_OBJC"

    def test_binary_ref_resolution(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        # Row 0 frame 1 (CFGetAllocator) has binary ref="12" -> CoreFoundation
        row0_backtrace = table.rows[0][6]
        frame1 = row0_backtrace.children[1]
        assert frame1.attrs.get("name") == "CFGetAllocator"
        binary = frame1.child("binary")
        assert binary is not None
        assert binary.attrs.get("name") == "CoreFoundation"

    def test_thread_ref_preserves_children(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        # Row 1 uses thread ref="2"
        row1_thread = table.rows[1][1]
        assert row1_thread.fmt == "Finder 0x644a1c3 (Finder, pid: 751)"
        tid = row1_thread.child("tid")
        assert tid is not None
        assert tid.int_value == 105161155

    def test_sentinel_elements_preserved(self, os_signpost_xml: str) -> None:
        table = parse_table_xml(os_signpost_xml)
        # Third row has a sentinel for scope
        row2 = table.rows[2]
        scope_elem = row2[4]  # scope column
        assert scope_elem.tag == "sentinel"


class TestSchemaParser:
    """Test schema column extraction."""

    def test_cpu_profile_schema(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        assert table.schema_name == "cpu-profile"
        assert len(table.columns) == 7
        mnemonics = [c.mnemonic for c in table.columns]
        assert mnemonics == [
            "time", "thread", "process", "core", "thread-state", "weight", "stack"
        ]

    def test_hangs_schema(self, potential_hangs_xml: str) -> None:
        table = parse_table_xml(potential_hangs_xml)
        assert table.schema_name == "potential-hangs"
        assert len(table.columns) == 5

    def test_signpost_schema(self, os_signpost_xml: str) -> None:
        table = parse_table_xml(os_signpost_xml)
        assert table.schema_name == "os-signpost"
        assert len(table.columns) == 10


class TestResolvedElement:
    """Test ResolvedElement helper methods."""

    def test_value_prefers_fmt(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        core = table.rows[0][3]  # core column
        assert core.value == "CPU 0 (E Core)"
        assert core.text == "0"

    def test_int_value(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        weight = table.rows[0][5]  # weight column
        assert weight.int_value == 837200

    def test_child_lookup(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        thread = table.rows[0][1]
        tid = thread.child("tid")
        assert tid is not None
        assert tid.int_value == 105161155
        assert thread.child("nonexistent") is None

    def test_children_by_tag(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        backtrace = table.rows[0][6]
        frames = backtrace.children_by_tag("frame")
        assert len(frames) == 3


class TestRowCount:
    """Test that all rows are parsed correctly."""

    def test_cpu_profile_row_count(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        assert len(table.rows) == 3

    def test_hangs_row_count(self, potential_hangs_xml: str) -> None:
        table = parse_table_xml(potential_hangs_xml)
        assert len(table.rows) == 2


class TestIterRows:
    """Test the streaming row iterator."""

    def test_iter_rows_matches_parse(self, cpu_profile_xml: str) -> None:
        table = parse_table_xml(cpu_profile_xml)
        iter_result = list(iter_rows(cpu_profile_xml))
        assert len(iter_result) == len(table.rows)
        # Check first row, first column matches
        assert iter_result[0][0].int_value == table.rows[0][0].int_value
