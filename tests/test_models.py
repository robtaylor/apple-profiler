"""Tests for data model construction."""

from __future__ import annotations

from apple_profiler._models import (
    CpuSample,
    Frame,
    Hang,
    Process,
    SignpostEvent,
    SignpostInterval,
    TableInfo,
    Thread,
    TraceInfo,
)


class TestProcess:
    def test_equality(self) -> None:
        p1 = Process(pid=751, name="Finder")
        p2 = Process(pid=751, name="Finder")
        assert p1 == p2

    def test_inequality(self) -> None:
        p1 = Process(pid=751, name="Finder")
        p2 = Process(pid=752, name="Safari")
        assert p1 != p2

    def test_hashable(self) -> None:
        p = Process(pid=751, name="Finder")
        s = {p}
        assert p in s


class TestThread:
    def test_creation(self) -> None:
        proc = Process(pid=751, name="Finder")
        t = Thread(tid=12345, name="Main Thread", process=proc)
        assert t.tid == 12345
        assert t.process.pid == 751


class TestFrame:
    def test_minimal_frame(self) -> None:
        f = Frame(name="main", address="0x100000")
        assert f.binary_name is None

    def test_full_frame(self) -> None:
        f = Frame(
            name="objc_msgSend",
            address="0x19200e380",
            binary_name="libobjc.A.dylib",
            binary_path="/usr/lib/libobjc.A.dylib",
            binary_uuid="AB4E5CC8-D99D-314A-9FDD-6F4F6B4A2B67",
        )
        assert f.binary_name == "libobjc.A.dylib"


class TestCpuSample:
    def test_default_backtrace(self) -> None:
        proc = Process(pid=1, name="test")
        thread = Thread(tid=1, name="main", process=proc)
        s = CpuSample(
            time_ns=1000,
            thread=thread,
            process=proc,
            core="CPU 0",
            state="Running",
            weight=100,
        )
        assert s.backtrace == []


class TestTraceInfo:
    def test_defaults(self) -> None:
        info = TraceInfo()
        assert info.device_name == ""
        assert info.duration_seconds == 0.0
        assert info.target_process is None


class TestTableInfo:
    def test_with_attributes(self) -> None:
        t = TableInfo(schema="cpu-profile", attributes={"target-pid": "SINGLE"})
        assert t.schema == "cpu-profile"
        assert t.attributes["target-pid"] == "SINGLE"


class TestSignpostEvent:
    def test_optional_fields(self) -> None:
        e = SignpostEvent(
            time_ns=1000,
            event_type="Event",
            name="CacheHit",
            subsystem="com.example",
            category="caching",
            message="hit",
        )
        assert e.scope is None
        assert e.thread is None


class TestSignpostInterval:
    def test_creation(self) -> None:
        proc = Process(pid=1, name="test")
        interval = SignpostInterval(
            start_ns=1000,
            duration_ns=5000,
            name="Request",
            subsystem="com.example",
            category="net",
            identifier=42,
            process=proc,
        )
        assert interval.start_message == ""
        assert interval.end_message == ""


class TestHang:
    def test_creation(self) -> None:
        proc = Process(pid=751, name="Finder")
        thread = Thread(tid=1, name="Main Thread", process=proc)
        h = Hang(
            start_ns=500000000,
            duration_ns=2000000000,
            hang_type="Severe Hang",
            thread=thread,
            process=proc,
        )
        assert h.duration_ns == 2000000000
