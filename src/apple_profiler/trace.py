"""High-level TraceFile API for working with .trace files."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

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
from apple_profiler._parser import ParsedTable, ResolvedElement, parse_table_xml, parse_toc_xml
from apple_profiler._xctrace import export_table, export_toc


class TraceFile:
    """High-level interface to an xctrace .trace file.

    Parses the table of contents on initialization, then lazily loads
    individual data tables on demand.

    Usage::

        t = TraceFile("/path/to/recording.trace")
        print(t.info)
        for sample in t.cpu_samples()[:10]:
            print(sample.backtrace[0].name, sample.weight)
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Trace file not found: {self._path}")
        self._toc_xml = export_toc(self._path)
        self._toc = parse_toc_xml(self._toc_xml)
        self._info: TraceInfo | None = None
        self._table_cache: dict[str, ParsedTable] = {}
        self._table_loader: Callable[[str, dict[str, str]], str] | None = None

    @classmethod
    def from_xml(
        cls,
        toc_xml: str,
        table_loader: Callable[[str, dict[str, str]], str] | None = None,
    ) -> TraceFile:
        """Create a TraceFile from pre-loaded XML strings (for testing).

        Args:
            toc_xml: The TOC XML string.
            table_loader: A callable that takes (schema, kwargs) and returns XML.
        """
        obj = object.__new__(cls)
        obj._path = Path("<test>")
        obj._toc_xml = toc_xml
        obj._toc = parse_toc_xml(toc_xml)
        obj._info = None
        obj._table_cache = {}
        obj._table_loader = table_loader
        return obj

    @property
    def info(self) -> TraceInfo:
        """Trace recording metadata."""
        if self._info is not None:
            return self._info

        run = self._toc.find(".//run")
        assert run is not None, "No run element found in TOC"

        info = TraceInfo()

        device = run.find(".//device")
        if device is not None:
            info.device_name = device.get("name", "")
            info.device_model = device.get("model", "")
            info.os_version = device.get("os-version", "")
            info.platform = device.get("platform", "")

        target_proc = run.find(".//target/process")
        if target_proc is not None:
            info.target_process = target_proc.get("name")
            pid_str = target_proc.get("pid")
            if pid_str is not None:
                info.target_pid = int(pid_str)

        summary = run.find(".//summary")
        if summary is not None:
            info.start_date = _text(summary, "start-date")
            info.end_date = _text(summary, "end-date")
            duration_str = _text(summary, "duration")
            if duration_str:
                info.duration_seconds = float(duration_str)
            info.instruments_version = _text(summary, "instruments-version")
            info.template_name = _text(summary, "template-name")
            info.recording_mode = _text(summary, "recording-mode")
            info.end_reason = _text(summary, "end-reason")

        self._info = info
        return info

    def tables(self) -> list[TableInfo]:
        """List all available data tables in the trace."""
        result: list[TableInfo] = []
        for table_elem in self._toc.iter("table"):
            schema = table_elem.get("schema", "")
            attrs = {k: v for k, v in table_elem.attrib.items() if k != "schema"}
            result.append(TableInfo(schema=schema, attributes=attrs))
        return result

    def has_table(self, schema: str) -> bool:
        """Check if a table with the given schema exists."""
        return any(t.schema == schema for t in self.tables())

    def _load_table(self, schema: str) -> ParsedTable:
        """Load and cache a table by schema name."""
        if schema not in self._table_cache:
            loader = self._table_loader
            if loader is not None:
                xml = loader(schema, {})
            else:
                xml = export_table(self._path, schema)
            self._table_cache[schema] = parse_table_xml(xml)
        return self._table_cache[schema]

    # ── CPU Profiling ──

    def cpu_samples(self) -> list[CpuSample]:
        """All CPU profile samples."""
        table = self._load_table("cpu-profile")
        col_index = _column_index(table)
        samples: list[CpuSample] = []

        for row in table.rows:
            time_elem = _get_col(row, col_index, "time")
            thread_elem = _get_col(row, col_index, "thread")
            process_elem = _get_col(row, col_index, "process")
            core_elem = _get_col(row, col_index, "core")
            state_elem = _get_col(row, col_index, "thread-state")
            weight_elem = _get_col(row, col_index, "weight")
            stack_elem = _get_col(row, col_index, "stack")

            process = _extract_process(process_elem)
            thread = _extract_thread(thread_elem, process)

            backtrace: list[Frame] = []
            if stack_elem is not None:
                for frame_elem in stack_elem.children_by_tag("frame"):
                    backtrace.append(_extract_frame(frame_elem))

            samples.append(CpuSample(
                time_ns=time_elem.int_value if time_elem else 0,
                thread=thread,
                process=process,
                core=core_elem.value if core_elem else "",
                state=state_elem.value if state_elem else "",
                weight=weight_elem.int_value if weight_elem else 0,
                backtrace=backtrace,
            ))

        return samples

    def top_functions(self, n: int = 20) -> list[tuple[str, int]]:
        """Top N functions by total cycle weight."""
        counter: Counter[str] = Counter()
        for sample in self.cpu_samples():
            if sample.backtrace:
                for frame in sample.backtrace:
                    counter[frame.name] += sample.weight
        return counter.most_common(n)

    # ── Hangs ──

    def hangs(self) -> list[Hang]:
        """All detected hangs/unresponsiveness intervals."""
        table = self._load_table("potential-hangs")
        col_index = _column_index(table)
        result: list[Hang] = []

        for row in table.rows:
            start_elem = _get_col(row, col_index, "start")
            duration_elem = _get_col(row, col_index, "duration")
            hang_type_elem = _get_col(row, col_index, "hang-type")
            thread_elem = _get_col(row, col_index, "thread")
            process_elem = _get_col(row, col_index, "process")

            process = _extract_process(process_elem)
            thread = _extract_thread(thread_elem, process)

            result.append(Hang(
                start_ns=start_elem.int_value if start_elem else 0,
                duration_ns=duration_elem.int_value if duration_elem else 0,
                hang_type=hang_type_elem.value if hang_type_elem else "",
                thread=thread,
                process=process,
            ))

        return result

    # ── Signposts ──

    def signpost_events(
        self,
        *,
        subsystem: str | None = None,
        category: str | None = None,
        name: str | None = None,
    ) -> list[SignpostEvent]:
        """Raw signpost events with optional filtering."""
        table = self._load_table("os-signpost")
        col_index = _column_index(table)
        result: list[SignpostEvent] = []

        for row in table.rows:
            time_elem = _get_col(row, col_index, "time")
            thread_elem = _get_col(row, col_index, "thread")
            process_elem = _get_col(row, col_index, "process")
            event_type_elem = _get_col(row, col_index, "event-type")
            scope_elem = _get_col(row, col_index, "scope")
            identifier_elem = _get_col(row, col_index, "identifier")
            name_elem = _get_col(row, col_index, "name")
            subsystem_elem = _get_col(row, col_index, "subsystem")
            category_elem = _get_col(row, col_index, "category")
            message_elem = _get_col(row, col_index, "message")

            sp_name = name_elem.value if name_elem else ""
            sp_subsystem = subsystem_elem.value if subsystem_elem else ""
            sp_category = category_elem.value if category_elem else ""

            if name is not None and sp_name != name:
                continue
            if subsystem is not None and sp_subsystem != subsystem:
                continue
            if category is not None and sp_category != category:
                continue

            process = _extract_process(process_elem)
            thread = _extract_thread(thread_elem, process)

            result.append(SignpostEvent(
                time_ns=time_elem.int_value if time_elem else 0,
                event_type=event_type_elem.value if event_type_elem else "",
                name=sp_name,
                subsystem=sp_subsystem,
                category=sp_category,
                message=message_elem.value if message_elem else "",
                scope=scope_elem.value if scope_elem and scope_elem.tag != "sentinel" else None,
                identifier=identifier_elem.int_value if identifier_elem else 0,
                thread=thread,
                process=process,
            ))

        return result

    def signpost_intervals(
        self,
        *,
        subsystem: str | None = None,
        category: str | None = None,
        name: str | None = None,
    ) -> list[SignpostInterval]:
        """Matched signpost intervals with optional filtering."""
        table = self._load_table("os-signpost-interval")
        col_index = _column_index(table)
        result: list[SignpostInterval] = []

        for row in table.rows:
            start_elem = _get_col(row, col_index, "start")
            duration_elem = _get_col(row, col_index, "duration")
            name_elem = _get_col(row, col_index, "name")
            category_elem = _get_col(row, col_index, "category")
            subsystem_elem = _get_col(row, col_index, "subsystem")
            identifier_elem = _get_col(row, col_index, "identifier")
            process_elem = _get_col(row, col_index, "process")
            start_thread_elem = _get_col(row, col_index, "start-thread")
            end_thread_elem = _get_col(row, col_index, "end-thread")
            start_msg_elem = _get_col(row, col_index, "start-message")
            end_msg_elem = _get_col(row, col_index, "end-message")

            sp_name = name_elem.value if name_elem else ""
            sp_subsystem = subsystem_elem.value if subsystem_elem else ""
            sp_category = category_elem.value if category_elem else ""

            if name is not None and sp_name != name:
                continue
            if subsystem is not None and sp_subsystem != subsystem:
                continue
            if category is not None and sp_category != category:
                continue

            process = _extract_process(process_elem)
            start_thread = _extract_thread(start_thread_elem, process)
            end_thread = _extract_thread(end_thread_elem, process)

            result.append(SignpostInterval(
                start_ns=start_elem.int_value if start_elem else 0,
                duration_ns=duration_elem.int_value if duration_elem else 0,
                name=sp_name,
                subsystem=sp_subsystem,
                category=sp_category,
                identifier=identifier_elem.int_value if identifier_elem else 0,
                process=process,
                start_thread=start_thread,
                end_thread=end_thread,
                start_message=start_msg_elem.value if start_msg_elem else "",
                end_message=end_msg_elem.value if end_msg_elem else "",
            ))

        return result

    # ── Process/Thread info ──

    def processes(self) -> list[Process]:
        """All processes listed in the TOC."""
        result: list[Process] = []
        for proc_elem in self._toc.iter("process"):
            name = proc_elem.get("name", "")
            pid_str = proc_elem.get("pid", "0")
            result.append(Process(pid=int(pid_str), name=name))
        return result

    def threads(self) -> list[Thread]:
        """All unique threads seen across CPU samples (if available)."""
        if not self.has_table("cpu-profile"):
            return []
        seen: set[int] = set()
        result: list[Thread] = []
        for sample in self.cpu_samples():
            if sample.thread.tid not in seen:
                seen.add(sample.thread.tid)
                result.append(sample.thread)
        return result


# ── Helper functions ──


def _text(parent: object, tag: str) -> str:
    """Get text content of a child element, or empty string."""
    # parent is ET.Element but we use object for the type annotation
    # because pyright doesn't like the overloaded find() method
    from xml.etree.ElementTree import Element

    assert isinstance(parent, Element)
    elem = parent.find(tag)
    if elem is not None and elem.text:
        return elem.text.strip()
    return ""


def _column_index(table: ParsedTable) -> dict[str, int]:
    """Build a mnemonic -> column index mapping."""
    return {col.mnemonic: i for i, col in enumerate(table.columns)}


def _get_col(
    row: list[ResolvedElement], col_index: dict[str, int], mnemonic: str
) -> ResolvedElement | None:
    """Get a column value from a row by mnemonic, or None if missing."""
    idx = col_index.get(mnemonic)
    if idx is None or idx >= len(row):
        return None
    elem = row[idx]
    if elem.tag == "sentinel":
        return None
    return elem


def _extract_process(elem: ResolvedElement | None) -> Process:
    """Extract a Process from a resolved element."""
    if elem is None:
        return Process(pid=0, name="Unknown")

    pid = 0
    pid_child = elem.child("pid")
    if pid_child is not None:
        pid = pid_child.int_value
    name = elem.fmt or elem.text or "Unknown"

    # fmt is like "Finder (751)" - extract just the name part
    if "(" in name:
        name = name.split("(")[0].strip()

    return Process(pid=pid, name=name)


def _extract_thread(elem: ResolvedElement | None, fallback_process: Process) -> Thread:
    """Extract a Thread from a resolved element."""
    if elem is None:
        return Thread(tid=0, name="Unknown", process=fallback_process)

    tid = 0
    tid_child = elem.child("tid")
    if tid_child is not None:
        tid = tid_child.int_value

    # Extract process from thread's children if present
    proc_child = elem.child("process")
    if proc_child is not None:
        process = _extract_process(proc_child)
    else:
        process = fallback_process

    return Thread(
        tid=tid,
        name=elem.fmt or elem.text or "Unknown",
        process=process,
    )


def _extract_frame(elem: ResolvedElement) -> Frame:
    """Extract a Frame from a resolved frame element."""
    binary_child = elem.child("binary")
    return Frame(
        name=elem.attrs.get("name", ""),
        address=elem.attrs.get("addr", ""),
        binary_name=binary_child.attrs.get("name") if binary_child else None,
        binary_path=binary_child.attrs.get("path") if binary_child else None,
        binary_uuid=binary_child.attrs.get("UUID") if binary_child else None,
    )
