"""High-level TraceFile API for working with .trace files."""

from __future__ import annotations

import re as _re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

# Schemas that contain CPU sample data with compatible columns
# (time, thread, process, core, thread-state, weight, stack).
# "cpu-profile" is used by CPU Profiler template; "time-profile" is used by
# Time Profiler and Metal System Trace templates — same column schema.
CPU_SAMPLE_SCHEMAS = ("cpu-profile", "time-profile")


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
        if run is None:
            raise ValueError("No run element found in TOC")

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

    def _find_cpu_table(self) -> str | None:
        """Find the first available CPU sample schema, or None."""
        available = {t.schema for t in self.tables()}
        for schema in CPU_SAMPLE_SCHEMAS:
            if schema in available:
                return schema
        return None

    def has_cpu_samples(self) -> bool:
        """Check if the trace contains any CPU sample data."""
        return self._find_cpu_table() is not None

    def load_table(self, schema: str) -> ParsedTable:
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

    def cpu_samples(
        self,
        *,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> list[CpuSample]:
        """CPU profile samples, optionally filtered to a time range.

        Args:
            start_ns: Include only samples at or after this timestamp (nanoseconds).
            end_ns: Include only samples at or before this timestamp (nanoseconds).
        """
        schema = self._find_cpu_table()
        if schema is None:
            available = ", ".join(t.schema for t in self.tables())
            raise ValueError(f"No CPU sample table found. Available schemas: {available}")
        table = self.load_table(schema)
        col_index = _column_index(table)
        samples: list[CpuSample] = []

        for row in table.rows:
            time_elem = _get_col(row, col_index, "time")
            time_ns = time_elem.int_value if time_elem else 0
            if start_ns is not None and time_ns < start_ns:
                continue
            if end_ns is not None and time_ns > end_ns:
                continue
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

            samples.append(
                CpuSample(
                    time_ns=time_ns,
                    thread=thread,
                    process=process,
                    core=core_elem.value if core_elem else "",
                    state=state_elem.value if state_elem else "",
                    weight=weight_elem.int_value if weight_elem else 0,
                    backtrace=backtrace,
                )
            )

        return samples

    def top_functions(
        self,
        n: int = 20,
        *,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> list[tuple[str, int]]:
        """Top N functions by total cycle weight, optionally in a time range."""
        counter: Counter[str] = Counter()
        for sample in self.cpu_samples(start_ns=start_ns, end_ns=end_ns):
            if sample.backtrace:
                for frame in sample.backtrace:
                    counter[frame.name] += sample.weight
        return counter.most_common(n)

    # ── Hangs ──

    def hangs(self) -> list[Hang]:
        """All detected hangs/unresponsiveness intervals."""
        table = self.load_table("potential-hangs")
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

            result.append(
                Hang(
                    start_ns=start_elem.int_value if start_elem else 0,
                    duration_ns=duration_elem.int_value if duration_elem else 0,
                    hang_type=hang_type_elem.value if hang_type_elem else "",
                    thread=thread,
                    process=process,
                )
            )

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
        table = self.load_table("os-signpost")
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

            result.append(
                SignpostEvent(
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
                )
            )

        return result

    def signpost_intervals(
        self,
        *,
        subsystem: str | None = None,
        category: str | None = None,
        name: str | None = None,
    ) -> list[SignpostInterval]:
        """Matched signpost intervals with optional filtering."""
        table = self.load_table("os-signpost-interval")
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

            result.append(
                SignpostInterval(
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
                )
            )

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
        if not self.has_cpu_samples():
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
    """Get text content of a child element, or empty string.

    Args:
        parent: An xml.etree.ElementTree.Element. Typed as object because
                pyright has trouble with Element.find()'s overloaded signatures.
    """
    from xml.etree.ElementTree import Element

    if not isinstance(parent, Element):
        raise TypeError(f"Expected Element, got {type(parent).__name__}")
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


# ── GPU Interval Extraction ──


@dataclass
class GpuInterval:
    """A GPU execution interval from metal-gpu-intervals."""

    start_ns: int
    duration_ns: int
    label: str  # shader/encoder label


def _clean_gpu_label(raw_label: str) -> str:
    """Simplify verbose Metal GPU interval labels.

    Raw labels look like:
      "Command Buffer 0:Compute Command 0     ( phase_test (22507) )  0x304f50ce"
      "Read Surface: 155 186 193 -> Write Surface: 6     ( WindowServer (420) )"
      "coreanimation.assembly-encoder"

    We strip:
      - trailing hex IDs (0x...)
      - process info in parentheses at the end
      - excessive whitespace
    """
    label = raw_label.strip()
    # Remove trailing hex like "  0x304f50ce"
    label = _re.sub(r"\s+0x[0-9a-fA-F]+\s*$", "", label)
    # Remove trailing "  ( process_name (PID) )" or similar
    label = _re.sub(r"\s+\(\s*\S+\s+\(\d+\)\s*\)\s*$", "", label)
    # Collapse multiple spaces
    label = _re.sub(r"\s{2,}", " ", label)
    return label.strip() or raw_label.strip()


def _extract_gpu_intervals(
    trace: TraceFile,
    target_process: str | None = None,
) -> list[GpuInterval]:
    """Extract GPU intervals from a Metal System Trace.

    Tries metal-gpu-intervals first, then metal-application-intervals,
    falls back to metal-driver-intervals.  Returns intervals sorted by
    start time.

    If *target_process* is given, only intervals whose ``process`` column
    matches (case-insensitive substring) are returned.
    """
    intervals: list[GpuInterval] = []

    # Priority order: gpu execution > app intervals > driver processing
    for schema in (
        "metal-gpu-intervals",
        "metal-application-intervals",
        "metal-driver-intervals",
    ):
        if not trace.has_table(schema):
            continue
        table = trace.load_table(schema)
        if not table.rows:
            continue

        col_index = _column_index(table)

        for row in table.rows:
            # Optionally filter by process
            if target_process:
                proc_elem = _get_col(row, col_index, "process")
                if proc_elem and target_process.lower() not in proc_elem.value.lower():
                    continue

            start_elem = _get_col(row, col_index, "start")
            dur_elem = _get_col(row, col_index, "duration")

            # Try different column names for the label (ordered by specificity)
            label_elem = None
            for mnemonic in (
                "event-label",  # metal-gpu-intervals & metal-driver-intervals
                "label",  # generic
                "channel-name",  # metal-gpu-intervals channel
                "narr",  # some schemas
                "name",  # fallback
                "shader",  # potential future schema
                "function",  # potential future schema
            ):
                label_elem = _get_col(row, col_index, mnemonic)
                if label_elem is not None and label_elem.value.strip():
                    break

            start_ns = start_elem.int_value if start_elem else 0
            dur_ns = dur_elem.int_value if dur_elem else 0
            label = _clean_gpu_label(label_elem.value) if label_elem else schema

            if start_ns > 0 or dur_ns > 0:
                intervals.append(
                    GpuInterval(
                        start_ns=start_ns,
                        duration_ns=dur_ns,
                        label=label,
                    )
                )

        if intervals:
            break  # Use the first table that has data

    intervals.sort(key=lambda iv: iv.start_ns)
    return intervals


# ── Phase Classification ──


PHASE_CPU_BOUND = "CPU_BOUND"
PHASE_GPU_BOUND = "GPU_BOUND"
PHASE_BALANCED = "BALANCED"
PHASE_PIPELINE_BUBBLE = "PIPELINE_BUBBLE"
PHASE_IDLE = "IDLE"


def _classify_phase(cpu_pct: float, gpu_pct: float) -> str:
    """Classify a time bucket's execution phase.

    Phase definitions:
        CPU_BOUND:       CPU > 40%, GPU < 20%  (CPU is the bottleneck)
        GPU_BOUND:       GPU > 40%, CPU < 20%  (GPU is the bottleneck)
        BALANCED:        CPU > 20% AND GPU > 20%  (good overlap/pipelining)
        PIPELINE_BUBBLE: Both 5-20% (sync overhead, neither fully utilized)
        IDLE:            Both < 5% (nothing happening)

    Args:
        cpu_pct: CPU utilization as a percentage (0-100).
        gpu_pct: GPU utilization as a percentage (0-100).
    """
    if cpu_pct > 40 and gpu_pct < 20:
        return PHASE_CPU_BOUND
    if gpu_pct > 40 and cpu_pct < 20:
        return PHASE_GPU_BOUND
    if cpu_pct > 20 and gpu_pct > 20:
        return PHASE_BALANCED
    if cpu_pct > 5 or gpu_pct > 5:
        return PHASE_PIPELINE_BUBBLE
    return PHASE_IDLE


# ── Correlated Timeline ──


def correlated_timeline(
    trace: TraceFile,
    *,
    num_buckets: int | None = None,
    bucket_ms: float | None = None,
    target_process: str | None = None,
) -> dict[str, Any]:
    """Build a time-aligned correlated CPU+GPU timeline from a Metal System Trace.

    Reads CPU samples (time-profile) and GPU intervals (metal-gpu-intervals)
    from the same .trace file, buckets them into aligned time windows, and
    produces a unified view showing CPU and GPU activity side by side.

    This is designed to be maximally comprehensible for LLM consumption:
    - Pre-joined (no mental timestamp correlation needed)
    - Millisecond resolution (not nanoseconds)
    - Bounded output (20-50 buckets by default)
    - Phase classification vocabulary (CPU_BOUND, GPU_BOUND, etc.)

    Args:
        trace: An opened TraceFile from a Metal System Trace recording.
        num_buckets: Number of time buckets (default: auto 20-50).
        bucket_ms: Explicit bucket width in milliseconds (overrides num_buckets).
        target_process: Filter CPU samples to this process name.

    Returns:
        Dict with keys:
          trace_info     - recording metadata
          timeline       - list of time bucket dicts
          phases         - list of detected phase dicts
          summary        - overall analysis summary
    """
    info = trace.info

    # Default to the trace's target process if not explicitly provided
    if target_process is None:
        target_process = info.target_process

    # ── Gather CPU samples ──
    cpu_samples = trace.cpu_samples() if trace.has_cpu_samples() else []
    if target_process:
        cpu_samples = [s for s in cpu_samples if target_process.lower() in s.process.name.lower()]

    # ── Gather GPU intervals ──
    gpu_intervals = _extract_gpu_intervals(trace, target_process=target_process)

    # ── Determine time range ──
    all_times: list[int] = []
    for s in cpu_samples:
        all_times.append(s.time_ns)
    for iv in gpu_intervals:
        all_times.append(iv.start_ns)
        all_times.append(iv.start_ns + iv.duration_ns)

    if not all_times:
        return {
            "trace_info": _trace_info_dict(info),
            "timeline": [],
            "phases": [],
            "summary": {
                "error": "No CPU samples or GPU intervals found in this trace.",
                "has_cpu_data": trace.has_cpu_samples(),
                "has_gpu_data": bool(gpu_intervals),
            },
        }

    t_min = min(all_times)
    t_max = max(all_times)
    total_duration_ns = t_max - t_min

    if total_duration_ns <= 0:
        total_duration_ns = 1_000_000  # 1ms floor

    # ── Compute bucket size ──
    if bucket_ms is not None:
        bucket_ns = int(bucket_ms * 1_000_000)
    elif num_buckets is not None:
        bucket_ns = total_duration_ns // max(num_buckets, 1)
    else:
        # Auto: aim for 20-50 buckets, clamped to reasonable range
        target = max(20, min(50, total_duration_ns // 10_000_000))  # ~10ms min per bucket
        bucket_ns = total_duration_ns // target

    bucket_ns = max(bucket_ns, 1_000)  # Floor at 1µs
    n_buckets = (total_duration_ns + bucket_ns - 1) // bucket_ns

    # ── Build CPU bucket data ──
    # For each bucket: total weight, top function, function weights
    cpu_bucket_weights: list[int] = [0] * n_buckets
    cpu_bucket_funcs: list[Counter[str]] = [Counter() for _ in range(n_buckets)]

    for sample in cpu_samples:
        bucket_idx = (sample.time_ns - t_min) // bucket_ns
        if 0 <= bucket_idx < n_buckets:
            cpu_bucket_weights[bucket_idx] += sample.weight
            if sample.backtrace:
                # Attribute weight to the top (leaf) frame
                cpu_bucket_funcs[bucket_idx][sample.backtrace[0].name] += sample.weight

    # Compute max possible CPU weight per bucket for % calculation
    # Weight is in time units (ns of CPU time). For % we normalize by bucket duration.
    # ── Build GPU bucket data ──
    # For each bucket: total active time, active shader labels
    gpu_bucket_active_ns: list[int] = [0] * n_buckets
    gpu_bucket_shaders: list[Counter[str]] = [Counter() for _ in range(n_buckets)]

    for iv in gpu_intervals:
        iv_start = iv.start_ns
        iv_end = iv.start_ns + iv.duration_ns

        # Spread interval across overlapping buckets
        first_bucket = max(0, (iv_start - t_min) // bucket_ns)
        last_bucket = min(n_buckets - 1, (iv_end - t_min) // bucket_ns)

        for b in range(first_bucket, last_bucket + 1):
            b_start = t_min + b * bucket_ns
            b_end = b_start + bucket_ns

            # Overlap between interval and bucket
            overlap_start = max(iv_start, b_start)
            overlap_end = min(iv_end, b_end)
            overlap_ns = max(0, overlap_end - overlap_start)

            if overlap_ns > 0:
                gpu_bucket_active_ns[b] += overlap_ns
                gpu_bucket_shaders[b][iv.label] += overlap_ns

    # ── Assemble timeline buckets ──
    timeline: list[dict[str, Any]] = []

    for b in range(n_buckets):
        b_start_ns = t_min + b * bucket_ns
        b_end_ns = min(b_start_ns + bucket_ns, t_max)
        b_duration_ns = b_end_ns - b_start_ns

        # Skip degenerate buckets (e.g., rounding remainder at end)
        if b_duration_ns < bucket_ns // 10:
            continue

        # CPU metrics
        cpu_weight = cpu_bucket_weights[b]
        cpu_funcs = cpu_bucket_funcs[b]
        top_cpu_fn = cpu_funcs.most_common(1)[0][0] if cpu_funcs else "(idle)"
        # CPU% = weight / bucket_duration * 100 (weight is in ns of CPU time)
        cpu_pct = (cpu_weight / b_duration_ns * 100) if b_duration_ns > 0 else 0.0
        cpu_pct = min(cpu_pct, 100.0)  # Can exceed 100% with multiple cores

        # GPU metrics
        gpu_active = gpu_bucket_active_ns[b]
        gpu_shaders = gpu_bucket_shaders[b]
        top_gpu_shader = gpu_shaders.most_common(1)[0][0] if gpu_shaders else "(idle)"
        gpu_pct = (gpu_active / b_duration_ns * 100) if b_duration_ns > 0 else 0.0
        gpu_pct = min(gpu_pct, 100.0)

        phase = _classify_phase(cpu_pct, gpu_pct)

        bucket_dict: dict[str, Any] = {
            "start_ms": round(b_start_ns / 1_000_000, 2),
            "end_ms": round(b_end_ns / 1_000_000, 2),
            "cpu_top_function": top_cpu_fn,
            "cpu_pct": round(cpu_pct, 1),
            "gpu_top_shader": top_gpu_shader,
            "gpu_pct": round(gpu_pct, 1),
            "phase": phase,
        }

        # Add secondary CPU functions if multiple are active
        if len(cpu_funcs) > 1:
            secondary = cpu_funcs.most_common(4)[1:]  # top 3 secondary
            if secondary:
                bucket_dict["cpu_other_functions"] = [
                    {"name": name, "weight": w} for name, w in secondary
                ]

        # Add secondary GPU shaders if multiple are active
        if len(gpu_shaders) > 1:
            secondary = gpu_shaders.most_common(4)[1:]
            if secondary:
                bucket_dict["gpu_other_shaders"] = [
                    {"name": name, "active_ns": ns} for name, ns in secondary
                ]

        timeline.append(bucket_dict)

    # ── Detect phases (merge consecutive same-phase buckets) ──
    phases: list[dict[str, Any]] = []
    if timeline:
        current_phase = timeline[0]["phase"]
        phase_start_ms = timeline[0]["start_ms"]
        phase_cpu_total = 0.0
        phase_gpu_total = 0.0
        phase_count = 0
        phase_cpu_funcs: Counter[str] = Counter()
        phase_gpu_shaders: Counter[str] = Counter()

        for bucket in timeline:
            if bucket["phase"] != current_phase:
                # Emit completed phase
                phases.append(
                    {
                        "start_ms": phase_start_ms,
                        "end_ms": bucket["start_ms"],
                        "duration_ms": round(bucket["start_ms"] - phase_start_ms, 2),
                        "phase": current_phase,
                        "avg_cpu_pct": round(phase_cpu_total / max(phase_count, 1), 1),
                        "avg_gpu_pct": round(phase_gpu_total / max(phase_count, 1), 1),
                        "top_cpu_function": phase_cpu_funcs.most_common(1)[0][0]
                        if phase_cpu_funcs
                        else "(idle)",
                        "top_gpu_shader": phase_gpu_shaders.most_common(1)[0][0]
                        if phase_gpu_shaders
                        else "(idle)",
                    }
                )
                current_phase = bucket["phase"]
                phase_start_ms = bucket["start_ms"]
                phase_cpu_total = 0.0
                phase_gpu_total = 0.0
                phase_count = 0
                phase_cpu_funcs = Counter()
                phase_gpu_shaders = Counter()

            phase_cpu_total += bucket["cpu_pct"]
            phase_gpu_total += bucket["gpu_pct"]
            phase_count += 1
            phase_cpu_funcs[bucket["cpu_top_function"]] += 1
            phase_gpu_shaders[bucket["gpu_top_shader"]] += 1

        # Emit final phase
        last_end = timeline[-1]["end_ms"]
        phases.append(
            {
                "start_ms": phase_start_ms,
                "end_ms": last_end,
                "duration_ms": round(last_end - phase_start_ms, 2),
                "phase": current_phase,
                "avg_cpu_pct": round(phase_cpu_total / max(phase_count, 1), 1),
                "avg_gpu_pct": round(phase_gpu_total / max(phase_count, 1), 1),
                "top_cpu_function": phase_cpu_funcs.most_common(1)[0][0]
                if phase_cpu_funcs
                else "(idle)",
                "top_gpu_shader": phase_gpu_shaders.most_common(1)[0][0]
                if phase_gpu_shaders
                else "(idle)",
            }
        )

    # ── Summary statistics ──
    total_ms = total_duration_ns / 1_000_000
    phase_time: Counter[str] = Counter()
    for p in phases:
        phase_time[p["phase"]] += p["duration_ms"]

    # Determine overall bottleneck
    bottleneck = "UNKNOWN"
    if phase_time[PHASE_GPU_BOUND] > phase_time[PHASE_CPU_BOUND]:
        bottleneck = "GPU_BOUND"
    elif phase_time[PHASE_CPU_BOUND] > phase_time[PHASE_GPU_BOUND]:
        bottleneck = "CPU_BOUND"
    elif phase_time[PHASE_BALANCED] > total_ms * 0.5:
        bottleneck = "BALANCED"

    # Overall top functions/shaders
    all_cpu_funcs: Counter[str] = Counter()
    for sample in cpu_samples:
        if sample.backtrace:
            all_cpu_funcs[sample.backtrace[0].name] += sample.weight

    all_gpu_shaders: Counter[str] = Counter()
    for iv in gpu_intervals:
        all_gpu_shaders[iv.label] += iv.duration_ns

    summary: dict[str, Any] = {
        "total_duration_ms": round(total_ms, 2),
        "bucket_width_ms": round(bucket_ns / 1_000_000, 2),
        "num_buckets": n_buckets,
        "num_cpu_samples": len(cpu_samples),
        "num_gpu_intervals": len(gpu_intervals),
        "bottleneck": bottleneck,
        "phase_breakdown": {
            phase: {
                "duration_ms": round(phase_time.get(phase, 0), 2),
                "pct": round(phase_time.get(phase, 0) / total_ms * 100, 1) if total_ms > 0 else 0,
            }
            for phase in (
                PHASE_CPU_BOUND,
                PHASE_GPU_BOUND,
                PHASE_BALANCED,
                PHASE_PIPELINE_BUBBLE,
                PHASE_IDLE,
            )
        },
        "top_cpu_functions": [
            {"name": name, "total_weight": w} for name, w in all_cpu_funcs.most_common(10)
        ],
        "top_gpu_shaders": [
            {"name": name, "total_ns": ns, "total_ms": round(ns / 1_000_000, 2)}
            for name, ns in all_gpu_shaders.most_common(10)
        ],
    }

    return {
        "trace_info": _trace_info_dict(info),
        "timeline": timeline,
        "phases": phases,
        "summary": summary,
    }


def _trace_info_dict(info: TraceInfo) -> dict[str, Any]:
    """Serialize TraceInfo to a dict."""
    return {
        "template_name": info.template_name,
        "duration_seconds": info.duration_seconds,
        "device_name": info.device_name,
        "target_process": info.target_process,
    }
