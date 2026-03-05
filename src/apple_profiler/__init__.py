"""apple-profiler: Python wrapper for xctrace export XML parsing."""

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
from apple_profiler._parser import ParsedTable, ResolvedElement, SchemaColumn, parse_table_xml
from apple_profiler._xctrace import XctraceError
from apple_profiler.trace import TraceFile

__all__ = [
    "CpuSample",
    "Frame",
    "Hang",
    "ParsedTable",
    "Process",
    "ResolvedElement",
    "SchemaColumn",
    "SignpostEvent",
    "SignpostInterval",
    "TableInfo",
    "Thread",
    "TraceFile",
    "TraceInfo",
    "XctraceError",
    "parse_table_xml",
]
