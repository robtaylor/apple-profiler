"""Integration tests for Metal System Trace analysis.

Compiles a minimal Metal compute program, records it with xctrace using
the Metal System Trace template, then verifies that profiler_query_table
can extract meaningful Metal table data.

Requires:
    - macOS with Metal GPU support
    - xctrace (Xcode command line tools)
    - Swift compiler (swiftc)

Run:
    uv run pytest tests/test_metal_trace.py -v -m metal_integration
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from apple_profiler.trace import TraceFile

# Minimal Swift program that runs a Metal compute shader.
# Uses inline shader compilation so no .metal file is needed.
METAL_PROGRAM_SOURCE = (
    'import Metal\n'
    'import Foundation\n'
    '\n'
    'guard let device = MTLCreateSystemDefaultDevice() else {\n'
    '    fputs("No Metal device available\\n", stderr)\n'
    '    exit(1)\n'
    '}\n'
    '\n'
    'let shaderSource = "kernel void fill(device float *buffer [[buffer(0)]],'
    ' uint id [[thread_position_in_grid]]) { buffer[id] = float(id) * 0.5; }"\n'
    '\n'
    'let library: MTLLibrary\n'
    'do {\n'
    '    library = try device.makeLibrary(source: shaderSource, options: nil)\n'
    '} catch {\n'
    '    fputs("Failed to compile shader: \\(error)\\n", stderr)\n'
    '    exit(1)\n'
    '}\n'
    '\n'
    'guard let function = library.makeFunction(name: "fill") else {\n'
    '    fputs("Function not found\\n", stderr)\n'
    '    exit(1)\n'
    '}\n'
    '\n'
    'let pipeline: MTLComputePipelineState\n'
    'do {\n'
    '    pipeline = try device.makeComputePipelineState(function: function)\n'
    '} catch {\n'
    '    fputs("Failed to create pipeline: \\(error)\\n", stderr)\n'
    '    exit(1)\n'
    '}\n'
    '\n'
    'let bufferCount = 256 * 1024\n'
    'guard let metalBuffer = device.makeBuffer(\n'
    '    length: bufferCount * MemoryLayout<Float>.size,\n'
    '    options: .storageModeShared\n'
    ') else {\n'
    '    fputs("Failed to create buffer\\n", stderr)\n'
    '    exit(1)\n'
    '}\n'
    '\n'
    'guard let commandQueue = device.makeCommandQueue() else {\n'
    '    fputs("Failed to create command queue\\n", stderr)\n'
    '    exit(1)\n'
    '}\n'
    '\n'
    'for _ in 0..<200 {\n'
    '    guard let commandBuffer = commandQueue.makeCommandBuffer(),\n'
    '          let encoder = commandBuffer.makeComputeCommandEncoder() else {\n'
    '        continue\n'
    '    }\n'
    '    encoder.setComputePipelineState(pipeline)\n'
    '    encoder.setBuffer(metalBuffer, offset: 0, index: 0)\n'
    '    let gridSize = MTLSize(width: bufferCount, height: 1, depth: 1)\n'
    '    let groupSize = MTLSize(\n'
    '        width: min(pipeline.maxTotalThreadsPerThreadgroup, bufferCount),\n'
    '        height: 1, depth: 1\n'
    '    )\n'
    '    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)\n'
    '    encoder.endEncoding()\n'
    '    commandBuffer.commit()\n'
    '    commandBuffer.waitUntilCompleted()\n'
    '}\n'
    '\n'
    'fputs("Metal compute completed successfully\\n", stderr)\n'
)

pytestmark = pytest.mark.metal_integration


def _has_metal() -> bool:
    """Check if Metal is available (macOS with GPU)."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "Metal" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_xctrace() -> bool:
    """Check if xctrace is available."""
    try:
        result = subprocess.run(
            ["xcrun", "xctrace", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_swiftc() -> bool:
    """Check if Swift compiler is available."""
    try:
        result = subprocess.run(
            ["xcrun", "swiftc", "--version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


skip_reasons: list[str] = []
if not _has_metal():
    skip_reasons.append("no Metal GPU")
if not _has_xctrace():
    skip_reasons.append("xctrace not available")
if not _has_swiftc():
    skip_reasons.append("swiftc not available")


@pytest.fixture(scope="module")
def metal_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the minimal Metal compute program."""
    build_dir = tmp_path_factory.mktemp("metal_build")
    source_file = build_dir / "metal_compute.swift"
    source_file.write_text(METAL_PROGRAM_SOURCE)
    binary = build_dir / "metal_compute"

    result = subprocess.run(
        [
            "xcrun", "swiftc",
            "-O",
            "-framework", "Metal",
            "-framework", "Foundation",
            str(source_file),
            "-o", str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"swiftc failed: {result.stderr}"
    assert binary.exists()
    return binary


@pytest.fixture(scope="module")
def metal_trace(metal_binary: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Record the Metal program with xctrace Metal System Trace."""
    trace_dir = tmp_path_factory.mktemp("metal_traces")
    trace_path = trace_dir / "metal_compute.trace"

    result = subprocess.run(
        [
            "xcrun", "xctrace", "record",
            "--template", "Metal System Trace",
            "--output", str(trace_path),
            "--time-limit", "10s",
            "--launch", "--", str(metal_binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"xctrace record failed: {result.stderr}"
    assert trace_path.exists()
    return trace_path


@pytest.mark.skipif(bool(skip_reasons), reason="; ".join(skip_reasons))
class TestMetalTraceAnalysis:
    """Test that Metal System Trace data is accessible via the profiler."""

    def test_trace_opens(self, metal_trace: Path) -> None:
        """Trace should open and contain Metal-related tables."""
        t = TraceFile(metal_trace)
        info = t.info
        assert info.template_name == "Metal System Trace"
        assert info.duration_seconds > 0

    def test_has_metal_tables(self, metal_trace: Path) -> None:
        """Trace should contain at least some Metal tables."""
        t = TraceFile(metal_trace)
        schemas = {tb.schema for tb in t.tables()}
        metal_schemas = {s for s in schemas if "metal" in s.lower()}
        assert len(metal_schemas) > 0, (
            f"No Metal tables found. Available schemas: {sorted(schemas)}"
        )

    def test_metal_driver_intervals(self, metal_trace: Path) -> None:
        """metal-driver-intervals should have data from our compute work."""
        t = TraceFile(metal_trace)
        if not t.has_table("metal-driver-intervals"):
            pytest.skip("metal-driver-intervals table not present")

        table = t.load_table("metal-driver-intervals")
        assert len(table.rows) > 0, "metal-driver-intervals has no rows"
        assert len(table.columns) > 0

        # Verify we can serialize rows (the generic query path)
        col_index = {col.mnemonic: i for i, col in enumerate(table.columns)}
        row = table.rows[0]
        row_dict = {}
        for mnemonic, idx in col_index.items():
            if idx < len(row):
                row_dict[mnemonic] = row[idx].value
        assert len(row_dict) > 0

    def test_metal_application_intervals(self, metal_trace: Path) -> None:
        """metal-application-intervals should capture our command buffer submissions."""
        t = TraceFile(metal_trace)
        if not t.has_table("metal-application-intervals"):
            pytest.skip("metal-application-intervals table not present")

        table = t.load_table("metal-application-intervals")
        assert len(table.rows) > 0, "metal-application-intervals has no rows"

    def test_generic_query_serialization(self, metal_trace: Path) -> None:
        """The generic table query should produce valid JSON for Metal tables."""
        t = TraceFile(metal_trace)
        # Find any Metal table with data
        metal_table = None
        for tb in t.tables():
            if "metal" in tb.schema.lower():
                table = t.load_table(tb.schema)
                if len(table.rows) > 0:
                    metal_table = table
                    break

        if metal_table is None:
            pytest.skip("No Metal tables with data")

        # Simulate what profiler_query_table does
        col_index = {col.mnemonic: i for i, col in enumerate(metal_table.columns)}
        rows_data = []
        for row in metal_table.rows[:10]:
            row_dict: dict[str, str] = {}
            for mnemonic, idx in col_index.items():
                if idx < len(row):
                    row_dict[mnemonic] = row[idx].value
            rows_data.append(row_dict)

        result = {
            "schema": metal_table.schema_name,
            "columns": [
                {"mnemonic": col.mnemonic, "name": col.name, "type": col.engineering_type}
                for col in metal_table.columns
            ],
            "total_rows": len(metal_table.rows),
            "returned": len(rows_data),
            "rows": rows_data,
        }

        # Should be valid JSON
        json_str = json.dumps(result, indent=2)
        parsed = json.loads(json_str)
        assert parsed["total_rows"] > 0
        assert len(parsed["rows"]) > 0
        assert len(parsed["columns"]) > 0

    def test_metal_tables_summary(self, metal_trace: Path) -> None:
        """Print a summary of all Metal tables and their row counts (diagnostic)."""
        t = TraceFile(metal_trace)
        metal_tables = []
        for tb in t.tables():
            if "metal" in tb.schema.lower() or "gpu" in tb.schema.lower():
                try:
                    table = t.load_table(tb.schema)
                    metal_tables.append((tb.schema, len(table.rows)))
                except Exception:
                    metal_tables.append((tb.schema, -1))

        assert len(metal_tables) > 0, "No Metal/GPU tables found"
        # At least one Metal table should have data
        tables_with_data = [(name, count) for name, count in metal_tables if count > 0]
        assert len(tables_with_data) > 0, (
            f"No Metal tables have data. Tables: {metal_tables}"
        )
