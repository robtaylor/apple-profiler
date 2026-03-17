# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyobjc-core",
#     "pyobjc-framework-Cocoa",
# ]
# ///
"""Diagnostic: hexdump MTSP records for setBuffer:offset:atIndex: calls.

Investigates whether the MTSP stream records encode read vs write access
modes for buffer bindings. This is needed to determine if we can build
accurate (non-conservative) dependency graphs.

Usage:
    uv run tools/gputrace_dump_setbuffer.py /path/to/capture.gputrace
"""
from __future__ import annotations

import ctypes
import os
import re
import struct
import sys
import warnings
from collections import Counter

import objc  # type: ignore[import-untyped]
from Foundation import NSBundle, NSURL  # type: ignore[import-untyped]

warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)

SHARED_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"
_FRAMEWORK_NAMES = [
    "GPUToolsCore", "GPUTools", "GPUToolsPlatform",
    "GLToolsCore", "GPUToolsServices",
]


def _ensure_dyld_framework_path() -> None:
    """Re-exec with DYLD_FRAMEWORK_PATH if not set."""
    if os.environ.get("DYLD_FRAMEWORK_PATH") != SHARED_FW:
        os.environ["DYLD_FRAMEWORK_PATH"] = SHARED_FW
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _load_frameworks() -> None:
    for name in _FRAMEWORK_NAMES:
        bundle = NSBundle.bundleWithPath_(f"{SHARED_FW}/{name}.framework")
        if bundle is not None:
            bundle.load()
    sys_bundle = NSBundle.bundleWithPath_(
        "/System/Library/PrivateFrameworks/GPUToolsCapture.framework"
    )
    if sys_bundle is not None:
        sys_bundle.load()


_load_frameworks()

DYCaptureArchive = objc.lookUpClass("DYCaptureArchive")
DYFunctionTracer = objc.lookUpClass("DYFunctionTracer")

SET_BUFFER_IDX = -16336  # setBuffer:offset:atIndex:

# Also dump these related calls for comparison
# (Corrected indices from rosetta stone verification 2026-03-17)
RELATED_INDICES = {
    -16337: "setBytes:length:atIndex:",
    -16338: "setComputePipelineState:",
    -16327: "dispatchThreadgroups:threadsPerThreadgroup:",
    -16078: "dispatchThreads:threadsPerThreadgroup: (non-uniform)",
    -16009: "memoryBarrierWithScope: (NOT a dispatch!)",
}


def hexdump(data: bytes, width: int = 16) -> str:
    """Format bytes as a hex dump with ASCII sidebar."""
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {i:4d}: {hex_part:<{width * 3}}  {ascii_part}")
    return "\n".join(lines)


def main() -> None:
    _ensure_dyld_framework_path()
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/baspacho_ffi.gputrace"

    url = NSURL.fileURLWithPath_(path)
    archive = DYCaptureArchive.alloc().initWithURL_options_error_(url, 0, None)
    if archive is None:
        print(f"ERROR: Failed to open: {path}", file=sys.stderr)
        sys.exit(1)

    capture_file = archive.openFileWithFilename_error_("unsorted-capture", None)
    fstream = capture_file.openFunctionStream_(None)

    tracer = DYFunctionTracer.alloc().init()
    tracer.setCompact_(True)
    tracer.setNativePointerSize_(8)

    max_dump = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    count = 0
    func_idx = 0

    # Collect MTSP record byte patterns to find distinguishing fields
    mtsp_byte_patterns: list[bytes] = []
    # Track buffer indices seen with their MTSP patterns
    buffer_index_patterns: dict[int, list[bytes]] = {}

    print(f"=== MTSP Record Dump for setBuffer:offset:atIndex: ===")
    print(f"File: {path}")
    print(f"Dumping first {max_dump} records\n")

    while True:
        func_ptr = fstream.readFunction()
        if func_ptr is None:
            break

        ptr_addr = func_ptr.pointerAsInteger
        raw_header = ctypes.string_at(ptr_addr, 8)
        idx = struct.unpack_from("<i", raw_header, 0)[0]

        if idx == SET_BUFFER_IDX:
            # Read the full function struct (enough to get MTSP pointer)
            # Function struct is at least 448 bytes based on format docs
            func_struct = ctypes.string_at(ptr_addr, 512)

            # Extract arguments from trace text
            trace = str(tracer.traceFunction_error_(func_ptr, None))

            # Extract buffer address from trace
            hex_addrs = [
                int(m.group(1), 16)
                for m in re.finditer(r"0x([0-9a-fA-F]+)l?", trace)
            ]
            buffer_addr = hex_addrs[0] if hex_addrs else 0

            # Extract buffer index from trace
            m = re.search(r"(\d+)ul\)$", trace)
            buf_index = int(m.group(1)) if m else -1

            # Read MTSP record pointer at offset 440
            mtsp_ptr = struct.unpack_from("<Q", func_struct, 440)[0]

            # Also read surrounding fields for context
            f9 = struct.unpack_from("<Q", func_struct, 432)[0]
            f11 = struct.unpack_from("<Q", func_struct, 448)[0]

            if count < max_dump:
                print(f"--- setBuffer #{count} (func_idx={func_idx}) ---")
                print(f"  Trace: {trace}")
                print(f"  Buffer: 0x{buffer_addr:x}, Index: {buf_index}")
                print(f"  Offsets 432-456: f9=0x{f9:x} f10/MTSP=0x{mtsp_ptr:x} f11=0x{f11:x}")

            # Try to read MTSP record if pointer looks valid
            if mtsp_ptr != 0 and mtsp_ptr > 0x1000:
                try:
                    mtsp_data = ctypes.string_at(mtsp_ptr, 128)
                    mtsp_byte_patterns.append(mtsp_data)

                    if buf_index not in buffer_index_patterns:
                        buffer_index_patterns[buf_index] = []
                    buffer_index_patterns[buf_index].append(mtsp_data)

                    if count < max_dump:
                        print(f"  MTSP record (128 bytes from 0x{mtsp_ptr:x}):")
                        print(hexdump(mtsp_data))

                        # Check for common tags
                        if b"Cul" in mtsp_data:
                            print("  ** Found 'Cul' tag (possible READ marker)")
                        if b"Cuw" in mtsp_data:
                            print("  ** Found 'Cuw' tag (possible WRITE marker)")
                        if b"read" in mtsp_data.lower():
                            print("  ** Found 'read' string")
                        if b"write" in mtsp_data.lower():
                            print("  ** Found 'write' string")

                except Exception as e:
                    if count < max_dump:
                        print(f"  MTSP read failed: {e}")
            else:
                if count < max_dump:
                    print(f"  MTSP pointer is null/invalid: 0x{mtsp_ptr:x}")

            if count < max_dump:
                # Also dump the argument region (offsets 24-120) for access mode bits
                print(f"  Argument region (offsets 24-120):")
                print(hexdump(func_struct[24:120]))
                print()

            count += 1

        func_idx += 1

    print(f"\n=== Summary ===")
    print(f"Total setBuffer calls: {count}")

    if mtsp_byte_patterns:
        print(f"\nMTSP records collected: {len(mtsp_byte_patterns)}")

        # Analyze byte-by-byte variance to find fields that differ
        print("\nByte variance analysis (first 64 bytes of MTSP records):")
        for offset in range(min(64, len(mtsp_byte_patterns[0]))):
            values = Counter(p[offset] for p in mtsp_byte_patterns)
            if len(values) > 1:
                top = values.most_common(3)
                vals_str = ", ".join(
                    f"0x{v:02x}({c})" for v, c in top
                )
                print(f"  Offset {offset:3d}: {len(values)} distinct values: {vals_str}")

        # Look for consistent differences between buffer indices
        print("\nPer-buffer-index MTSP patterns:")
        for bidx in sorted(buffer_index_patterns.keys()):
            patterns = buffer_index_patterns[bidx]
            print(f"  Buffer index {bidx}: {len(patterns)} records")
            if patterns:
                # Show first 32 bytes of first record
                print(f"    First record (32 bytes): {patterns[0][:32].hex()}")


if __name__ == "__main__":
    main()
