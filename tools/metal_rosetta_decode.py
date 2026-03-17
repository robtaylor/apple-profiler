# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyobjc-core",
#     "pyobjc-framework-Cocoa",
# ]
# ///
"""Decode all rosetta variant traces and display side-by-side."""
import ctypes
import os
import struct
import sys
import warnings

SHARED_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"
if os.environ.get("DYLD_FRAMEWORK_PATH") != SHARED_FW:
    os.environ["DYLD_FRAMEWORK_PATH"] = SHARED_FW
    os.execv(sys.executable, [sys.executable] + sys.argv)

import objc  # noqa: E402
from Foundation import NSBundle, NSURL  # noqa: E402
warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)

for name in [
    "GPUToolsCore", "GPUTools", "GPUToolsPlatform",
    "GLToolsCore", "GPUToolsServices",
]:
    bundle = NSBundle.bundleWithPath_(f"{SHARED_FW}/{name}.framework")
    if bundle is not None:
        bundle.load()

sys_bundle = NSBundle.bundleWithPath_(
    "/System/Library/PrivateFrameworks/GPUToolsCapture.framework"
)
if sys_bundle is not None:
    sys_bundle.load()

DYCaptureArchive = objc.lookUpClass("DYCaptureArchive")
DYFunctionTracer = objc.lookUpClass("DYFunctionTracer")


def read_stream(path: str) -> list[tuple[int, str]]:
    url = NSURL.fileURLWithPath_(path)
    archive = DYCaptureArchive.alloc().initWithURL_options_error_(url, 0, None)
    if archive is None:
        return []
    capture_file = archive.openFileWithFilename_error_("unsorted-capture", None)
    fstream = capture_file.openFunctionStream_(None)
    tracer = DYFunctionTracer.alloc().init()
    tracer.setCompact_(True)
    tracer.setNativePointerSize_(8)
    results = []
    while True:
        func_ptr = fstream.readFunction()
        if func_ptr is None:
            break
        ptr_addr = func_ptr.pointerAsInteger
        raw_header = ctypes.string_at(ptr_addr, 8)
        idx = struct.unpack_from("<i", raw_header, 0)[0]
        try:
            trace = str(tracer.traceFunction_error_(func_ptr, None))
        except Exception:
            trace = "<error>"
        results.append((idx, trace))
    return results


# Known mappings (updated)
KNOWN = {
    -16371: "setPurgeableState:",
    -16370: "endEncoding (alt)",
    -16363: "commit",
    -16361: "waitUntilCompleted",
    -16355: "computeCommandEncoder",
    -16352: "commandBuffer",
    -16343: "newBufferWithLength",
    -16338: "setComputePipelineState:",
    -16337: "setBytes:length:atIndex:",
    -16336: "setBuffer:offset:atIndex:",
    -16327: "dispatchThreadgroups:",
    -16325: "endEncoding",
    -16316: "newComputePipelineState: (v2)",
    -16314: "newBufferWithBytes",
    -16299: "newComputePipelineState:",
    -16290: "newFunctionWithName:",
    -16227: "setBytes:inline",
    -16067: "supportsFamily:",
    -16078: "dispatchThreads:",
    -16009: "memoryBarrierWithScope:",
    -16008: "memoryBarrier(resources:)",
    -15996: "newCPSwithDescriptor:",
    -15990: "addCompletedHandler:",
    -15736: "newLibraryWithURL:",
    -15422: "SharedEvent.notifyListener:",
    -10228: "GPUTools.bufferDidModify",
    -10223: "GPUTools.bufferContents",
    -10203: "GPUTools.resourceID",
    -10191: "GPUTools.bufferGPUAddress",
    -10186: "GPUTools.bufferCompleted",
}

VARIANT_LABELS = {
    0: "Baseline (dispatch, barrier, dispatch, 2 pipelines)",
    1: "dispatchThreads (non-uniform) instead of dispatchThreadgroups",
    2: "setBytes with inline data",
    3: "Two encoders in one command buffer",
    4: "memoryBarrier(resources:) instead of scope",
    5: "Two command buffers",
    6: "Blit encoder (test endEncoding index)",
    7: "makeBuffer(bytes:) — test -16314",
    8: "addCompletedHandler — test -15990",
    9: "makeComputePipelineState(descriptor:) — test -15996",
    10: "MTLSharedEvent — test -15422",
    11: "makeFunction(name:) — test -16290",
    12: "setPurgeableState — test -16371",
}

base_dir = "/tmp/claude/barrier_test"

MAX_V = 13
for v in range(MAX_V):
    path = f"{base_dir}/rosetta_v{v}.gputrace"
    if not os.path.exists(path):
        continue
    label = VARIANT_LABELS.get(v, f"Variant {v}")
    print(f"\n{'='*80}")
    print(f"VARIANT {v}: {label}")
    print(f"{'='*80}")
    funcs = read_stream(path)
    for i, (idx, trace) in enumerate(funcs):
        name = KNOWN.get(idx, f"**UNKNOWN({idx})**")
        # Shorten trace
        t = trace[:90] + "..." if len(trace) > 90 else trace
        print(f"  [{i:3d}] {idx:7d}  {name:35s}  {t}")
    print(f"  Total: {len(funcs)} functions")

    # Summary: just the indices
    indices = [idx for idx, _ in funcs]
    unknowns = [idx for idx in indices if idx not in KNOWN]
    if unknowns:
        print(f"  *** UNKNOWN INDICES: {set(unknowns)}")

# Cross-variant comparison: find indices unique to specific variants
print(f"\n{'='*80}")
print("CROSS-VARIANT INDEX COMPARISON")
print(f"{'='*80}")

all_indices: dict[int, set[int]] = {}  # idx -> set of variants containing it
for v in range(MAX_V):
    path = f"{base_dir}/rosetta_v{v}.gputrace"
    if not os.path.exists(path):
        continue
    funcs = read_stream(path)
    for idx, _ in funcs:
        if idx not in all_indices:
            all_indices[idx] = set()
        all_indices[idx].add(v)

for idx in sorted(all_indices.keys()):
    variants = sorted(all_indices[idx])
    name = KNOWN.get(idx, f"**UNKNOWN({idx})**")
    v_str = ",".join(str(v) for v in variants)
    marker = ""
    if len(variants) < 6:
        marker = f"  ← only in variant(s) {v_str}"
    print(f"  {idx:7d}  {name:35s}  variants: [{v_str}]{marker}")
