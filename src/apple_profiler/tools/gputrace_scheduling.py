# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyobjc-core",
#     "pyobjc-framework-Cocoa",
# ]
# ///
"""Analyze GPU dispatch scheduling overhead from .gputrace files.

Identifies performance opportunities where combining GPU operations
would eliminate CPU-GPU scheduling gaps. Uses real GPU timestamps from
the MIO hardware profiler to measure:

  1. Inter-encoder gaps: Time lost between command encoder submissions
     (the GPU-CPU-GPU roundtrip cost of ending one encoder and starting
     the next). Combining encoders eliminates these gaps entirely.

  2. Dispatch fusion candidates: Sequences of many small dispatches of
     the same kernel that could be replaced by fewer, larger dispatches
     to reduce per-dispatch scheduling overhead.

  3. Overlapping dispatch analysis: How well the GPU pipelines
     dispatches within each encoder (concurrency ratio).

Requirements:
  - macOS with Xcode installed
  - .gputrace file with shader profiling data (streamData)
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

try:
    from ._frameworks import ensure_dyld_framework_path
except ImportError:
    from _frameworks import ensure_dyld_framework_path  # type: ignore[no-redef]

ensure_dyld_framework_path()

try:
    from .gputrace_timeline import read_gputrace, read_gputrace_timestamps
except ImportError:
    from gputrace_timeline import read_gputrace, read_gputrace_timestamps  # type: ignore[no-redef]


def analyze_scheduling(gputrace_path: str) -> dict[str, Any]:
    """Analyze dispatch scheduling overhead from a .gputrace file.

    Returns a structured dict with:
      - summary: Overall scheduling overhead statistics
      - encoders: Per-encoder timing breakdown
      - encoder_gaps: Inter-encoder gap details with combine assessment
      - fusion_candidates: Sequences of small dispatches that could be fused
      - recommendations: Prioritized list of optimization suggestions
    """
    data = read_gputrace(gputrace_path)
    ts_result = read_gputrace_timestamps(gputrace_path, event_data=data)

    if not ts_result or not ts_result.get("timestamps"):
        return {
            "error": "No GPU timestamps available. Requires shader profiling data (streamData)."
        }

    timestamps = ts_result["timestamps"]
    timeline_end_ns = ts_result["timeline_end_ns"]
    gpu_time_ns = ts_result["gpu_time_ns"]

    # Build dispatch list with timestamps
    dispatches = []
    for e in data["events"]:
        if e["type"] == "dispatch":
            fidx = e["index"]
            ts = timestamps.get(fidx)
            if ts:
                dispatches.append(
                    {
                        "func_idx": fidx,
                        "ts_begin": ts[0],
                        "ts_end": ts[1],
                        "duration": ts[1] - ts[0],
                        "kernel": e.get("kernel", "unknown"),
                        "encoder_idx": e.get("encoder_idx", -1),
                        "threadgroups": e.get("threadgroups"),
                        "threads_per_tg": e.get("threads_per_threadgroup"),
                    }
                )

    if not dispatches:
        return {"error": "No dispatches with timestamps found."}

    dispatches.sort(key=lambda d: d["ts_begin"])

    # ── Group by encoder ──
    encoders_map: dict[int, list[dict]] = {}
    for d in dispatches:
        eidx = d["encoder_idx"]
        if eidx not in encoders_map:
            encoders_map[eidx] = []
        encoders_map[eidx].append(d)

    # ── Per-encoder analysis ──
    encoder_results = []
    for eidx in sorted(encoders_map.keys()):
        enc = encoders_map[eidx]
        n = len(enc)
        first_ts = enc[0]["ts_begin"]
        last_ts = enc[-1]["ts_end"]
        span_ns = last_ts - first_ts
        total_compute_ns = sum(d["duration"] for d in enc)
        concurrency = total_compute_ns / span_ns if span_ns > 0 else 0

        # Kernel breakdown
        kernels: dict[str, dict] = {}
        for d in enc:
            k = d["kernel"]
            if k not in kernels:
                kernels[k] = {"count": 0, "total_ns": 0, "durations": []}
            kernels[k]["count"] += 1
            kernels[k]["total_ns"] += d["duration"]
            kernels[k]["durations"].append(d["duration"])

        kernel_list = []
        for kname, kinfo in sorted(kernels.items(), key=lambda x: -x[1]["total_ns"]):
            pct = kinfo["total_ns"] / total_compute_ns * 100 if total_compute_ns > 0 else 0
            kernel_list.append(
                {
                    "kernel": kname,
                    "count": kinfo["count"],
                    "total_us": round(kinfo["total_ns"] / 1e3, 1),
                    "avg_us": round(kinfo["total_ns"] / kinfo["count"] / 1e3, 1),
                    "pct_of_encoder": round(pct, 1),
                }
            )

        encoder_results.append(
            {
                "encoder_idx": eidx,
                "dispatch_count": n,
                "span_ms": round(span_ns / 1e6, 3),
                "total_compute_ms": round(total_compute_ns / 1e6, 3),
                "concurrency": round(concurrency, 1),
                "kernels": kernel_list,
            }
        )

    # ── Inter-encoder gaps ──
    encoder_gaps = []
    total_gap_ns = 0
    enc_list = sorted(encoders_map.keys())
    for i in range(1, len(enc_list)):
        prev_enc = encoders_map[enc_list[i - 1]]
        curr_enc = encoders_map[enc_list[i]]

        prev_end = prev_enc[-1]["ts_end"]
        curr_begin = curr_enc[0]["ts_begin"]
        gap_ns = curr_begin - prev_end

        if gap_ns <= 0:
            continue

        total_gap_ns += gap_ns

        prev_kernels = set(d["kernel"] for d in prev_enc)
        curr_kernels = set(d["kernel"] for d in curr_enc)
        shared = prev_kernels & curr_kernels

        prev_span = (prev_enc[-1]["ts_end"] - prev_enc[0]["ts_begin"]) / 1e6
        curr_span = (curr_enc[-1]["ts_end"] - curr_enc[0]["ts_begin"]) / 1e6

        # Assessment
        if shared:
            assessment = "COMBINABLE"
            reason = (
                f"Encoders share {len(shared)} kernel(s): "
                f"{', '.join(sorted(shared)[:3])}"
                f"{' ...' if len(shared) > 3 else ''}. "
                f"Merging into one encoder would eliminate {gap_ns / 1e3:.0f}us gap."
            )
        else:
            assessment = "COMBINABLE_DIFFERENT_KERNELS"
            reason = (
                f"Different kernels but same command buffer type (compute). "
                f"Can merge into one encoder if no CPU readback needed between them. "
                f"Would eliminate {gap_ns / 1e3:.0f}us gap."
            )

        encoder_gaps.append(
            {
                "from_encoder": enc_list[i - 1],
                "to_encoder": enc_list[i],
                "gap_us": round(gap_ns / 1e3, 1),
                "gap_ms": round(gap_ns / 1e6, 3),
                "from_dispatch_count": len(prev_enc),
                "from_span_ms": round(prev_span, 3),
                "to_dispatch_count": len(curr_enc),
                "to_span_ms": round(curr_span, 3),
                "shared_kernels": sorted(shared) if shared else [],
                "assessment": assessment,
                "reason": reason,
            }
        )

    # ── Dispatch fusion candidates ──
    # Find runs of same kernel with small average dispatch duration
    fusion_candidates = []
    SMALL_DISPATCH_THRESHOLD_NS = 100_000  # 100 us

    for eidx in sorted(encoders_map.keys()):
        enc = encoders_map[eidx]

        # Find runs of same kernel
        runs: list[list[dict]] = []
        current_run = [enc[0]]
        for j in range(1, len(enc)):
            if enc[j]["kernel"] == enc[j - 1]["kernel"]:
                current_run.append(enc[j])
            else:
                if len(current_run) >= 3:  # At least 3 consecutive same-kernel dispatches
                    runs.append(current_run)
                current_run = [enc[j]]
        if len(current_run) >= 3:
            runs.append(current_run)

        for run in runs:
            durations = [d["duration"] for d in run]
            avg_ns = statistics.mean(durations)
            total_ns = sum(durations)
            span_ns = run[-1]["ts_end"] - run[0]["ts_begin"]
            overhead_ns = span_ns - total_ns if span_ns > total_ns else 0

            # Collect unique threadgroup sizes
            tg_sizes = set()
            for d in run:
                if d["threadgroups"]:
                    tg_sizes.add(tuple(d["threadgroups"]))

            # Only flag if dispatches are small or scheduling overhead is significant
            overhead_pct = overhead_ns / span_ns * 100 if span_ns > 0 else 0
            is_small = avg_ns < SMALL_DISPATCH_THRESHOLD_NS
            has_overhead = overhead_pct > 15

            if is_small or has_overhead:
                fusion_candidates.append(
                    {
                        "encoder_idx": eidx,
                        "kernel": run[0]["kernel"],
                        "dispatch_count": len(run),
                        "avg_duration_us": round(avg_ns / 1e3, 1),
                        "total_compute_us": round(total_ns / 1e3, 1),
                        "span_us": round(span_ns / 1e3, 1),
                        "scheduling_overhead_us": round(overhead_ns / 1e3, 1),
                        "scheduling_overhead_pct": round(overhead_pct, 1),
                        "threadgroup_sizes": [list(t) for t in sorted(tg_sizes)],
                        "varying_threadgroups": len(tg_sizes) > 1,
                        "reason": _fusion_reason(len(run), avg_ns, overhead_pct, len(tg_sizes) > 1),
                    }
                )

    # ── Build recommendations ──
    recommendations = _build_recommendations(
        encoder_gaps,
        fusion_candidates,
        encoder_results,
        total_gap_ns,
        timeline_end_ns,
        gpu_time_ns,
    )

    # ── Summary ──
    total_compute_ns = sum(d["duration"] for d in dispatches)
    summary = {
        "timeline_ms": round(timeline_end_ns / 1e6, 3),
        "gpu_active_ms": round(gpu_time_ns / 1e6, 3),
        "total_compute_ms": round(total_compute_ns / 1e6, 3),
        "dispatch_count": len(dispatches),
        "encoder_count": len(encoders_map),
        "inter_encoder_gap_ms": round(total_gap_ns / 1e6, 3),
        "inter_encoder_gap_pct": round(total_gap_ns / timeline_end_ns * 100, 1)
        if timeline_end_ns > 0
        else 0,
        "avg_concurrency": round(total_compute_ns / gpu_time_ns, 1) if gpu_time_ns > 0 else 0,
        "fusion_candidates_count": len(fusion_candidates),
        "potential_savings_ms": round(total_gap_ns / 1e6, 3),
    }

    return {
        "summary": summary,
        "encoders": encoder_results,
        "encoder_gaps": encoder_gaps,
        "fusion_candidates": fusion_candidates,
        "recommendations": recommendations,
    }


def _fusion_reason(count: int, avg_ns: float, overhead_pct: float, varying_tg: bool) -> str:
    """Generate a human-readable fusion recommendation."""
    parts = []
    if avg_ns < 10_000:  # < 10 us
        parts.append(f"{count} very short dispatches (avg {avg_ns / 1e3:.1f}us)")
    elif avg_ns < 100_000:  # < 100 us
        parts.append(f"{count} short dispatches (avg {avg_ns / 1e3:.1f}us)")

    if overhead_pct > 30:
        parts.append(f"high scheduling overhead ({overhead_pct:.0f}%)")
    elif overhead_pct > 15:
        parts.append(f"moderate scheduling overhead ({overhead_pct:.0f}%)")

    if varying_tg:
        parts.append("varying threadgroup sizes (may need indirect dispatch to fuse)")
    else:
        parts.append("same threadgroup size (straightforward to fuse)")

    return "; ".join(parts) + "."


def _build_recommendations(
    encoder_gaps: list[dict],
    fusion_candidates: list[dict],
    encoder_results: list[dict],
    total_gap_ns: int,
    timeline_end_ns: int,
    gpu_time_ns: int,
) -> list[dict]:
    """Build prioritized optimization recommendations."""
    recs = []

    # Recommendation 1: Combine encoders
    if encoder_gaps:
        combined_savings_ms = sum(g["gap_ms"] for g in encoder_gaps)
        savings_pct = total_gap_ns / timeline_end_ns * 100 if timeline_end_ns > 0 else 0

        if len(encoder_gaps) == 1:
            gap = encoder_gaps[0]
            recs.append(
                {
                    "priority": 1,
                    "type": "COMBINE_ENCODERS",
                    "savings_ms": round(combined_savings_ms, 3),
                    "savings_pct": round(savings_pct, 1),
                    "description": (
                        f"Combine encoder {gap['from_encoder']} and {gap['to_encoder']} "
                        f"into a single compute encoder to eliminate {gap['gap_ms']:.1f}ms "
                        f"scheduling gap ({savings_pct:.0f}% of total GPU time)."
                    ),
                    "details": encoder_gaps,
                }
            )
        else:
            combinable = [g for g in encoder_gaps if "COMBINABLE" in g["assessment"]]
            if combinable:
                recs.append(
                    {
                        "priority": 1,
                        "type": "COMBINE_ENCODERS",
                        "savings_ms": round(combined_savings_ms, 3),
                        "savings_pct": round(savings_pct, 1),
                        "description": (
                            f"Combine {len(encoder_gaps) + 1} separate compute encoders into "
                            f"fewer encoders to eliminate {combined_savings_ms:.1f}ms of "
                            f"inter-encoder scheduling gaps "
                            f"({savings_pct:.0f}% of total GPU time). "
                            f"Each gap is a CPU-GPU roundtrip where the GPU sits idle waiting "
                            f"for the next encoder to be submitted."
                        ),
                        "details": encoder_gaps,
                    }
                )

    # Recommendation 2: Fuse small dispatches
    for fc in sorted(fusion_candidates, key=lambda x: -x["scheduling_overhead_us"]):
        recs.append(
            {
                "priority": 2 if fc["scheduling_overhead_us"] > 100 else 3,
                "type": "FUSE_DISPATCHES",
                "savings_ms": round(fc["scheduling_overhead_us"] / 1e3, 3),
                "savings_pct": round(fc["scheduling_overhead_us"] * 1e3 / timeline_end_ns * 100, 1)
                if timeline_end_ns > 0
                else 0,
                "description": (
                    f"Fuse {fc['dispatch_count']}x '{fc['kernel']}' dispatches "
                    f"(encoder {fc['encoder_idx']}, avg {fc['avg_duration_us']:.0f}us each) "
                    f"into fewer, larger dispatches. "
                    f"Current overhead: {fc['scheduling_overhead_pct']:.0f}% of span."
                ),
                "details": fc,
            }
        )

    # Recommendation 3: GPU utilization note
    gpu_idle_pct = (
        (timeline_end_ns - gpu_time_ns) / timeline_end_ns * 100 if timeline_end_ns > 0 else 0
    )
    if gpu_idle_pct > 10 and not encoder_gaps:
        recs.append(
            {
                "priority": 3,
                "type": "GPU_IDLE",
                "savings_ms": round((timeline_end_ns - gpu_time_ns) / 1e6, 3),
                "savings_pct": round(gpu_idle_pct, 1),
                "description": (
                    f"GPU is idle {gpu_idle_pct:.0f}% of the timeline. "
                    f"Consider overlapping CPU work with GPU execution or "
                    f"submitting commands earlier."
                ),
                "details": None,
            }
        )

    return sorted(recs, key=lambda r: (r["priority"], -r["savings_ms"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze GPU dispatch scheduling overhead",
    )
    parser.add_argument("gputrace_path", help="Path to .gputrace file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = analyze_scheduling(args.gputrace_path)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Human-readable output
    s = result["summary"]
    print(f"{'=' * 80}")
    print("GPU SCHEDULING OVERHEAD ANALYSIS")
    print(f"{'=' * 80}")
    print(f"  Timeline: {s['timeline_ms']:.2f} ms")
    print(f"  GPU active: {s['gpu_active_ms']:.2f} ms")
    print(f"  Dispatches: {s['dispatch_count']} across {s['encoder_count']} encoders")
    print(f"  Avg concurrency: {s['avg_concurrency']:.1f}x overlapping dispatches")
    print(
        f"  Inter-encoder gaps: {s['inter_encoder_gap_ms']:.2f} ms "
        f"({s['inter_encoder_gap_pct']:.1f}%)"
    )

    if result["encoder_gaps"]:
        print(f"\n{'=' * 80}")
        print("INTER-ENCODER GAPS (GPU idle between encoder submissions)")
        print(f"{'=' * 80}")
        for g in result["encoder_gaps"]:
            print(
                f"\n  Encoder {g['from_encoder']} -> {g['to_encoder']}: "
                f"{g['gap_us']:.0f} us ({g['gap_ms']:.2f} ms)"
            )
            print(f"    From: {g['from_dispatch_count']} dispatches, {g['from_span_ms']:.2f} ms")
            print(f"    To:   {g['to_dispatch_count']} dispatches, {g['to_span_ms']:.2f} ms")
            print(f"    {g['assessment']}: {g['reason']}")

    if result["fusion_candidates"]:
        print(f"\n{'=' * 80}")
        print("DISPATCH FUSION CANDIDATES")
        print(f"{'=' * 80}")
        for fc in result["fusion_candidates"]:
            print(f"\n  Encoder {fc['encoder_idx']}: {fc['dispatch_count']}x '{fc['kernel']}'")
            print(
                f"    Avg duration: {fc['avg_duration_us']:.1f} us, "
                f"overhead: {fc['scheduling_overhead_pct']:.0f}%"
            )
            print(f"    {fc['reason']}")

    if result["recommendations"]:
        print(f"\n{'=' * 80}")
        print("RECOMMENDATIONS (by priority)")
        print(f"{'=' * 80}")
        for i, rec in enumerate(result["recommendations"], 1):
            savings = rec["savings_ms"]
            pct = rec["savings_pct"]
            print(f"\n  {i}. [{rec['type']}] Save ~{savings:.2f} ms ({pct:.1f}%)")
            print(f"     {rec['description']}")


if __name__ == "__main__":
    main()
