"""Single source of truth for GPU performance counter taxonomy.

Each counter belongs to a named category with a display priority. Both
gputrace_timeline.py and gputrace_perfetto.py derive their data structures
from this canonical definition, eliminating counter name duplication.
"""

from __future__ import annotations

# Ordered list of (group_name, counter_names). Position in the list determines
# sort priority (lower index = higher priority). Counter order within each
# group is also significant (determines display order).
COUNTER_CATEGORIES: list[tuple[str, list[str]]] = [
    (
        "GPU Activity",
        [
            "GT Active Core Count",
            "Raytracing Active GT",
        ],
    ),
    (
        "Occupancy",
        [
            "Total Occupancy",
            "Compute Occupancy",
            "Fragment Occupancy",
            "Vertex Occupancy",
            "Occupancy Manager Target",
            "Total Simdgroups Inflight Per Shader Core",
            "Compute Simdgroups Inflight Per Shader Core",
            "Fragment Simdgroups Inflight Per Shader Core",
            "Vertex Simdgroups Inflight Per Shader Core",
            "Occupancy Management L1 Eviction Rate",
        ],
    ),
    (
        "Memory Bandwidth",
        [
            "AF Bandwidth",
            "AF Read Bandwidth",
            "AF Write Bandwidth",
            "AF Peak Bandwidth",
            "AF Peak Read Bandwidth",
            "AF Peak Write Bandwidth",
            "L2 Bandwidth",
        ],
    ),
    (
        "Shader Utilization",
        [
            "Shader Core Utilization",
            "Shader Core Limiter",
            "ALU Utilization",
            "F16 Utilization",
            "F16 Limiter",
            "F32 Utilization",
            "F32 Limiter",
            "IC Utilization",
            "IC Limiter",
            "SCIB Utilization",
            "SCIB Limiter",
            "Control Flow Utilization",
            "Control Flow Limiter",
            "Instruction Dispatch Utilization",
            "Instruction Dispatch Limiter",
            "Instruction Issue Utilization",
            "Instruction Issue Limiter",
            "Address Generation Utilization",
            "Address Generation Limiter",
        ],
    ),
    (
        "Shader Launch",
        [
            "Compute Shader Launch Utilization",
            "Compute Shader Launch Limiter",
            "Fragment Shader Launch Utilization",
            "Fragment Shader Launch Limiter",
            "Vertex Shader Launch Utilization",
            "Vertex Shader Launch Limiter",
        ],
    ),
    (
        "L1 Cache",
        [
            "L1 Load Bandwidth",
            "L1 Store Bandwidth",
            "L1 Cache Utilization",
            "L1 Cache Limiter",
            "Buffer L1 Load Bandwidth",
            "Buffer L1 Store Bandwidth",
            "Buffer L1 Load Ratio",
            "Buffer L1 Store Ratio",
            "Buffer L1 Miss Rate",
            "Imageblock L1 Load Bandwidth",
            "Imageblock L1 Store Bandwidth",
            "Imageblock L1 Load Ratio",
            "Imageblock L1 Store Ratio",
            "Threadgroup Memory L1 Load Bandwidth",
            "Threadgroup Memory L1 Store Bandwidth",
            "Threadgroup L1 Load Ratio",
            "Threadgroup L1 Store Ratio",
            "Stack L1 Load Bandwidth",
            "Stack L1 Store Bandwidth",
            "Stack L1 Load Ratio",
            "Stack L1 Store Ratio",
            "GPR L1 Load Bandwidth",
            "GPR L1 Store Bandwidth",
            "GPR L1 Read Ratio",
            "GPR L1 Write Ratio",
            "Other L1 Load Bandwidth",
            "Other L1 Store Bandwidth",
            "Other L1 Loads Ratio",
            "Other L1 Stores Ratio",
        ],
    ),
    (
        "L1 Residency",
        [
            "L1 Total Occupancy",
            "L1 Total Bytes Occupancy",
            "L1 Buffer Occupancy",
            "L1 Buffer Bytes Occupancy",
            "L1 Imageblock Occupancy",
            "L1 Imageblock Bytes Occupancy",
            "L1 Threadgroup Occupancy",
            "L1 Threadgroup Bytes Occupancy",
            "L1 GPR Occupancy",
            "L1 GPR Bytes Occupancy",
            "L1 Stack Occupancy",
            "L1 Stack Bytes Occupancy",
            "L1 Other Occupancy",
            "L1 Other Bytes Occupancy",
            "L1 Raytracing Scratch Occupancy",
            "L1 Raytracing Scratch Bytes Occupancy",
        ],
    ),
    (
        "L2 / Texture / MMU",
        [
            "L2 Cache Utilization",
            "L2 Cache Limiter",
            "Texture Cache Utilization",
            "Texture Cache Limiter",
            "Texture Read Utilization",
            "Texture Read Limiter",
            "Texture Write Utilization",
            "Texture Write Limiter",
            "TextureFilteringLimiter",
            "CompressionRatioTextureMemoryRead",
            "MMU Utilization",
            "MMU Limiter",
        ],
    ),
    (
        "Raytracing",
        [
            "Raytracing Active",
            "Ray Occupancy",
            "Leaf Test Occupancy",
            "Ray T Leaf Test",
            "Raytracing Node Test",
            "Intersect Ray Threads",
            "Raytracing Scratch L1 Load Bandwidth",
            "Raytracing Scratch L1 Store Bandwidth",
            "Raytracing Scratch L1 Load Ratio",
            "Raytracing Scratch L1 Store Ratio",
        ],
    ),
]


def build_sort_map() -> dict[str, tuple[int, int]]:
    """Build counter name -> (category_priority, index_within_category).

    Used by gputrace_timeline for sorting counters in display order.
    """
    sort_map: dict[str, tuple[int, int]] = {}
    for cat_prio, (_, names) in enumerate(COUNTER_CATEGORIES):
        for idx, name in enumerate(names):
            sort_map[name] = (cat_prio, idx)
    return sort_map


def build_group_map() -> dict[str, str]:
    """Build counter name -> group display name.

    Used by gputrace_perfetto for Perfetto track grouping.
    """
    group_map: dict[str, str] = {}
    for group_name, names in COUNTER_CATEGORIES:
        for name in names:
            group_map[name] = group_name
    return group_map


def group_order() -> list[str]:
    """Return the ordered list of group display names."""
    return [group_name for group_name, _ in COUNTER_CATEGORIES]
