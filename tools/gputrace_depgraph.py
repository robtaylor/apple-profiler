# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyobjc-core",
#     "pyobjc-framework-Cocoa",
# ]
# ///
"""Build a dependency graph from GPU trace dispatch buffer hazards.

Analyzes .gputrace files to construct a DAG showing how compute dispatches
depend on each other through shared buffer resources. Detects RAW (read-after-
write), WAW (write-after-write), and WAR (write-after-read) hazards.

Since the .gputrace format does not encode buffer access modes (read vs write),
the tool operates in conservative mode by default: any shared buffer between
two dispatches creates a dependency edge. Future versions may use heuristics
or external annotations to classify access modes.

Output formats:
  - DOT (Graphviz): visual graph with command buffer clusters
  - JSON: machine-readable nodes, edges, and summary

Usage:
    DYLD_FRAMEWORK_PATH="/Applications/Xcode.app/Contents/SharedFrameworks" \\
        uv run tools/gputrace_depgraph.py /path/to/capture.gputrace

    # JSON only
    ... gputrace_depgraph.py trace.gputrace -f json -o deps.json

    # DOT only, no transitive reduction
    ... gputrace_depgraph.py trace.gputrace -f dot --no-reduce

    # Summary statistics only
    ... gputrace_depgraph.py trace.gputrace --summary-only

    # Filter to specific kernels
    ... gputrace_depgraph.py trace.gputrace --filter-kernel "lu_factor*"
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class AccessMode(Enum):
    READ = "READ"
    WRITE = "WRITE"
    READ_WRITE = "READ_WRITE"
    UNKNOWN = "UNKNOWN"


class DepType(Enum):
    RAW = "RAW"   # Read After Write — true dependency
    WAW = "WAW"   # Write After Write — output dependency
    WAR = "WAR"   # Write After Read — anti-dependency
    SHARED = "SHARED"  # Conservative: shared buffer, unknown access


@dataclass
class BufferBinding:
    buffer_addr: int
    buffer_index: int
    access_mode: AccessMode = AccessMode.UNKNOWN


@dataclass
class DispatchNode:
    dispatch_id: int
    func_idx: int
    kernel: str
    buffers: list[BufferBinding]
    threadgroups: tuple[int, ...] | None = None
    threads_per_threadgroup: tuple[int, ...] | None = None
    command_buffer_idx: int = -1


@dataclass
class DependencyEdge:
    source_id: int
    target_id: int
    dep_type: DepType
    buffer_addr: int


@dataclass
class DependencyGraph:
    nodes: list[DispatchNode] = field(default_factory=list)
    edges: list[DependencyEdge] = field(default_factory=list)
    # Adjacency: node_id → list of (target_id, edge)
    successors: dict[int, list[tuple[int, DependencyEdge]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    predecessors: dict[int, list[tuple[int, DependencyEdge]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_edge(self, edge: DependencyEdge) -> None:
        self.edges.append(edge)
        self.successors[edge.source_id].append((edge.target_id, edge))
        self.predecessors[edge.target_id].append((edge.source_id, edge))


# ---------------------------------------------------------------------------
# Extract dispatches from gputrace timeline
# ---------------------------------------------------------------------------

def _import_read_gputrace():
    """Import read_gputrace from the timeline tool."""
    tools_dir = str(Path(__file__).parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from gputrace_timeline import read_gputrace
    return read_gputrace


def extract_dispatches(
    trace_data: dict[str, Any],
) -> tuple[list[DispatchNode], int]:
    """Convert timeline events into DispatchNode list.

    Returns (nodes, num_command_buffers).
    """
    nodes: list[DispatchNode] = []
    dispatch_id = 0

    # Build a dispatch_func_idx → command_buffer_idx mapping
    cb_map: dict[int, int] = {}
    for cb_idx, cb in enumerate(trace_data.get("command_buffers", [])):
        for d in cb.get("dispatches", []):
            cb_map[d["index"]] = cb_idx

    for event in trace_data.get("events", []):
        if event.get("type") != "dispatch":
            continue

        buffers = []
        for buf_index, buf_addr in event.get("buffers_bound", {}).items():
            buffers.append(BufferBinding(
                buffer_addr=buf_addr,
                buffer_index=int(buf_index),
                access_mode=AccessMode.UNKNOWN,
            ))

        node = DispatchNode(
            dispatch_id=dispatch_id,
            func_idx=event["index"],
            kernel=event.get("kernel", "unknown"),
            buffers=buffers,
            threadgroups=event.get("threadgroups"),
            threads_per_threadgroup=event.get("threads_per_threadgroup"),
            command_buffer_idx=cb_map.get(event["index"], -1),
        )
        nodes.append(node)
        dispatch_id += 1

    num_cbs = len(trace_data.get("command_buffers", []))
    return nodes, num_cbs


# ---------------------------------------------------------------------------
# Dependency graph construction
# ---------------------------------------------------------------------------

def build_dependency_graph(
    nodes: list[DispatchNode],
    conservative: bool = True,
) -> DependencyGraph:
    """Build a dependency DAG from buffer hazard analysis.

    In conservative mode (default), all shared buffers create SHARED edges
    since we cannot determine read vs write access from the trace format.

    When access modes are available (future), creates RAW/WAW/WAR edges.

    Args:
        nodes: Dispatches in trace order.
        conservative: If True, treat all buffer accesses as read-write.
    """
    graph = DependencyGraph(nodes=nodes)

    if conservative:
        return _build_conservative(graph, nodes)
    else:
        return _build_hazard_based(graph, nodes)


def _build_conservative(
    graph: DependencyGraph,
    nodes: list[DispatchNode],
) -> DependencyGraph:
    """Conservative mode: any shared buffer between dispatches = dependency.

    Only creates edges to the most recent user of each buffer (not all
    previous users), keeping the graph sparse while preserving ordering.
    """
    # For each buffer address, track the last dispatch that used it
    last_user: dict[int, int] = {}  # buffer_addr → dispatch_id

    # Track which edges we've already added (source, target) to avoid dupes
    seen_edges: set[tuple[int, int]] = set()

    for node in nodes:
        for binding in node.buffers:
            addr = binding.buffer_addr
            if addr in last_user:
                src = last_user[addr]
                tgt = node.dispatch_id
                if src != tgt and (src, tgt) not in seen_edges:
                    graph.add_edge(DependencyEdge(
                        source_id=src,
                        target_id=tgt,
                        dep_type=DepType.SHARED,
                        buffer_addr=addr,
                    ))
                    seen_edges.add((src, tgt))
            last_user[addr] = node.dispatch_id

    return graph


def _build_hazard_based(
    graph: DependencyGraph,
    nodes: list[DispatchNode],
) -> DependencyGraph:
    """Hazard-based mode: uses buffer access modes for precise dependencies.

    For each buffer, tracks:
      - last_writer: most recent dispatch that wrote to this buffer
      - readers_since_write: dispatches that read since the last write
    """
    last_writer: dict[int, int] = {}  # buffer_addr → dispatch_id
    readers_since_write: dict[int, set[int]] = defaultdict(set)
    seen_edges: set[tuple[int, int, str]] = set()

    def _add(src: int, tgt: int, dtype: DepType, addr: int) -> None:
        key = (src, tgt, dtype.value)
        if key not in seen_edges:
            graph.add_edge(DependencyEdge(src, tgt, dtype, addr))
            seen_edges.add(key)

    for node in nodes:
        for binding in node.buffers:
            addr = binding.buffer_addr
            mode = binding.access_mode

            if mode in (AccessMode.UNKNOWN, AccessMode.READ_WRITE):
                # Treat as both read and write
                if addr in last_writer:
                    _add(last_writer[addr], node.dispatch_id, DepType.RAW, addr)
                for reader_id in readers_since_write.get(addr, set()):
                    if reader_id != node.dispatch_id:
                        _add(reader_id, node.dispatch_id, DepType.WAR, addr)
                if addr in last_writer:
                    _add(last_writer[addr], node.dispatch_id, DepType.WAW, addr)
                last_writer[addr] = node.dispatch_id
                readers_since_write[addr] = set()

            elif mode == AccessMode.WRITE:
                # WAR: this write depends on previous readers
                for reader_id in readers_since_write.get(addr, set()):
                    if reader_id != node.dispatch_id:
                        _add(reader_id, node.dispatch_id, DepType.WAR, addr)
                # WAW: this write depends on previous write
                if addr in last_writer:
                    _add(last_writer[addr], node.dispatch_id, DepType.WAW, addr)
                last_writer[addr] = node.dispatch_id
                readers_since_write[addr] = set()

            elif mode == AccessMode.READ:
                # RAW: this read depends on previous writer
                if addr in last_writer:
                    _add(last_writer[addr], node.dispatch_id, DepType.RAW, addr)
                readers_since_write[addr].add(node.dispatch_id)

    return graph


# ---------------------------------------------------------------------------
# Transitive reduction
# ---------------------------------------------------------------------------

def transitive_reduction(graph: DependencyGraph) -> DependencyGraph:
    """Remove edges implied by transitivity to simplify the graph.

    An edge A→C is redundant if there exists a path A→B→...→C.
    Uses DFS-based reachability from each node.
    """
    # Build adjacency set for fast lookup
    adj: dict[int, set[int]] = defaultdict(set)
    for edge in graph.edges:
        adj[edge.source_id].add(edge.target_id)

    # For each node, find transitive closure via DFS
    redundant: set[tuple[int, int]] = set()

    for src in adj:
        for direct_target in list(adj[src]):
            # Check if direct_target is reachable from src via other paths
            # DFS from src, excluding the direct edge src→direct_target
            visited: set[int] = set()
            stack = []
            for neighbor in adj[src]:
                if neighbor != direct_target:
                    stack.append(neighbor)

            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                if node == direct_target:
                    redundant.add((src, direct_target))
                    break
                for neighbor in adj.get(node, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)

    if redundant:
        log.info(
            "Transitive reduction removed %d/%d edges",
            len(redundant), len(graph.edges),
        )

    # Build new graph without redundant edges
    reduced = DependencyGraph(nodes=graph.nodes)
    for edge in graph.edges:
        if (edge.source_id, edge.target_id) not in redundant:
            reduced.add_edge(edge)

    return reduced


# ---------------------------------------------------------------------------
# DAG validation
# ---------------------------------------------------------------------------

def validate_dag(graph: DependencyGraph) -> bool:
    """Check that the graph has no cycles (is a valid DAG)."""
    # Kahn's algorithm
    in_degree: dict[int, int] = defaultdict(int)
    node_ids = {n.dispatch_id for n in graph.nodes}
    for nid in node_ids:
        in_degree[nid] = 0
    for edge in graph.edges:
        in_degree[edge.target_id] += 1

    queue = [nid for nid in node_ids if in_degree[nid] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target, _edge in graph.successors.get(node, []):
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)

    is_dag = visited == len(node_ids)
    if not is_dag:
        log.warning("Graph contains cycles! visited=%d, total=%d", visited, len(node_ids))
    return is_dag


# ---------------------------------------------------------------------------
# DOT output
# ---------------------------------------------------------------------------

_DEP_COLORS = {
    DepType.RAW: "red",
    DepType.WAW: "orange",
    DepType.WAR: "blue",
    DepType.SHARED: "gray40",
}


def format_dot(
    graph: DependencyGraph,
    cluster_by_cb: bool = True,
) -> str:
    """Format the dependency graph as Graphviz DOT."""
    lines = [
        "digraph gpu_deps {",
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fillcolor=lightyellow, fontsize=10];',
        '  edge [fontsize=8];',
        "",
    ]

    if cluster_by_cb:
        # Group nodes by command buffer
        cb_groups: dict[int, list[DispatchNode]] = defaultdict(list)
        ungrouped: list[DispatchNode] = []
        for node in graph.nodes:
            if node.command_buffer_idx >= 0:
                cb_groups[node.command_buffer_idx].append(node)
            else:
                ungrouped.append(node)

        for cb_idx in sorted(cb_groups.keys()):
            lines.append(f"  subgraph cluster_cb{cb_idx} {{")
            lines.append(f'    label="Command Buffer #{cb_idx}";')
            lines.append('    style=dashed; color=gray60;')
            for node in cb_groups[cb_idx]:
                lines.append(f"    {_dot_node(node)}")
            lines.append("  }")
            lines.append("")

        for node in ungrouped:
            lines.append(f"  {_dot_node(node)}")
    else:
        for node in graph.nodes:
            lines.append(f"  {_dot_node(node)}")

    lines.append("")

    # Edges
    for edge in graph.edges:
        color = _DEP_COLORS.get(edge.dep_type, "black")
        label = edge.dep_type.value
        lines.append(
            f"  D{edge.source_id} -> D{edge.target_id} "
            f'[color={color}, label="{label}", '
            f'tooltip="buf 0x{edge.buffer_addr:x}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def _dot_node(node: DispatchNode) -> str:
    """Format a single node for DOT."""
    # Truncate long kernel names for readability
    kernel = node.kernel
    if len(kernel) > 40:
        kernel = kernel[:37] + "..."

    tg_str = ""
    if node.threadgroups:
        tg = "x".join(str(x) for x in node.threadgroups)
        tg_str = f"\\ntg={tg}"
        if node.threads_per_threadgroup:
            tpt = "x".join(str(x) for x in node.threads_per_threadgroup)
            tg_str += f" tpt={tpt}"

    bufs = len(node.buffers)
    label = f"D{node.dispatch_id}: {kernel}{tg_str}\\nbufs={bufs}"
    return f'D{node.dispatch_id} [label="{label}"];'


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def format_json(graph: DependencyGraph) -> dict[str, Any]:
    """Format the dependency graph as a JSON-serializable dict."""
    nodes = []
    for node in graph.nodes:
        n: dict[str, Any] = {
            "id": node.dispatch_id,
            "func_idx": node.func_idx,
            "kernel": node.kernel,
            "buffers": [
                {
                    "addr": f"0x{b.buffer_addr:x}",
                    "index": b.buffer_index,
                    "access": b.access_mode.value,
                }
                for b in node.buffers
            ],
            "command_buffer": node.command_buffer_idx,
        }
        if node.threadgroups:
            n["threadgroups"] = list(node.threadgroups)
        if node.threads_per_threadgroup:
            n["threads_per_threadgroup"] = list(node.threads_per_threadgroup)
        nodes.append(n)

    edges = [
        {
            "source": e.source_id,
            "target": e.target_id,
            "type": e.dep_type.value,
            "buffer": f"0x{e.buffer_addr:x}",
        }
        for e in graph.edges
    ]

    edge_types: dict[str, int] = defaultdict(int)
    for e in graph.edges:
        edge_types[e.dep_type.value] += 1

    # Compute critical path length (longest path in DAG)
    critical_path = _compute_critical_path_length(graph)

    # Find isolated nodes (no incoming or outgoing edges)
    connected = set()
    for e in graph.edges:
        connected.add(e.source_id)
        connected.add(e.target_id)
    isolated = [n.dispatch_id for n in graph.nodes if n.dispatch_id not in connected]

    summary: dict[str, Any] = {
        "total_dispatches": len(graph.nodes),
        "total_edges": len(graph.edges),
        "edge_types": dict(edge_types),
        "critical_path_length": critical_path,
        "isolated_nodes": len(isolated),
        "is_dag": validate_dag(graph),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": summary,
    }


def _compute_critical_path_length(graph: DependencyGraph) -> int:
    """Compute the longest path in the DAG (critical path length)."""
    if not graph.nodes:
        return 0

    # Topological sort via Kahn's
    in_degree: dict[int, int] = {n.dispatch_id: 0 for n in graph.nodes}
    for edge in graph.edges:
        in_degree[edge.target_id] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    topo_order: list[int] = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for target, _edge in graph.successors.get(node, []):
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)

    # Longest path via DP on topo order
    dist: dict[int, int] = {n.dispatch_id: 0 for n in graph.nodes}
    for node_id in topo_order:
        for target, _edge in graph.successors.get(node_id, []):
            if dist[target] < dist[node_id] + 1:
                dist[target] = dist[node_id] + 1

    return max(dist.values()) if dist else 0


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def print_summary(graph: DependencyGraph, num_cbs: int) -> None:
    """Print a human-readable summary of the dependency graph."""
    data = format_json(graph)
    s = data["summary"]

    print(f"\n=== Dependency Graph Summary ===")
    print(f"Dispatches:          {s['total_dispatches']}")
    print(f"Dependency edges:    {s['total_edges']}")
    print(f"Edge types:          {s['edge_types']}")
    print(f"Critical path:       {s['critical_path_length']} dispatches")
    print(f"Isolated nodes:      {s['isolated_nodes']}")
    print(f"Is DAG:              {s['is_dag']}")
    print(f"Command buffers:     {num_cbs}")

    # Density: edges / max possible edges
    n = s["total_dispatches"]
    max_edges = n * (n - 1) // 2 if n > 1 else 1
    density = s["total_edges"] / max_edges if max_edges > 0 else 0
    print(f"Graph density:       {density:.4f} ({s['total_edges']}/{max_edges})")

    # Per-kernel statistics
    kernel_dispatch_count: dict[str, int] = defaultdict(int)
    kernel_edge_in: dict[str, int] = defaultdict(int)
    kernel_edge_out: dict[str, int] = defaultdict(int)

    node_map = {n.dispatch_id: n for n in graph.nodes}
    for node in graph.nodes:
        kernel_dispatch_count[node.kernel] += 1
    for edge in graph.edges:
        src_kernel = node_map[edge.source_id].kernel
        tgt_kernel = node_map[edge.target_id].kernel
        kernel_edge_out[src_kernel] += 1
        kernel_edge_in[tgt_kernel] += 1

    print(f"\nPer-kernel breakdown:")
    print(f"  {'Kernel':<45s} {'Count':>6s} {'In':>6s} {'Out':>6s}")
    print(f"  {'-' * 45} {'-' * 6} {'-' * 6} {'-' * 6}")
    for kernel in sorted(kernel_dispatch_count, key=lambda k: -kernel_dispatch_count[k]):
        cnt = kernel_dispatch_count[kernel]
        ein = kernel_edge_in.get(kernel, 0)
        eout = kernel_edge_out.get(kernel, 0)
        name = kernel if len(kernel) <= 45 else kernel[:42] + "..."
        print(f"  {name:<45s} {cnt:>6d} {ein:>6d} {eout:>6d}")

    # Node degree distribution
    in_deg: dict[int, int] = defaultdict(int)
    out_deg: dict[int, int] = defaultdict(int)
    for edge in graph.edges:
        out_deg[edge.source_id] += 1
        in_deg[edge.target_id] += 1

    if graph.nodes:
        max_in = max((in_deg.get(n.dispatch_id, 0) for n in graph.nodes), default=0)
        max_out = max((out_deg.get(n.dispatch_id, 0) for n in graph.nodes), default=0)
        avg_in = sum(in_deg.values()) / len(graph.nodes)
        avg_out = sum(out_deg.values()) / len(graph.nodes)
        print(f"\nDegree stats:")
        print(f"  Max in-degree:  {max_in}")
        print(f"  Max out-degree: {max_out}")
        print(f"  Avg in-degree:  {avg_in:.1f}")
        print(f"  Avg out-degree: {avg_out:.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build dependency graph from .gputrace buffer hazards",
    )
    p.add_argument(
        "trace_path",
        nargs="?",
        default="/tmp/baspacho_ffi.gputrace",
        help="Path to .gputrace file",
    )
    p.add_argument(
        "-f", "--format",
        choices=["dot", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    p.add_argument(
        "-o", "--output",
        help="Output path (without extension for 'both' format)",
    )
    p.add_argument(
        "--conservative",
        action="store_true",
        default=True,
        help="Treat unknown access modes as read-write (default: True)",
    )
    p.add_argument(
        "--no-reduce",
        action="store_true",
        help="Skip transitive reduction",
    )
    p.add_argument(
        "--no-cluster",
        action="store_true",
        help="Don't group dispatches by command buffer in DOT output",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary statistics only, no graph output",
    )
    p.add_argument(
        "--filter-kernel",
        help="Only include dispatches matching this glob pattern",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import and run the timeline reader
    read_gputrace = _import_read_gputrace()
    log.info("Reading trace: %s", args.trace_path)
    trace_data = read_gputrace(args.trace_path)

    if trace_data is None:
        log.error("Failed to read trace file")
        sys.exit(1)

    # Extract dispatches
    nodes, num_cbs = extract_dispatches(trace_data)
    log.info("Extracted %d dispatches from %d command buffers", len(nodes), num_cbs)

    if not nodes:
        log.warning("No dispatches found in trace")
        sys.exit(0)

    # Apply kernel filter
    if args.filter_kernel:
        pattern = args.filter_kernel
        filtered = [n for n in nodes if fnmatch.fnmatch(n.kernel, pattern)]
        log.info(
            "Kernel filter '%s': %d/%d dispatches",
            pattern, len(filtered), len(nodes),
        )
        # Re-index filtered nodes
        id_map = {n.dispatch_id: i for i, n in enumerate(filtered)}
        for i, n in enumerate(filtered):
            n.dispatch_id = i
        for n in filtered:
            # Remap buffer bindings stay the same
            pass
        nodes = filtered

    # Build dependency graph
    graph = build_dependency_graph(nodes, conservative=args.conservative)
    log.info("Built graph: %d edges", len(graph.edges))

    # Validate DAG
    assert validate_dag(graph), "Dependency graph contains cycles!"

    # Transitive reduction
    if not args.no_reduce and graph.edges:
        graph = transitive_reduction(graph)
        log.info("After reduction: %d edges", len(graph.edges))

    # Summary
    print_summary(graph, num_cbs)

    if args.summary_only:
        return

    # Determine output paths
    if args.output:
        base = Path(args.output)
    else:
        base = Path(args.trace_path).with_suffix("")

    # Output DOT
    if args.format in ("dot", "both"):
        dot_path = base.with_suffix(".dot") if args.format == "both" else base
        if args.format == "dot" and not str(dot_path).endswith(".dot"):
            dot_path = dot_path.with_suffix(".dot")
        dot_content = format_dot(graph, cluster_by_cb=not args.no_cluster)
        dot_path.write_text(dot_content)
        log.info("DOT written to: %s", dot_path)
        log.info("Render with: dot -Tsvg %s -o %s", dot_path, dot_path.with_suffix(".svg"))

    # Output JSON
    if args.format in ("json", "both"):
        json_path = base.with_suffix(".json") if args.format == "both" else base
        if args.format == "json" and not str(json_path).endswith(".json"):
            json_path = json_path.with_suffix(".json")
        json_data = format_json(graph)
        json_path.write_text(json.dumps(json_data, indent=2))
        log.info("JSON written to: %s", json_path)


if __name__ == "__main__":
    main()
