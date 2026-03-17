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

import os
import sys

_SHARED_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"


def _ensure_dyld_framework_path() -> None:
    """Re-exec with DYLD_FRAMEWORK_PATH if not set.

    dyld reads this variable at process startup to resolve @rpath references,
    so it must be set before any GPU framework is loaded. When missing, we
    set it and os.execv() to restart the process.
    """
    if os.environ.get("DYLD_FRAMEWORK_PATH") != _SHARED_FW:
        os.environ["DYLD_FRAMEWORK_PATH"] = _SHARED_FW
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import fnmatch
import json
import logging
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
    encoder_idx: int = -1


@dataclass
class BarrierNode:
    """A memory barrier synchronization point within an encoder.

    Barriers enforce ordering: all dispatches before the barrier in the same
    encoder must complete before any dispatch after it can begin.
    """
    barrier_id: int
    scope: str              # "buffers" or "resources"
    encoder_idx: int = -1
    command_buffer_idx: int = -1
    after_dispatch_id: int = -1  # last dispatch before this barrier


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
) -> tuple[list[DispatchNode], list[BarrierNode], int, int]:
    """Convert timeline events into DispatchNode and BarrierNode lists.

    Returns (dispatch_nodes, barrier_nodes, num_command_buffers, num_encoders).

    Barriers are synchronization points: all dispatches before a barrier in the
    same encoder must complete before any dispatch after the barrier can start.
    """
    nodes: list[DispatchNode] = []
    barriers: list[BarrierNode] = []
    dispatch_id = 0
    barrier_id = 0

    # Build a dispatch_func_idx → command_buffer_idx mapping
    cb_map: dict[int, int] = {}
    for cb_idx, cb in enumerate(trace_data.get("command_buffers", [])):
        for d in cb.get("dispatches", []):
            cb_map[d["index"]] = cb_idx

    # Track the last dispatch_id per encoder (for barrier placement)
    last_dispatch_in_encoder: dict[int, int] = {}

    for event in trace_data.get("events", []):
        etype = event.get("type")

        if etype == "dispatch":
            buffers = []
            for buf_index, buf_addr in event.get("buffers_bound", {}).items():
                buffers.append(BufferBinding(
                    buffer_addr=buf_addr,
                    buffer_index=int(buf_index),
                    access_mode=AccessMode.UNKNOWN,
                ))

            enc_idx = event.get("encoder_idx", -1)
            node = DispatchNode(
                dispatch_id=dispatch_id,
                func_idx=event["index"],
                kernel=event.get("kernel", "unknown"),
                buffers=buffers,
                threadgroups=event.get("threadgroups"),
                threads_per_threadgroup=event.get("threads_per_threadgroup"),
                command_buffer_idx=cb_map.get(event["index"], -1),
                encoder_idx=enc_idx,
            )
            nodes.append(node)
            last_dispatch_in_encoder[enc_idx] = dispatch_id
            dispatch_id += 1

        elif etype == "barrier":
            enc_idx = event.get("encoder_idx", -1)
            scope = event.get("scope", "buffers")
            after_did = last_dispatch_in_encoder.get(enc_idx, -1)
            barriers.append(BarrierNode(
                barrier_id=barrier_id,
                scope=scope,
                encoder_idx=enc_idx,
                command_buffer_idx=event.get("command_buffer_idx", -1),
                after_dispatch_id=after_did,
            ))
            barrier_id += 1

    num_cbs = len(trace_data.get("command_buffers", []))
    num_encoders = len(trace_data.get("compute_encoders", []))
    return nodes, barriers, num_cbs, num_encoders


# ---------------------------------------------------------------------------
# Dependency graph construction
# ---------------------------------------------------------------------------

def build_dependency_graph(
    nodes: list[DispatchNode],
    conservative: bool = True,
    barriers: list[BarrierNode] | None = None,
) -> DependencyGraph:
    """Build a dependency DAG from buffer hazard analysis.

    In conservative mode (default), all shared buffers create SHARED edges
    since we cannot determine read vs write access from the trace format.

    When access modes are available (future), creates RAW/WAW/WAR edges.

    If barriers are provided, they act as synchronization points within each
    encoder: every dispatch before a barrier must complete before every
    dispatch after it (within the same encoder).

    Args:
        nodes: Dispatches in trace order.
        conservative: If True, treat all buffer accesses as read-write.
        barriers: Optional barrier nodes from extract_dispatches().
    """
    graph = DependencyGraph(nodes=nodes)

    if conservative:
        graph = _build_conservative(graph, nodes)
    else:
        graph = _build_hazard_based(graph, nodes)

    # Apply barrier-induced ordering
    if barriers:
        _apply_barrier_edges(graph, nodes, barriers)

    return graph


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


def _apply_barrier_edges(
    graph: DependencyGraph,
    nodes: list[DispatchNode],
    barriers: list[BarrierNode],
) -> None:
    """Add edges enforced by memory barriers.

    A barrier between dispatch A and dispatch B within the same encoder means
    A must complete before B starts. For each barrier, we find the dispatches
    before and after it in the same encoder, and ensure the last dispatch before
    the barrier has an edge to the first dispatch after it.

    This is more precise than connecting all pre-barrier to all post-barrier:
    buffer-based edges already handle the transitive cases, and adding just
    the "last before → first after" edge per barrier is sufficient since
    transitive reduction will handle the rest.
    """
    # Group dispatches by encoder, preserving dispatch_id order
    enc_dispatches: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        enc_dispatches[node.encoder_idx].append(node.dispatch_id)

    # Track existing edges to avoid duplicates
    existing_edges: set[tuple[int, int]] = set()
    for edge in graph.edges:
        existing_edges.add((edge.source_id, edge.target_id))

    for barrier in barriers:
        enc_idx = barrier.encoder_idx
        dispatches_in_enc = enc_dispatches.get(enc_idx, [])
        if not dispatches_in_enc:
            continue

        # Split dispatches into before and after the barrier.
        # barrier.after_dispatch_id is the last dispatch before the barrier.
        split = barrier.after_dispatch_id
        if split < 0:
            # Barrier before any dispatch in this encoder — no ordering to enforce
            continue

        before = [d for d in dispatches_in_enc if d <= split]
        after = [d for d in dispatches_in_enc if d > split]

        if not before or not after:
            continue

        # Add edge from last dispatch before barrier to first dispatch after.
        # Buffer-based edges + transitive reduction handle the rest.
        last_before = before[-1]
        first_after = after[0]

        if (last_before, first_after) not in existing_edges:
            graph.add_edge(DependencyEdge(
                source_id=last_before,
                target_id=first_after,
                dep_type=DepType.SHARED,
                buffer_addr=0,  # barrier-induced, no specific buffer
            ))
            existing_edges.add((last_before, first_after))


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
    skip_isolated: bool = True,
    barriers: list[BarrierNode] | None = None,
) -> str:
    """Format the dependency graph as Graphviz DOT.

    Args:
        graph: The dependency graph to format.
        cluster_by_cb: Group nodes by command buffer.
        skip_isolated: Omit nodes with no edges (default True for large graphs).
        barriers: Optional barrier nodes to render as diamonds.
    """
    # Determine which nodes have edges
    connected_ids: set[int] | None = None
    if skip_isolated:
        connected_ids = set()
        for edge in graph.edges:
            connected_ids.add(edge.source_id)
            connected_ids.add(edge.target_id)

    def _include(node: DispatchNode) -> bool:
        if connected_ids is None:
            return True
        return node.dispatch_id in connected_ids

    lines = [
        "digraph gpu_deps {",
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fillcolor=lightyellow, fontsize=10];',
        '  edge [fontsize=8];',
        "",
    ]

    if cluster_by_cb:
        cb_groups: dict[int, list[DispatchNode]] = defaultdict(list)
        ungrouped: list[DispatchNode] = []
        for node in graph.nodes:
            if not _include(node):
                continue
            if node.command_buffer_idx >= 0:
                cb_groups[node.command_buffer_idx].append(node)
            else:
                ungrouped.append(node)

        for cb_idx in sorted(cb_groups.keys()):
            nodes_in_cb = cb_groups[cb_idx]
            if not nodes_in_cb:
                continue
            lines.append(f"  subgraph cluster_cb{cb_idx} {{")
            lines.append(f'    label="Command Buffer #{cb_idx}";')
            lines.append('    style=dashed; color=gray60;')
            for node in nodes_in_cb:
                lines.append(f"    {_dot_node(node)}")
            lines.append("  }")
            lines.append("")

        for node in ungrouped:
            lines.append(f"  {_dot_node(node)}")
    else:
        for node in graph.nodes:
            if _include(node):
                lines.append(f"  {_dot_node(node)}")

    # Render barrier nodes as diamonds between dispatches
    if barriers:
        lines.append("  // Barrier nodes")
        for b in barriers:
            if b.after_dispatch_id < 0:
                continue
            scope_label = "B" if b.scope == "buffers" else "R"
            lines.append(
                f'  barrier{b.barrier_id} [shape=diamond, '
                f'style=filled, fillcolor=lightsalmon, '
                f'label="{scope_label}", width=0.4, height=0.4, '
                f'tooltip="barrier #{b.barrier_id} ({b.scope})"];'
            )
        lines.append("")

    lines.append("")

    for edge in graph.edges:
        color = _DEP_COLORS.get(edge.dep_type, "black")
        label = edge.dep_type.value
        if edge.buffer_addr == 0:
            # Barrier-induced edge
            lines.append(
                f"  D{edge.source_id} -> D{edge.target_id} "
                f'[color=red, style=dashed, label="barrier"];'
            )
        else:
            lines.append(
                f"  D{edge.source_id} -> D{edge.target_id} "
                f'[color={color}, label="{label}", '
                f'tooltip="buf 0x{edge.buffer_addr:x}"];'
            )

    # Render barrier edges (barrier diamond between pre/post dispatches)
    if barriers:
        # Group dispatches by encoder for barrier edge rendering
        enc_dispatches: dict[int, list[int]] = defaultdict(list)
        for node in graph.nodes:
            enc_dispatches[node.encoder_idx].append(node.dispatch_id)

        for b in barriers:
            if b.after_dispatch_id < 0:
                continue
            dispatches_in_enc = enc_dispatches.get(b.encoder_idx, [])
            after = [d for d in dispatches_in_enc if d > b.after_dispatch_id]
            if after:
                lines.append(
                    f"  D{b.after_dispatch_id} -> barrier{b.barrier_id} "
                    f"[style=dashed, color=red, arrowhead=none];"
                )
                lines.append(
                    f"  barrier{b.barrier_id} -> D{after[0]} "
                    f"[style=dashed, color=red];"
                )

    lines.append("}")
    return "\n".join(lines)


def format_kernel_dot(graph: DependencyGraph) -> str:
    """Format a kernel-level summary graph as DOT.

    Collapses all dispatches of the same kernel into a single node,
    with edge weights showing how many dispatch-level dependencies exist.
    Much more readable for large traces.
    """
    agg = build_kernel_graph(graph.nodes, graph)
    return format_aggregated_dot(agg, title="kernel_deps")


# ---------------------------------------------------------------------------
# Aggregated graph data model and builders
# ---------------------------------------------------------------------------

@dataclass
class AggregatedNode:
    node_id: str           # "CB0", "E5", "K3"
    label: str             # "CB #0\n65 dispatches, 3 barriers\nlu_gemv(48), lu_solve(17)"
    dispatch_count: int
    kernel_composition: dict[str, int]  # kernel → count
    barrier_count: int = 0

    @property
    def short_composition(self) -> str:
        """Top kernels as compact string."""
        top = sorted(self.kernel_composition.items(), key=lambda x: -x[1])[:4]
        parts = []
        for k, c in top:
            name = k if len(k) <= 25 else k[:22] + "..."
            parts.append(f"{name}({c})")
        if len(self.kernel_composition) > 4:
            parts.append("...")
        return ", ".join(parts)


@dataclass
class AggregatedEdge:
    source_id: str
    target_id: str
    weight: int            # number of dispatch-level edges this represents
    buffer_addrs: set[int] = field(default_factory=set)


@dataclass
class AggregatedGraph:
    nodes: list[AggregatedNode] = field(default_factory=list)
    edges: list[AggregatedEdge] = field(default_factory=list)
    scale: str = "unknown"
    cluster_key: str | None = None  # group nodes by this attribute in DOT
    clusters: dict[str, list[str]] | None = None  # cluster_label → [node_ids]


def build_cb_graph(
    nodes: list[DispatchNode],
    graph: DependencyGraph,
    barriers: list[BarrierNode] | None = None,
) -> AggregatedGraph:
    """Collapse dependency graph to command buffer level.

    One node per command buffer. Edge between CBs if any dispatch in one
    shares a buffer dependency with a dispatch in the other.
    """
    node_map = {n.dispatch_id: n for n in nodes}

    # Count barriers per CB
    cb_barrier_count: dict[int, int] = defaultdict(int)
    for b in (barriers or []):
        cb_barrier_count[b.command_buffer_idx] += 1

    # Group dispatches by CB
    cb_dispatches: dict[int, list[DispatchNode]] = defaultdict(list)
    for n in nodes:
        cb_dispatches[n.command_buffer_idx].append(n)

    # Build aggregated nodes
    agg_nodes: dict[str, AggregatedNode] = {}
    for cb_idx in sorted(cb_dispatches.keys()):
        dispatches = cb_dispatches[cb_idx]
        composition: dict[str, int] = defaultdict(int)
        for d in dispatches:
            composition[d.kernel] += 1
        nid = f"CB{cb_idx}"
        b_count = cb_barrier_count.get(cb_idx, 0)
        agg_nodes[nid] = AggregatedNode(
            node_id=nid,
            label="",  # filled below
            dispatch_count=len(dispatches),
            kernel_composition=dict(composition),
            barrier_count=b_count,
        )
        node = agg_nodes[nid]
        b_str = f", {b_count} barriers" if b_count else ""
        node.label = f"CB #{cb_idx}\\n{node.dispatch_count} dispatches{b_str}\\n{node.short_composition}"

    # Build aggregated edges
    agg_edges: dict[tuple[str, str], AggregatedEdge] = {}
    for edge in graph.edges:
        src_cb = node_map[edge.source_id].command_buffer_idx
        tgt_cb = node_map[edge.target_id].command_buffer_idx
        if src_cb == tgt_cb:
            continue  # skip intra-CB edges
        key = (f"CB{src_cb}", f"CB{tgt_cb}")
        if key not in agg_edges:
            agg_edges[key] = AggregatedEdge(
                source_id=key[0], target_id=key[1], weight=0,
            )
        agg_edges[key].weight += 1
        agg_edges[key].buffer_addrs.add(edge.buffer_addr)

    return AggregatedGraph(
        nodes=list(agg_nodes.values()),
        edges=list(agg_edges.values()),
        scale="cb",
    )


def build_encoder_graph(
    nodes: list[DispatchNode],
    graph: DependencyGraph,
    barriers: list[BarrierNode] | None = None,
) -> AggregatedGraph:
    """Collapse dependency graph to compute encoder level.

    One node per encoder. Clustered by command buffer.
    """
    node_map = {n.dispatch_id: n for n in nodes}

    # Count barriers per encoder
    enc_barrier_count: dict[int, int] = defaultdict(int)
    for b in (barriers or []):
        enc_barrier_count[b.encoder_idx] += 1

    # Group dispatches by encoder
    enc_dispatches: dict[int, list[DispatchNode]] = defaultdict(list)
    for n in nodes:
        enc_dispatches[n.encoder_idx].append(n)

    # Build aggregated nodes
    agg_nodes: dict[str, AggregatedNode] = {}
    enc_to_cb: dict[str, int] = {}
    for enc_idx in sorted(enc_dispatches.keys()):
        dispatches = enc_dispatches[enc_idx]
        composition: dict[str, int] = defaultdict(int)
        for d in dispatches:
            composition[d.kernel] += 1
        nid = f"E{enc_idx}"
        cb_idx = dispatches[0].command_buffer_idx if dispatches else -1
        enc_to_cb[nid] = cb_idx
        b_count = enc_barrier_count.get(enc_idx, 0)
        agg_nodes[nid] = AggregatedNode(
            node_id=nid,
            label="",
            dispatch_count=len(dispatches),
            kernel_composition=dict(composition),
            barrier_count=b_count,
        )
        node = agg_nodes[nid]
        comp = node.short_composition
        b_str = f", {b_count} barriers" if b_count else ""
        node.label = f"Encoder #{enc_idx}\\n{node.dispatch_count} dispatches{b_str}\\n{comp}"

    # Build aggregated edges
    agg_edges: dict[tuple[str, str], AggregatedEdge] = {}
    for edge in graph.edges:
        src_enc = node_map[edge.source_id].encoder_idx
        tgt_enc = node_map[edge.target_id].encoder_idx
        if src_enc == tgt_enc:
            continue
        key = (f"E{src_enc}", f"E{tgt_enc}")
        if key not in agg_edges:
            agg_edges[key] = AggregatedEdge(
                source_id=key[0], target_id=key[1], weight=0,
            )
        agg_edges[key].weight += 1
        agg_edges[key].buffer_addrs.add(edge.buffer_addr)

    # Build clusters by command buffer
    clusters: dict[str, list[str]] = defaultdict(list)
    for nid, cb_idx in enc_to_cb.items():
        clusters[f"CB #{cb_idx}"].append(nid)

    return AggregatedGraph(
        nodes=list(agg_nodes.values()),
        edges=list(agg_edges.values()),
        scale="encoder",
        cluster_key="command_buffer",
        clusters=dict(clusters),
    )


def build_kernel_graph(
    nodes: list[DispatchNode],
    graph: DependencyGraph,
) -> AggregatedGraph:
    """Collapse dependency graph to kernel level.

    One node per unique kernel name. Self-loops (intra-kernel deps) counted
    but not drawn as edges.
    """
    node_map = {n.dispatch_id: n for n in nodes}

    # Aggregate dispatches by kernel
    kernel_dispatches: dict[str, list[DispatchNode]] = defaultdict(list)
    for n in nodes:
        kernel_dispatches[n.kernel].append(n)

    # Self-loop count
    self_deps: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        src_k = node_map[edge.source_id].kernel
        tgt_k = node_map[edge.target_id].kernel
        if src_k == tgt_k:
            self_deps[src_k] += 1

    agg_nodes: dict[str, AggregatedNode] = {}
    kernels = sorted(kernel_dispatches.keys())
    for i, kernel in enumerate(kernels):
        dispatches = kernel_dispatches[kernel]
        nid = f"K{i}"
        name = kernel if len(kernel) <= 40 else kernel[:37] + "..."
        self_count = self_deps.get(kernel, 0)
        self_str = f"\\nself-deps={self_count}" if self_count else ""
        agg_nodes[kernel] = AggregatedNode(
            node_id=nid,
            label=f"{name}\\ndispatches={len(dispatches)}{self_str}",
            dispatch_count=len(dispatches),
            kernel_composition={kernel: len(dispatches)},
        )

    # Map kernel name → node_id for edges
    kernel_to_nid = {k: agg_nodes[k].node_id for k in kernels}

    agg_edges: dict[tuple[str, str], AggregatedEdge] = {}
    for edge in graph.edges:
        src_k = node_map[edge.source_id].kernel
        tgt_k = node_map[edge.target_id].kernel
        if src_k == tgt_k:
            continue
        key = (kernel_to_nid[src_k], kernel_to_nid[tgt_k])
        if key not in agg_edges:
            agg_edges[key] = AggregatedEdge(
                source_id=key[0], target_id=key[1], weight=0,
            )
        agg_edges[key].weight += 1
        agg_edges[key].buffer_addrs.add(edge.buffer_addr)

    return AggregatedGraph(
        nodes=list(agg_nodes.values()),
        edges=list(agg_edges.values()),
        scale="kernel",
    )


def format_aggregated_dot(
    agg: AggregatedGraph,
    title: str = "gpu_deps",
) -> str:
    """Format an aggregated graph as Graphviz DOT.

    Node width proportional to dispatch count. Edge penwidth proportional
    to weight. Clusters if available.
    """
    lines = [
        f"digraph {title} {{",
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fillcolor=lightyellow, fontsize=11];',
        '  edge [fontsize=9];',
        "",
    ]

    # Max dispatch count for sizing
    max_count = max((n.dispatch_count for n in agg.nodes), default=1)

    def _node_line(node: AggregatedNode) -> str:
        # Scale width 1.5..4 based on dispatch count
        width = 1.5 + 2.5 * (node.dispatch_count / max(max_count, 1))
        return (
            f'  {node.node_id} [label="{node.label}", '
            f'width={width:.1f}];'
        )

    if agg.clusters:
        # Emit clustered nodes
        emitted: set[str] = set()
        for i, (cluster_label, node_ids) in enumerate(
            sorted(agg.clusters.items())
        ):
            lines.append(f"  subgraph cluster_{i} {{")
            lines.append(f'    label="{cluster_label}";')
            lines.append('    style=dashed; color=gray60;')
            for node in agg.nodes:
                if node.node_id in node_ids:
                    lines.append(f"  {_node_line(node)}")
                    emitted.add(node.node_id)
            lines.append("  }")
            lines.append("")
        # Emit unclustered nodes
        for node in agg.nodes:
            if node.node_id not in emitted:
                lines.append(_node_line(node))
    else:
        for node in agg.nodes:
            lines.append(_node_line(node))

    lines.append("")

    for edge in sorted(agg.edges, key=lambda e: -e.weight):
        pw = min(1 + edge.weight / 50, 5)
        bufs = len(edge.buffer_addrs)
        lines.append(
            f'  {edge.source_id} -> {edge.target_id} '
            f'[label="{edge.weight}", penwidth={pw:.1f}, '
            f'tooltip="{bufs} shared buffers"];'
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
            "encoder": node.encoder_idx,
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

def print_summary(
    graph: DependencyGraph,
    num_cbs: int,
    num_barriers: int = 0,
) -> None:
    """Print a human-readable summary of the dependency graph."""
    data = format_json(graph)
    s = data["summary"]

    print("\n=== Dependency Graph Summary ===")
    print(f"Dispatches:          {s['total_dispatches']}")
    if num_barriers:
        print(f"Barriers:            {num_barriers}")
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

    print("\nPer-kernel breakdown:")
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
        print("\nDegree stats:")
        print(f"  Max in-degree:  {max_in}")
        print(f"  Max out-degree: {max_out}")
        print(f"  Avg in-degree:  {avg_in:.1f}")
        print(f"  Avg out-degree: {avg_out:.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MAX_UNSCOPED_DISPATCHES = 2000


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
        "--scale",
        choices=["dispatch", "encoder", "kernel", "cb"],
        default="encoder",
        help="Graph scale: cb, encoder, kernel, or dispatch (default: encoder)",
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
    p.add_argument(
        "--filter-cb",
        type=int,
        help="Only include dispatches in this command buffer index",
    )
    p.add_argument(
        "--filter-encoder",
        type=int,
        help="Only include dispatches in this encoder index",
    )
    p.add_argument(
        "--include-isolated",
        action="store_true",
        help="Include isolated nodes (no edges) in DOT output",
    )
    return p.parse_args()


def _apply_filters(
    nodes: list[DispatchNode],
    filter_kernel: str | None = None,
    filter_cb: int | None = None,
    filter_encoder: int | None = None,
) -> list[DispatchNode]:
    """Apply scope filters and re-index dispatch IDs."""
    filtered = nodes

    if filter_cb is not None:
        filtered = [n for n in filtered if n.command_buffer_idx == filter_cb]
        log.info("CB filter %d: %d/%d dispatches", filter_cb, len(filtered), len(nodes))

    if filter_encoder is not None:
        filtered = [n for n in filtered if n.encoder_idx == filter_encoder]
        log.info("Encoder filter %d: %d/%d dispatches", filter_encoder, len(filtered), len(nodes))

    if filter_kernel:
        filtered = [n for n in filtered if fnmatch.fnmatch(n.kernel, filter_kernel)]
        log.info("Kernel filter '%s': %d/%d dispatches", filter_kernel, len(filtered), len(nodes))

    # Re-index dispatch IDs for the filtered subset
    for i, n in enumerate(filtered):
        n.dispatch_id = i

    return filtered


def _scale_suffix(scale: str) -> str:
    """Return filename suffix for the given scale."""
    return {"cb": "_cb", "encoder": "_encoder", "kernel": "_kernel", "dispatch": ""}[scale]


def main() -> None:
    _ensure_dyld_framework_path()
    args = parse_args()

    # Import and run the timeline reader
    read_gputrace = _import_read_gputrace()
    log.info("Reading trace: %s", args.trace_path)
    trace_data = read_gputrace(args.trace_path)

    if trace_data is None:
        log.error("Failed to read trace file")
        sys.exit(1)

    # Extract dispatches and barriers
    nodes, barriers, num_cbs, num_encoders = extract_dispatches(trace_data)
    log.info(
        "Extracted %d dispatches, %d barriers from %d command buffers, %d encoders",
        len(nodes), len(barriers), num_cbs, num_encoders,
    )

    if not nodes:
        log.warning("No dispatches found in trace")
        sys.exit(0)

    # Apply scope filters
    has_filter = (
        args.filter_kernel is not None
        or args.filter_cb is not None
        or args.filter_encoder is not None
    )
    nodes = _apply_filters(
        nodes,
        filter_kernel=args.filter_kernel,
        filter_cb=args.filter_cb,
        filter_encoder=args.filter_encoder,
    )

    if not nodes:
        log.warning("No dispatches after filtering")
        sys.exit(0)

    # Guard against unscoped dispatch-level graphs
    if args.scale == "dispatch" and not has_filter and len(nodes) > _MAX_UNSCOPED_DISPATCHES:
        log.error(
            "Dispatch-level graph has %d nodes (threshold: %d). "
            "Use --filter-cb, --filter-encoder, or --filter-kernel to scope.",
            len(nodes), _MAX_UNSCOPED_DISPATCHES,
        )
        sys.exit(1)

    # Filter barriers to match filtered dispatches
    if has_filter:
        # Keep only barriers in filtered encoders with valid dispatch refs
        filtered_dispatch_ids = {n.dispatch_id for n in nodes}
        filtered_encoders = {n.encoder_idx for n in nodes}
        barriers = [
            b for b in barriers
            if b.encoder_idx in filtered_encoders
        ]

    # Build dependency graph
    graph = build_dependency_graph(
        nodes, conservative=args.conservative, barriers=barriers,
    )
    log.info("Built graph: %d edges", len(graph.edges))

    # Validate DAG
    assert validate_dag(graph), "Dependency graph contains cycles!"

    # Transitive reduction
    if not args.no_reduce and graph.edges:
        graph = transitive_reduction(graph)
        log.info("After reduction: %d edges", len(graph.edges))

    # Summary
    print_summary(graph, num_cbs, num_barriers=len(barriers))

    if args.summary_only:
        return

    # Determine output paths
    if args.output:
        base = Path(args.output)
    else:
        base = Path(args.trace_path).with_suffix("")

    # Build graph at requested scale
    scale = args.scale
    suffix = _scale_suffix(scale)

    if scale == "cb":
        agg = build_cb_graph(nodes, graph, barriers=barriers)
        dot_content = format_aggregated_dot(agg, title="cb_deps")
    elif scale == "encoder":
        agg = build_encoder_graph(nodes, graph, barriers=barriers)
        dot_content = format_aggregated_dot(agg, title="encoder_deps")
    elif scale == "kernel":
        agg = build_kernel_graph(nodes, graph)
        dot_content = format_aggregated_dot(agg, title="kernel_deps")
    elif scale == "dispatch":
        dot_content = format_dot(
            graph,
            cluster_by_cb=not args.no_cluster,
            skip_isolated=not args.include_isolated,
            barriers=barriers if barriers else None,
        )
        agg = None
    else:
        assert False, f"Unknown scale: {scale}"

    # Output DOT
    if args.format in ("dot", "both"):
        dot_path = base.with_suffix(".dot") if args.format == "both" else base
        if args.format == "dot" and not str(dot_path).endswith(".dot"):
            dot_path = dot_path.with_suffix(".dot")
        if suffix:
            dot_path = dot_path.with_stem(dot_path.stem + suffix)

        dot_path.write_text(dot_content)
        log.info("DOT written to: %s (%s scale)", dot_path, scale)

        # Suggest layout engine
        if agg is not None:
            n_nodes = len(agg.nodes)
        else:
            n_nodes = sum(
                1 for n in graph.nodes
                if any(e.source_id == n.dispatch_id or e.target_id == n.dispatch_id
                       for e in graph.edges)
            )

        if n_nodes > 500:
            log.info(
                "Large graph (%d nodes). Use sfdp: sfdp -Tsvg %s -o %s",
                n_nodes, dot_path, dot_path.with_suffix(".svg"),
            )
        else:
            log.info(
                "Render: dot -Tsvg %s -o %s",
                dot_path, dot_path.with_suffix(".svg"),
            )

    # Output JSON
    if args.format in ("json", "both"):
        json_path = base.with_suffix(".json") if args.format == "both" else base
        if args.format == "json" and not str(json_path).endswith(".json"):
            json_path = json_path.with_suffix(".json")
        json_data = format_json(graph)
        json_data["scale"] = scale
        if agg is not None:
            json_data["aggregated"] = {
                "nodes": [
                    {
                        "id": n.node_id,
                        "dispatch_count": n.dispatch_count,
                        "kernel_composition": n.kernel_composition,
                    }
                    for n in agg.nodes
                ],
                "edges": [
                    {
                        "source": e.source_id,
                        "target": e.target_id,
                        "weight": e.weight,
                        "shared_buffers": len(e.buffer_addrs),
                    }
                    for e in agg.edges
                ],
            }
        json_path.write_text(json.dumps(json_data, indent=2))
        log.info("JSON written to: %s", json_path)


if __name__ == "__main__":
    main()
