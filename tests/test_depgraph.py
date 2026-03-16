"""Tests for gputrace_depgraph core graph logic.

These tests exercise the dependency graph construction, transitive reduction,
and output formatting without requiring Apple frameworks or .gputrace files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add tools dir to import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from gputrace_depgraph import (
    AccessMode,
    BufferBinding,
    DepType,
    DependencyGraph,
    DispatchNode,
    build_dependency_graph,
    extract_dispatches,
    format_dot,
    format_json,
    transitive_reduction,
    validate_dag,
    _compute_critical_path_length,
)


def _make_node(
    dispatch_id: int,
    kernel: str,
    buffers: list[tuple[int, int, AccessMode]] | None = None,
    cb_idx: int = 0,
    threadgroups: tuple[int, ...] | None = None,
) -> DispatchNode:
    """Helper to create a DispatchNode."""
    buf_list = []
    if buffers:
        for addr, idx, mode in buffers:
            buf_list.append(BufferBinding(addr, idx, mode))
    return DispatchNode(
        dispatch_id=dispatch_id,
        func_idx=dispatch_id * 10,
        kernel=kernel,
        buffers=buf_list,
        threadgroups=threadgroups,
        command_buffer_idx=cb_idx,
    )


class TestConservativeGraph:
    """Test conservative mode (SHARED edges for any shared buffer)."""

    def test_no_shared_buffers_no_edges(self):
        """Dispatches with disjoint buffers have no dependencies."""
        nodes = [
            _make_node(0, "kernelA", [(0x100, 0, AccessMode.UNKNOWN)]),
            _make_node(1, "kernelB", [(0x200, 0, AccessMode.UNKNOWN)]),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        assert len(graph.edges) == 0

    def test_shared_buffer_creates_edge(self):
        """Two dispatches sharing a buffer get a SHARED edge."""
        nodes = [
            _make_node(0, "kernelA", [(0x100, 0, AccessMode.UNKNOWN)]),
            _make_node(1, "kernelB", [(0x100, 0, AccessMode.UNKNOWN)]),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        assert len(graph.edges) == 1
        assert graph.edges[0].dep_type == DepType.SHARED
        assert graph.edges[0].source_id == 0
        assert graph.edges[0].target_id == 1
        assert graph.edges[0].buffer_addr == 0x100

    def test_chain_of_three(self):
        """A→B→C chain where each pair shares a buffer."""
        nodes = [
            _make_node(0, "A", [(0x100, 0, AccessMode.UNKNOWN)]),
            _make_node(1, "B", [(0x100, 0, AccessMode.UNKNOWN), (0x200, 1, AccessMode.UNKNOWN)]),
            _make_node(2, "C", [(0x200, 0, AccessMode.UNKNOWN)]),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        # 0→1 (via 0x100), 1→2 (via 0x200)
        assert len(graph.edges) == 2

    def test_no_duplicate_edges(self):
        """Multiple shared buffers between same pair don't create dupe edges."""
        nodes = [
            _make_node(0, "A", [
                (0x100, 0, AccessMode.UNKNOWN),
                (0x200, 1, AccessMode.UNKNOWN),
            ]),
            _make_node(1, "B", [
                (0x100, 0, AccessMode.UNKNOWN),
                (0x200, 1, AccessMode.UNKNOWN),
            ]),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        # Only one edge 0→1 (deduped), but the buffer_addr will be whichever was first
        assert len(graph.edges) == 1

    def test_last_user_only(self):
        """Conservative mode only tracks the last user, keeping graph sparse."""
        # A uses buf 0x100, B uses buf 0x100, C uses buf 0x100
        # Expected: A→B (via 0x100), B→C (via 0x100), NOT A→C
        nodes = [
            _make_node(0, "A", [(0x100, 0, AccessMode.UNKNOWN)]),
            _make_node(1, "B", [(0x100, 0, AccessMode.UNKNOWN)]),
            _make_node(2, "C", [(0x100, 0, AccessMode.UNKNOWN)]),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        assert len(graph.edges) == 2
        edge_pairs = {(e.source_id, e.target_id) for e in graph.edges}
        assert (0, 1) in edge_pairs
        assert (1, 2) in edge_pairs
        assert (0, 2) not in edge_pairs

    def test_empty_trace(self):
        """Empty node list produces empty graph."""
        graph = build_dependency_graph([], conservative=True)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0


class TestHazardBasedGraph:
    """Test hazard-based mode with known access modes."""

    def test_raw_dependency(self):
        """Read after write creates RAW edge."""
        nodes = [
            _make_node(0, "writer", [(0x100, 0, AccessMode.WRITE)]),
            _make_node(1, "reader", [(0x100, 0, AccessMode.READ)]),
        ]
        graph = build_dependency_graph(nodes, conservative=False)
        assert len(graph.edges) == 1
        assert graph.edges[0].dep_type == DepType.RAW

    def test_war_dependency(self):
        """Write after read creates WAR edge."""
        nodes = [
            _make_node(0, "reader", [(0x100, 0, AccessMode.READ)]),
            _make_node(1, "writer", [(0x100, 0, AccessMode.WRITE)]),
        ]
        graph = build_dependency_graph(nodes, conservative=False)
        assert len(graph.edges) == 1
        assert graph.edges[0].dep_type == DepType.WAR

    def test_waw_dependency(self):
        """Write after write creates WAW edge."""
        nodes = [
            _make_node(0, "writer1", [(0x100, 0, AccessMode.WRITE)]),
            _make_node(1, "writer2", [(0x100, 0, AccessMode.WRITE)]),
        ]
        graph = build_dependency_graph(nodes, conservative=False)
        assert len(graph.edges) == 1
        assert graph.edges[0].dep_type == DepType.WAW

    def test_multiple_readers_no_edges(self):
        """Multiple reads with no prior write create no edges."""
        nodes = [
            _make_node(0, "reader1", [(0x100, 0, AccessMode.READ)]),
            _make_node(1, "reader2", [(0x100, 0, AccessMode.READ)]),
        ]
        graph = build_dependency_graph(nodes, conservative=False)
        assert len(graph.edges) == 0

    def test_write_read_read_write_pattern(self):
        """W→R,R→W pattern: RAW for reads, WAR+WAW for second write."""
        nodes = [
            _make_node(0, "W1", [(0x100, 0, AccessMode.WRITE)]),
            _make_node(1, "R1", [(0x100, 0, AccessMode.READ)]),
            _make_node(2, "R2", [(0x100, 0, AccessMode.READ)]),
            _make_node(3, "W2", [(0x100, 0, AccessMode.WRITE)]),
        ]
        graph = build_dependency_graph(nodes, conservative=False)
        edge_types = {(e.source_id, e.target_id): e.dep_type for e in graph.edges}
        # W1→R1 (RAW), W1→R2 (RAW), R1→W2 (WAR), R2→W2 (WAR), W1→W2 (WAW)
        assert edge_types.get((0, 1)) == DepType.RAW
        assert edge_types.get((0, 2)) == DepType.RAW
        assert edge_types.get((1, 3)) == DepType.WAR
        assert edge_types.get((2, 3)) == DepType.WAR
        assert edge_types.get((0, 3)) == DepType.WAW


class TestTransitiveReduction:
    """Test transitive reduction removes redundant edges."""

    def test_simple_reduction(self):
        """A→B→C with A→C: remove A→C."""
        nodes = [
            _make_node(0, "A"),
            _make_node(1, "B"),
            _make_node(2, "C"),
        ]
        graph = DependencyGraph(nodes=nodes)
        from gputrace_depgraph import DependencyEdge
        graph.add_edge(DependencyEdge(0, 1, DepType.SHARED, 0x100))
        graph.add_edge(DependencyEdge(1, 2, DepType.SHARED, 0x200))
        graph.add_edge(DependencyEdge(0, 2, DepType.SHARED, 0x100))

        reduced = transitive_reduction(graph)
        edge_pairs = {(e.source_id, e.target_id) for e in reduced.edges}
        assert (0, 1) in edge_pairs
        assert (1, 2) in edge_pairs
        assert (0, 2) not in edge_pairs

    def test_no_reduction_needed(self):
        """Linear chain has no redundant edges."""
        nodes = [_make_node(i, f"N{i}") for i in range(3)]
        graph = DependencyGraph(nodes=nodes)
        from gputrace_depgraph import DependencyEdge
        graph.add_edge(DependencyEdge(0, 1, DepType.SHARED, 0x100))
        graph.add_edge(DependencyEdge(1, 2, DepType.SHARED, 0x200))

        reduced = transitive_reduction(graph)
        assert len(reduced.edges) == 2


class TestValidation:
    """Test DAG validation."""

    def test_valid_dag(self):
        nodes = [_make_node(i, f"N{i}") for i in range(3)]
        graph = DependencyGraph(nodes=nodes)
        from gputrace_depgraph import DependencyEdge
        graph.add_edge(DependencyEdge(0, 1, DepType.SHARED, 0x100))
        graph.add_edge(DependencyEdge(1, 2, DepType.SHARED, 0x200))
        assert validate_dag(graph) is True

    def test_empty_dag(self):
        graph = DependencyGraph(nodes=[_make_node(0, "A")])
        assert validate_dag(graph) is True


class TestCriticalPath:
    """Test critical path computation."""

    def test_linear_chain(self):
        nodes = [_make_node(i, f"N{i}") for i in range(4)]
        graph = DependencyGraph(nodes=nodes)
        from gputrace_depgraph import DependencyEdge
        for i in range(3):
            graph.add_edge(DependencyEdge(i, i + 1, DepType.SHARED, 0x100))
        assert _compute_critical_path_length(graph) == 3

    def test_parallel_paths(self):
        """Two parallel paths: 0→1→3 and 0→2→3."""
        nodes = [_make_node(i, f"N{i}") for i in range(4)]
        graph = DependencyGraph(nodes=nodes)
        from gputrace_depgraph import DependencyEdge
        graph.add_edge(DependencyEdge(0, 1, DepType.SHARED, 0x100))
        graph.add_edge(DependencyEdge(0, 2, DepType.SHARED, 0x200))
        graph.add_edge(DependencyEdge(1, 3, DepType.SHARED, 0x100))
        graph.add_edge(DependencyEdge(2, 3, DepType.SHARED, 0x200))
        assert _compute_critical_path_length(graph) == 2


class TestDotOutput:
    """Test DOT formatting."""

    def test_dot_basic(self):
        nodes = [
            _make_node(0, "kernelA", [(0x100, 0, AccessMode.UNKNOWN)], cb_idx=0),
            _make_node(1, "kernelB", [(0x100, 0, AccessMode.UNKNOWN)], cb_idx=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        dot = format_dot(graph, cluster_by_cb=True)
        assert "digraph gpu_deps" in dot
        assert "D0" in dot
        assert "D1" in dot
        assert "D0 -> D1" in dot
        assert "cluster_cb0" in dot

    def test_dot_no_cluster(self):
        nodes = [
            _make_node(0, "kernelA", [(0x100, 0, AccessMode.UNKNOWN)]),
            _make_node(1, "kernelB", [(0x100, 0, AccessMode.UNKNOWN)]),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        dot = format_dot(graph, cluster_by_cb=False)
        assert "cluster" not in dot


class TestJsonOutput:
    """Test JSON formatting."""

    def test_json_structure(self):
        nodes = [
            _make_node(0, "kernelA", [(0x100, 0, AccessMode.UNKNOWN)]),
            _make_node(1, "kernelB", [(0x100, 0, AccessMode.UNKNOWN)]),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        data = format_json(graph)

        assert "nodes" in data
        assert "edges" in data
        assert "summary" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["summary"]["total_dispatches"] == 2
        assert data["summary"]["total_edges"] == 1
        assert data["summary"]["is_dag"] is True

        # Verify JSON serializable
        json.dumps(data)

    def test_json_buffer_format(self):
        nodes = [
            _make_node(0, "k", [(0xDEAD, 5, AccessMode.UNKNOWN)]),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        data = format_json(graph)
        buf = data["nodes"][0]["buffers"][0]
        assert buf["addr"] == "0xdead"
        assert buf["index"] == 5


class TestExtractDispatches:
    """Test extraction from timeline data format."""

    def test_basic_extraction(self):
        trace_data = {
            "events": [
                {"type": "set_pipeline", "kernel": "k1", "index": 0},
                {
                    "type": "dispatch",
                    "kernel": "k1",
                    "index": 5,
                    "buffers_bound": {0: 0x100, 1: 0x200},
                    "threadgroups": (4, 1, 1),
                    "threads_per_threadgroup": (256, 1, 1),
                },
                {
                    "type": "dispatch",
                    "kernel": "k2",
                    "index": 10,
                    "buffers_bound": {0: 0x200},
                },
            ],
            "command_buffers": [
                {
                    "func_idx": 15,
                    "dispatches": [
                        {"index": 5, "kernel": "k1"},
                        {"index": 10, "kernel": "k2"},
                    ],
                },
            ],
        }

        nodes, num_cbs = extract_dispatches(trace_data)
        assert len(nodes) == 2
        assert num_cbs == 1
        assert nodes[0].kernel == "k1"
        assert nodes[0].command_buffer_idx == 0
        assert len(nodes[0].buffers) == 2
        assert nodes[0].threadgroups == (4, 1, 1)
        assert nodes[1].kernel == "k2"
        assert nodes[1].command_buffer_idx == 0
        assert len(nodes[1].buffers) == 1

    def test_empty_trace(self):
        nodes, num_cbs = extract_dispatches({"events": [], "command_buffers": []})
        assert len(nodes) == 0
        assert num_cbs == 0
