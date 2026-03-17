"""Tests for gputrace_depgraph core graph logic.

These tests exercise the dependency graph construction, transitive reduction,
output formatting, and multi-scale aggregated graph builders without requiring
Apple frameworks or .gputrace files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add tools dir to import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from gputrace_depgraph import (  # noqa: I001
    AccessMode,
    AggregatedGraph,
    BarrierNode,
    BufferBinding,
    DepType,
    DependencyEdge,
    DependencyGraph,
    DispatchNode,
    build_cb_graph,
    build_dependency_graph,
    build_encoder_graph,
    build_kernel_graph,
    extract_dispatches,
    format_aggregated_dot,
    format_dot,
    format_json,
    format_kernel_dot,
    transitive_reduction,
    validate_dag,
    _apply_filters,
    _compute_critical_path_length,
)


def _make_node(
    dispatch_id: int,
    kernel: str,
    buffers: list[tuple[int, int, AccessMode]] | None = None,
    cb_idx: int = 0,
    encoder_idx: int = 0,
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
        encoder_idx=encoder_idx,
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
        graph.add_edge(DependencyEdge(0, 1, DepType.SHARED, 0x100))
        graph.add_edge(DependencyEdge(1, 2, DepType.SHARED, 0x200))

        reduced = transitive_reduction(graph)
        assert len(reduced.edges) == 2


class TestValidation:
    """Test DAG validation."""

    def test_valid_dag(self):
        nodes = [_make_node(i, f"N{i}") for i in range(3)]
        graph = DependencyGraph(nodes=nodes)
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
        for i in range(3):
            graph.add_edge(DependencyEdge(i, i + 1, DepType.SHARED, 0x100))
        assert _compute_critical_path_length(graph) == 3

    def test_parallel_paths(self):
        """Two parallel paths: 0→1→3 and 0→2→3."""
        nodes = [_make_node(i, f"N{i}") for i in range(4)]
        graph = DependencyGraph(nodes=nodes)
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

    def test_json_includes_encoder(self):
        """JSON output includes encoder_idx for each node."""
        nodes = [
            _make_node(0, "k", [(0x100, 0, AccessMode.UNKNOWN)], encoder_idx=3),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        data = format_json(graph)
        assert data["nodes"][0]["encoder"] == 3


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
                    "encoder_idx": 0,
                },
                {
                    "type": "dispatch",
                    "kernel": "k2",
                    "index": 10,
                    "buffers_bound": {0: 0x200},
                    "encoder_idx": 1,
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
            "compute_encoders": [
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 5}]},
                {"encoder_idx": 1, "command_buffer_idx": 0, "dispatches": [{"index": 10}]},
            ],
        }

        nodes, barriers, num_cbs, num_encoders = extract_dispatches(trace_data)
        assert len(nodes) == 2
        assert len(barriers) == 0
        assert num_cbs == 1
        assert num_encoders == 2
        assert nodes[0].kernel == "k1"
        assert nodes[0].command_buffer_idx == 0
        assert nodes[0].encoder_idx == 0
        assert len(nodes[0].buffers) == 2
        assert nodes[0].threadgroups == (4, 1, 1)
        assert nodes[1].kernel == "k2"
        assert nodes[1].command_buffer_idx == 0
        assert nodes[1].encoder_idx == 1
        assert len(nodes[1].buffers) == 1

    def test_empty_trace(self):
        nodes, barriers, num_cbs, num_encoders = extract_dispatches(
            {"events": [], "command_buffers": [], "compute_encoders": []}
        )
        assert len(nodes) == 0
        assert len(barriers) == 0
        assert num_cbs == 0
        assert num_encoders == 0


# ---------------------------------------------------------------------------
# Aggregated graph tests
# ---------------------------------------------------------------------------

def _make_multi_cb_nodes() -> tuple[list[DispatchNode], DependencyGraph]:
    """Create a test scenario with 2 CBs, 3 encoders, 2 kernels.

    CB0: Encoder0 [D0:kA, D1:kA], Encoder1 [D2:kB]
    CB1: Encoder2 [D3:kA, D4:kB]

    Buffer hazards:
      D0 → D1 (buf 0x100, same encoder)
      D1 → D2 (buf 0x200, cross-encoder, same CB)
      D2 → D3 (buf 0x300, cross-CB)
      D3 → D4 (buf 0x100, same encoder)
    """
    U = AccessMode.UNKNOWN
    nodes = [
        _make_node(0, "kA", [(0x100, 0, U)], cb_idx=0, encoder_idx=0),
        _make_node(1, "kA", [(0x100, 0, U), (0x200, 1, U)], cb_idx=0, encoder_idx=0),
        _make_node(2, "kB", [(0x200, 0, U), (0x300, 1, U)], cb_idx=0, encoder_idx=1),
        _make_node(3, "kA", [(0x300, 0, U), (0x100, 1, U)], cb_idx=1, encoder_idx=2),
        _make_node(4, "kB", [(0x100, 0, U)], cb_idx=1, encoder_idx=2),
    ]
    graph = build_dependency_graph(nodes, conservative=True)
    return nodes, graph


class TestBuildCBGraph:
    """Test command buffer level aggregation."""

    def test_cb_graph_nodes(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_cb_graph(nodes, graph)
        assert agg.scale == "cb"
        assert len(agg.nodes) == 2
        ids = {n.node_id for n in agg.nodes}
        assert ids == {"CB0", "CB1"}

    def test_cb_graph_dispatch_counts(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_cb_graph(nodes, graph)
        by_id = {n.node_id: n for n in agg.nodes}
        assert by_id["CB0"].dispatch_count == 3
        assert by_id["CB1"].dispatch_count == 2

    def test_cb_graph_kernel_composition(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_cb_graph(nodes, graph)
        by_id = {n.node_id: n for n in agg.nodes}
        assert by_id["CB0"].kernel_composition == {"kA": 2, "kB": 1}
        assert by_id["CB1"].kernel_composition == {"kA": 1, "kB": 1}

    def test_cb_graph_cross_cb_edge(self):
        """Only cross-CB edges appear (intra-CB skipped)."""
        nodes, graph = _make_multi_cb_nodes()
        agg = build_cb_graph(nodes, graph)
        assert len(agg.edges) == 1
        e = agg.edges[0]
        assert e.source_id == "CB0"
        assert e.target_id == "CB1"
        assert e.weight >= 1

    def test_cb_graph_no_intra_cb_edges(self):
        """Edges within same CB are not in the aggregated graph."""
        nodes, graph = _make_multi_cb_nodes()
        agg = build_cb_graph(nodes, graph)
        for e in agg.edges:
            assert e.source_id != e.target_id


class TestBuildEncoderGraph:
    """Test encoder level aggregation."""

    def test_encoder_graph_nodes(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_encoder_graph(nodes, graph)
        assert agg.scale == "encoder"
        assert len(agg.nodes) == 3
        ids = {n.node_id for n in agg.nodes}
        assert ids == {"E0", "E1", "E2"}

    def test_encoder_graph_dispatch_counts(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_encoder_graph(nodes, graph)
        by_id = {n.node_id: n for n in agg.nodes}
        assert by_id["E0"].dispatch_count == 2
        assert by_id["E1"].dispatch_count == 1
        assert by_id["E2"].dispatch_count == 2

    def test_encoder_graph_cross_encoder_edges(self):
        """Only cross-encoder edges appear."""
        nodes, graph = _make_multi_cb_nodes()
        agg = build_encoder_graph(nodes, graph)
        edge_pairs = {(e.source_id, e.target_id) for e in agg.edges}
        # E0→E1 (via buf 0x200), E1→E2 (via buf 0x300)
        assert ("E0", "E1") in edge_pairs
        assert ("E1", "E2") in edge_pairs

    def test_encoder_graph_clusters_by_cb(self):
        """Encoder graph clusters nodes by command buffer."""
        nodes, graph = _make_multi_cb_nodes()
        agg = build_encoder_graph(nodes, graph)
        assert agg.clusters is not None
        assert agg.cluster_key == "command_buffer"
        assert "E0" in agg.clusters.get("CB #0", [])
        assert "E1" in agg.clusters.get("CB #0", [])
        assert "E2" in agg.clusters.get("CB #1", [])


class TestBuildKernelGraph:
    """Test kernel level aggregation."""

    def test_kernel_graph_nodes(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_kernel_graph(nodes, graph)
        assert agg.scale == "kernel"
        assert len(agg.nodes) == 2

    def test_kernel_graph_dispatch_counts(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_kernel_graph(nodes, graph)
        # kA has 3 dispatches, kB has 2
        counts = {n.dispatch_count for n in agg.nodes}
        assert counts == {3, 2}

    def test_kernel_graph_cross_kernel_edges(self):
        """Cross-kernel edges exist, self-loops don't."""
        nodes, graph = _make_multi_cb_nodes()
        agg = build_kernel_graph(nodes, graph)
        for e in agg.edges:
            assert e.source_id != e.target_id
        # At least one cross-kernel edge
        assert len(agg.edges) >= 1

    def test_format_kernel_dot_uses_aggregated(self):
        """format_kernel_dot delegates to aggregated graph builder."""
        nodes, graph = _make_multi_cb_nodes()
        dot = format_kernel_dot(graph)
        assert "digraph kernel_deps" in dot
        assert "K0" in dot or "K1" in dot


class TestAggregatedDotOutput:
    """Test DOT formatting for aggregated graphs."""

    def test_cb_dot_renders(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_cb_graph(nodes, graph)
        dot = format_aggregated_dot(agg, title="test_cb")
        assert "digraph test_cb" in dot
        assert "CB0" in dot
        assert "CB1" in dot
        assert "CB0 -> CB1" in dot

    def test_encoder_dot_has_clusters(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_encoder_graph(nodes, graph)
        dot = format_aggregated_dot(agg, title="test_enc")
        assert "subgraph cluster_" in dot
        assert "E0" in dot
        assert "E1" in dot

    def test_kernel_dot_renders(self):
        nodes, graph = _make_multi_cb_nodes()
        agg = build_kernel_graph(nodes, graph)
        dot = format_aggregated_dot(agg, title="test_kernel")
        assert "digraph test_kernel" in dot

    def test_empty_aggregated_graph(self):
        agg = AggregatedGraph(nodes=[], edges=[], scale="empty")
        dot = format_aggregated_dot(agg)
        assert "digraph" in dot


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

class TestApplyFilters:
    """Test scope filtering of dispatches."""

    def test_filter_by_cb(self):
        nodes = [
            _make_node(0, "kA", cb_idx=0),
            _make_node(1, "kB", cb_idx=1),
            _make_node(2, "kC", cb_idx=0),
        ]
        filtered = _apply_filters(nodes, filter_cb=0)
        assert len(filtered) == 2
        assert all(n.command_buffer_idx == 0 for n in filtered)
        # Re-indexed
        assert filtered[0].dispatch_id == 0
        assert filtered[1].dispatch_id == 1

    def test_filter_by_encoder(self):
        nodes = [
            _make_node(0, "kA", encoder_idx=0),
            _make_node(1, "kB", encoder_idx=1),
            _make_node(2, "kC", encoder_idx=0),
        ]
        filtered = _apply_filters(nodes, filter_encoder=1)
        assert len(filtered) == 1
        assert filtered[0].kernel == "kB"

    def test_filter_by_kernel_pattern(self):
        nodes = [
            _make_node(0, "lu_factor"),
            _make_node(1, "lu_solve"),
            _make_node(2, "gemv"),
        ]
        filtered = _apply_filters(nodes, filter_kernel="lu_*")
        assert len(filtered) == 2
        assert {n.kernel for n in filtered} == {"lu_factor", "lu_solve"}

    def test_combined_filters(self):
        """Multiple filters are ANDed."""
        nodes = [
            _make_node(0, "kA", cb_idx=0, encoder_idx=0),
            _make_node(1, "kA", cb_idx=0, encoder_idx=1),
            _make_node(2, "kB", cb_idx=1, encoder_idx=2),
        ]
        filtered = _apply_filters(nodes, filter_cb=0, filter_kernel="kA")
        assert len(filtered) == 2

    def test_empty_result(self):
        nodes = [_make_node(0, "kA", cb_idx=0)]
        filtered = _apply_filters(nodes, filter_cb=99)
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Barrier tests
# ---------------------------------------------------------------------------

class TestBarrierExtraction:
    """Test extraction of barriers from timeline events."""

    def test_barrier_extracted(self):
        """Barrier events produce BarrierNode objects."""
        trace_data = {
            "events": [
                {
                    "type": "dispatch",
                    "kernel": "k1",
                    "index": 0,
                    "buffers_bound": {0: 0x100},
                    "encoder_idx": 0,
                },
                {
                    "type": "barrier",
                    "scope": "buffers",
                    "encoder_idx": 0,
                    "command_buffer_idx": 0,
                },
                {
                    "type": "dispatch",
                    "kernel": "k2",
                    "index": 1,
                    "buffers_bound": {0: 0x200},
                    "encoder_idx": 0,
                },
            ],
            "command_buffers": [
                {"func_idx": 5, "dispatches": [{"index": 0}, {"index": 1}]},
            ],
            "compute_encoders": [
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 0}, {"index": 1}]},
            ],
        }

        nodes, barriers, num_cbs, num_encoders = extract_dispatches(trace_data)
        assert len(nodes) == 2
        assert len(barriers) == 1
        assert barriers[0].scope == "buffers"
        assert barriers[0].encoder_idx == 0
        assert barriers[0].after_dispatch_id == 0  # last dispatch before barrier

    def test_barrier_before_any_dispatch(self):
        """Barrier at start of encoder has after_dispatch_id = -1."""
        trace_data = {
            "events": [
                {
                    "type": "barrier",
                    "scope": "buffers",
                    "encoder_idx": 0,
                },
                {
                    "type": "dispatch",
                    "kernel": "k1",
                    "index": 0,
                    "buffers_bound": {},
                    "encoder_idx": 0,
                },
            ],
            "command_buffers": [
                {"func_idx": 5, "dispatches": [{"index": 0}]},
            ],
            "compute_encoders": [
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 0}]},
            ],
        }

        nodes, barriers, _, _ = extract_dispatches(trace_data)
        assert len(barriers) == 1
        assert barriers[0].after_dispatch_id == -1

    def test_multiple_barriers(self):
        """Multiple barriers in same encoder tracked correctly."""
        trace_data = {
            "events": [
                {"type": "dispatch", "kernel": "k1", "index": 0, "buffers_bound": {}, "encoder_idx": 0},
                {"type": "barrier", "scope": "buffers", "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k2", "index": 1, "buffers_bound": {}, "encoder_idx": 0},
                {"type": "barrier", "scope": "resources", "encoder_idx": 0},
                {"type": "dispatch", "kernel": "k3", "index": 2, "buffers_bound": {}, "encoder_idx": 0},
            ],
            "command_buffers": [
                {"func_idx": 5, "dispatches": [{"index": 0}, {"index": 1}, {"index": 2}]},
            ],
            "compute_encoders": [
                {"encoder_idx": 0, "command_buffer_idx": 0, "dispatches": [{"index": 0}, {"index": 1}, {"index": 2}]},
            ],
        }

        nodes, barriers, _, _ = extract_dispatches(trace_data)
        assert len(barriers) == 2
        assert barriers[0].after_dispatch_id == 0
        assert barriers[0].scope == "buffers"
        assert barriers[1].after_dispatch_id == 1
        assert barriers[1].scope == "resources"


class TestBarrierEdges:
    """Test that barriers create edges in the dependency graph."""

    def test_barrier_creates_edge(self):
        """Barrier between two dispatches with no shared buffers still creates edge."""
        nodes = [
            _make_node(0, "kA", [(0x100, 0, AccessMode.UNKNOWN)], encoder_idx=0),
            _make_node(1, "kB", [(0x200, 0, AccessMode.UNKNOWN)], encoder_idx=0),
        ]
        barriers = [
            BarrierNode(barrier_id=0, scope="buffers", encoder_idx=0, after_dispatch_id=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=barriers)
        assert len(graph.edges) == 1
        assert graph.edges[0].source_id == 0
        assert graph.edges[0].target_id == 1

    def test_barrier_no_edge_if_already_connected(self):
        """Barrier doesn't duplicate an existing buffer-based edge."""
        nodes = [
            _make_node(0, "kA", [(0x100, 0, AccessMode.UNKNOWN)], encoder_idx=0),
            _make_node(1, "kB", [(0x100, 0, AccessMode.UNKNOWN)], encoder_idx=0),
        ]
        barriers = [
            BarrierNode(barrier_id=0, scope="buffers", encoder_idx=0, after_dispatch_id=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=barriers)
        # Only 1 edge (from shared buffer), barrier doesn't add duplicate
        assert len(graph.edges) == 1
        assert graph.edges[0].buffer_addr == 0x100

    def test_barrier_cross_encoder_ignored(self):
        """Barrier only affects dispatches within the same encoder."""
        nodes = [
            _make_node(0, "kA", [(0x100, 0, AccessMode.UNKNOWN)], encoder_idx=0),
            _make_node(1, "kB", [(0x200, 0, AccessMode.UNKNOWN)], encoder_idx=1),
        ]
        # This barrier is in encoder 0 but after_dispatch_id=0; dispatch 1 is in encoder 1
        barriers = [
            BarrierNode(barrier_id=0, scope="buffers", encoder_idx=0, after_dispatch_id=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=barriers)
        # No edges: no shared buffers, barrier only affects encoder 0
        assert len(graph.edges) == 0

    def test_barrier_before_any_dispatch_no_edge(self):
        """Barrier before first dispatch creates no edge."""
        nodes = [
            _make_node(0, "kA", [(0x100, 0, AccessMode.UNKNOWN)], encoder_idx=0),
        ]
        barriers = [
            BarrierNode(barrier_id=0, scope="buffers", encoder_idx=0, after_dispatch_id=-1),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=barriers)
        assert len(graph.edges) == 0

    def test_barrier_with_three_dispatches(self):
        """D0 → [barrier] → D1 → D2: barrier creates D0→D1 edge."""
        nodes = [
            _make_node(0, "kA", [(0x100, 0, AccessMode.UNKNOWN)], encoder_idx=0),
            _make_node(1, "kB", [(0x200, 0, AccessMode.UNKNOWN)], encoder_idx=0),
            _make_node(2, "kC", [(0x200, 0, AccessMode.UNKNOWN)], encoder_idx=0),
        ]
        barriers = [
            BarrierNode(barrier_id=0, scope="buffers", encoder_idx=0, after_dispatch_id=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=barriers)
        edge_pairs = {(e.source_id, e.target_id) for e in graph.edges}
        # D0→D1 from barrier, D1→D2 from shared buffer 0x200
        assert (0, 1) in edge_pairs
        assert (1, 2) in edge_pairs

    def test_no_barriers_unchanged(self):
        """Graph with no barriers behaves exactly as before."""
        nodes = [
            _make_node(0, "kA", [(0x100, 0, AccessMode.UNKNOWN)]),
            _make_node(1, "kB", [(0x100, 0, AccessMode.UNKNOWN)]),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=None)
        assert len(graph.edges) == 1
        assert graph.edges[0].dep_type == DepType.SHARED


class TestBarrierDotOutput:
    """Test that barrier nodes render in DOT output."""

    def test_barrier_diamond_in_dot(self):
        """Barriers render as diamond nodes in DOT."""
        nodes = [
            _make_node(0, "kA", [(0x100, 0, AccessMode.UNKNOWN)], cb_idx=0, encoder_idx=0),
            _make_node(1, "kB", [(0x200, 0, AccessMode.UNKNOWN)], cb_idx=0, encoder_idx=0),
        ]
        barriers = [
            BarrierNode(barrier_id=0, scope="buffers", encoder_idx=0, after_dispatch_id=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=barriers)
        dot = format_dot(graph, cluster_by_cb=True, barriers=barriers)
        assert "barrier0" in dot
        assert "diamond" in dot
        assert "D0 -> barrier0" in dot
        assert "barrier0 -> D1" in dot

    def test_no_barriers_no_diamonds(self):
        """DOT without barriers has no barrier rendering."""
        nodes = [
            _make_node(0, "kA", [(0x100, 0, AccessMode.UNKNOWN)], cb_idx=0),
            _make_node(1, "kB", [(0x100, 0, AccessMode.UNKNOWN)], cb_idx=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        dot = format_dot(graph, cluster_by_cb=True)
        assert "barrier" not in dot.lower() or "barrier" not in dot

    def test_barrier_edge_label(self):
        """Barrier-induced edges labeled 'barrier' in DOT."""
        nodes = [
            _make_node(0, "kA", [(0x100, 0, AccessMode.UNKNOWN)], cb_idx=0, encoder_idx=0),
            _make_node(1, "kB", [(0x200, 0, AccessMode.UNKNOWN)], cb_idx=0, encoder_idx=0),
        ]
        barriers = [
            BarrierNode(barrier_id=0, scope="buffers", encoder_idx=0, after_dispatch_id=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=barriers)
        dot = format_dot(graph, cluster_by_cb=True, barriers=barriers)
        assert 'label="barrier"' in dot


class TestAggregatedBarrierCounts:
    """Test that aggregated graphs include barrier counts."""

    def test_cb_graph_barrier_count(self):
        """CB graph includes barrier count in label."""
        U = AccessMode.UNKNOWN
        nodes = [
            _make_node(0, "kA", [(0x100, 0, U)], cb_idx=0, encoder_idx=0),
            _make_node(1, "kB", [(0x200, 0, U)], cb_idx=0, encoder_idx=0),
        ]
        barriers = [
            BarrierNode(barrier_id=0, scope="buffers", encoder_idx=0, command_buffer_idx=0, after_dispatch_id=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=barriers)
        agg = build_cb_graph(nodes, graph, barriers=barriers)
        assert agg.nodes[0].barrier_count == 1
        assert "1 barriers" in agg.nodes[0].label

    def test_encoder_graph_barrier_count(self):
        """Encoder graph includes barrier count in label."""
        U = AccessMode.UNKNOWN
        nodes = [
            _make_node(0, "kA", [(0x100, 0, U)], cb_idx=0, encoder_idx=0),
            _make_node(1, "kB", [(0x200, 0, U)], cb_idx=0, encoder_idx=0),
        ]
        barriers = [
            BarrierNode(barrier_id=0, scope="buffers", encoder_idx=0, command_buffer_idx=0, after_dispatch_id=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True, barriers=barriers)
        agg = build_encoder_graph(nodes, graph, barriers=barriers)
        by_id = {n.node_id: n for n in agg.nodes}
        assert by_id["E0"].barrier_count == 1
        assert "1 barriers" in by_id["E0"].label

    def test_no_barriers_no_count_in_label(self):
        """Without barriers, label doesn't mention barriers."""
        U = AccessMode.UNKNOWN
        nodes = [
            _make_node(0, "kA", [(0x100, 0, U)], cb_idx=0, encoder_idx=0),
        ]
        graph = build_dependency_graph(nodes, conservative=True)
        agg = build_encoder_graph(nodes, graph)
        assert "barrier" not in agg.nodes[0].label
