# Project Narrative: apple-profiler

## Summary
A Python wrapper for parsing Xcode trace files (.trace recordings) exported as XML. We're building a low-level API for programmatically analyzing iOS/macOS performance data—CPU samples, signposts, hangs—without leaving the Python ecosystem.

## Current Foci

- Extending dependency graph tools to support all GPU trace formats (not just compute)
- Adding barrier node support to dependency graphs (explicit sync point modeling)

## How It Works
The library sits at three abstraction levels:

1. **Low-level**: `_xctrace.py` – subprocess wrapper for `xctrace export` to extract XML from binary .trace files
2. **Middle-level**: `_parser.py` – XML-to-Python conversion, handles schema resolution and ElementTree traversal
3. **High-level**: `TraceFile` class – convenient API for accessing trace metadata and lazy-loading data tables

Tables are discovered via TOC (table of contents) XML, then loaded individually via `export_table()` when requested. Each table gets parsed into typed Python objects (CpuSample, SignpostEvent, Thread, Frame, etc.).

## The Story So Far
**Epoch 1 (6aa520a)**: Initial full implementation with data models, XML parser, and xctrace subprocess integration. Lazy-loaded table architecture optimized for multi-file analysis.

**Epoch 2 (9c346a4)**: Added real xctrace XML test fixtures covering cpu_profile, os_signpost, potential_hangs, and TOC structures for validation against actual Xcode traces.

**Epoch 3 (36b9a68)**: Launched as MCP server via FastMCP, exposing 7 tools: profiler_open_trace, profiler_cpu_samples, profiler_top_functions, profiler_hangs, profiler_signpost_events, profiler_signpost_intervals, profiler_list_tables. This enables programmatic trace analysis via MCP stdio transport.

**Epoch 4 (6881d10–68f4445)**: Completed testing & distribution infrastructure. Added MCP integration tests using Claude Code CLI stdio mocking (test_mcp_integration.py), Smithery marketplace registration, and console script entry point (apple-profiler CLI command). CI workflow validates linting and tests on every push. Real trace fixtures (cpu_profile, hangs, signpost) added as zipped binary data for comprehensive integration testing. The project is now production-ready for external distribution.

**Epoch 5 (d481aa5)**: Expanded distribution to Claude Code native plugin ecosystem. Added `.claude-plugin/` directory with marketplace.json and plugin.json, enabling the MCP server to be discovered and installed directly via Claude Code's plugin marketplace (in addition to Smithery). This dual-channel distribution (Smithery + Claude Code plugins) increases visibility and ease of adoption for users within the Claude ecosystem.

**Epoch 6 (Mar 17, 2026 — completed)**: Rosetta stone verification campaign for Metal API function index correctness. Developed controlled-experiment technique: compile Swift Metal programs with known API calls, capture .gputrace via MTLCaptureManager, extract function streams with Python, and diff variants to identify indices. Completed 13 variants covering compute pipelines, blit encoders, barriers, events, and descriptor creation. Discovered 7 major index misidentifications from original reverse engineering (e.g., -16009 is `memoryBarrierWithScope:`, not `dispatchThreads:`; -15996 is `makeSharedEvent()`, not pipeline creation). Built tools/metal_rosetta_stone.swift (13 test programs), tools/metal_rosetta_decode.py (stream decoder), and tools/gputrace_depgraph.py (dependency analyzer with multi-scale output). Enhanced gputrace_timeline.py with encoder tracking and implicit encoder synthesis. Results verified against real Xcode traces and documented in memory/gputrace-format.md.

**Mar 17, 2026 — Integration phase (completed)**: Integrated verified index corrections into production tools:
  - Fixed gputrace_timeline.py: -16009 now emits `"type": "barrier"` events instead of dispatches; -16078 is the real `dispatchThreads:`; -15996 (`makeSharedEvent()`) skipped from pipeline detection
  - Updated gputrace_depgraph.py: barriers filtered from dispatch count (documented that they are sync points, not compute resources); future enhancement to model barriers as explicit DAG nodes
  - Fixed gputrace_dump_setbuffer.py: corrected index labels to match verified mappings
  - Eliminates inflated dispatch counts (previously counted barriers as dispatches)

## Dragons & Gotchas
- **xctrace dependency**: We shell out to the xctrace binary (must be on PATH). Fails silently if not available or if the .trace file is corrupted.
- **Schema resolution**: The XML schema system uses column IDs and element references—required careful reverse-engineering of Xcode's schema format.
- **Type strictness**: pyright in strict mode catches subtle None/Optional edge cases; found several during initial development and refactoring.
- **Assertion patterns**: Heavy use of `assert` in trace.py (e.g., `assert run is not None`) can hide real parsing failures—should be replaced with explicit error handling before production release.

- **Function stream disorder**: The unsorted-capture function stream does NOT preserve API call order — commit, computeCommandEncoder, and setComputePipelineState appear AFTER the dispatches they logically precede. Order must be reconstructed from timestamps.

- **Index reuse**: -16352 (commandBuffer) fires once per encoder creation, not just per command buffer — in multi-encoder traces it appears multiple times with the same address. -16009 is not a dispatch call at all (it's memoryBarrierWithScope:).

- **Function buffer reuse**: fstream.readFunction() always returns the same pointer address — the framework deserializes one record at a time into a fixed buffer. Cannot use pointer addresses for file offset calculations.

## Open Questions
- How do we handle .trace files from different Xcode versions? Does the schema change?
- What's the adoption path for MCP server discovery—will Smithery listing drive adoption?
- Should we add a GUI or web dashboard for browsing trace data, or stay CLI/API-only?
- Are there performance hotspots when parsing very large traces (>1GB)?
- Should we integrate verified index mappings into a lookup table and deprecate the old FUNC_NAMES mapping?
- How does Xcode determine 'implicit barriers' — pure buffer hazard analysis or something stored in index/capture files?
