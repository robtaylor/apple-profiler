# Learnings

### Lazy Loading Xctrace XML Avoids Memory Overhead (2026-03-05)
**Insight**: Xcode trace files can be very large when exported to XML. Rather than loading the entire trace structure on open, we only parse the TOC and load individual tables (cpu_profile, os_signpost, etc.) when explicitly requested via `export_table()`.

**Context**: Large traces can have millions of samples. A naive load-all approach would stall startup. The table-by-table pattern scales better for batch analysis where users query specific tables.

### MCP Tools Should Be Thin Wrappers Around Core APIs (2026-03-05)
**Insight**: The mcp_server.py exposes 7 tools that directly call TraceFile methods (cpu_samples(), top_functions(), hangs(), signpost_events(), signpost_intervals()). No business logic lives in the MCP layer—it's purely JSON serialization and error handling.

**Context**: Keeps the core library testable and reusable independently of MCP. MCP is an integration boundary, not a logic barrier.

### Testing MCP Stdio Transport via Claude Code CLI (2026-03-05)
**Insight**: MCP integration tests use conftest.py to set up fixtures that mock the stdio transport. The pattern launches mcp_server in a subprocess and uses `mcp_client` fixture to send JSON-RPC messages directly over stdin/stdout, avoiding any dependency on Claude Code itself during tests.

**Context**: Essential for validating tool signatures, error handling, and response serialization without running against a real Claude instance. The test suite (test_mcp_integration.py) covers tool invocation, JSON-RPC protocol compliance, and integration with the core TraceFile API.

### Smithery Marketplace Distribution for MCP Servers (2026-03-05)
**Insight**: MCP servers are discovered via Smithery marketplace (smithery.yaml). The file declares the server name, description, contact, and installation method. Listing in Smithery enables users to discover and auto-integrate the server via MCP client tools.

**Context**: For wider adoption of MCP tools. Without marketplace registration, discoverability depends on word-of-mouth or hardcoded client configuration.

### Multi-Channel MCP Distribution Maximizes Adoption (2026-03-07)
**Insight**: MCP servers benefit from distribution across multiple discovery channels. Beyond Smithery, integration with Claude Code's native plugin marketplace (`.claude-plugin/marketplace.json` and `plugin.json`) directly surfaces the server to Claude Code users, reducing friction for in-tool discovery and installation.

**Context**: Users of Claude Code can discover and install the MCP server without leaving their IDE. Different channels reach different user segments—Smithery reaches generic MCP client users, Claude Code plugins reach IDE-native users. Dual registration increases total addressable audience and simplifies onboarding for the largest deployment target (Claude Code users).

### idx -16009 is memoryBarrierWithScope, not dispatchThreads (2026-03-17)
**Insight**: Controlled experiment (capture with/without barrier, diff streams) proved -16009 is memoryBarrierWithScope:.buffers. The real dispatchThreads:threadsPerThreadgroup: is -16078. This means all prior dispatch counts were inflated (barriers counted as dispatches).

**Context**: Discovered by writing tools/metal_barrier_test.swift with two variants and comparing captured .gputrace function streams


### Rosetta stone technique for reverse-engineering function indices (2026-03-17)
**Insight**: Write a Swift Metal program with known API calls, capture .gputrace via MTLCaptureManager (MTL_CAPTURE_ENABLED=1), then read the function stream with Python to map indices to APIs. 13 variants cover compute, blit, barriers, events, pipeline creation variants.

**Context**: tools/metal_rosetta_stone.swift has all 13 variants, tools/metal_rosetta_decode.py reads and compares them


### Multiple wrong index mappings from old reverse engineering (2026-03-17)
**Insight**: -15996 is makeSharedEvent() not newComputePipelineStateWithDescriptor. -15990 is encodeSignalEvent not addCompletedHandler. -16356 is addCompletedHandler. -16367 is user setPurgeableState vs -16371 GPUTools-inserted. -16370 is endEncoding for GPUTools auto-inserted blit encoder.

**Context**: All verified by rosetta stone variants 6-12


### Function stream reuses single buffer (2026-03-17)
**Insight**: fstream.readFunction() always returns the same pointer address - the framework deserializes one record at a time into a fixed buffer. Cannot use pointer addresses to compute file offsets.

**Context**: Discovered while trying to map function records to file positions for binary patching

### Integrating verified index corrections requires multi-tool updates (2026-03-17)
**Insight**: The rosetta stone corrections (-16009 barrier, -16078 real dispatch, -15996 SharedEvent) required coordinated updates across three tools: (1) gputrace_timeline.py changes event type from "dispatch" to "barrier"; (2) gputrace_depgraph.py filters out barrier nodes from dispatch counts with clear documentation; (3) gputrace_dump_setbuffer.py updates reference labels. Without all three fixes, barriers would still inflate dispatch statistics.

**Context**: Tool integration chain: upstream corrections (timeline) propagate to downstream consumers (depgraph, summary). Each tool independently validates its index assumptions, so corrections must be applied systematically. Partial updates create silent bugs where affected calculations appear correct but use wrong inputs.

