# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Known Apple GPU performance counter hash → human-readable name mappings.

Apple GPU performance counters (as found in .gputrace streamData) are identified
by 65-char hex hashes (_prefix + 64 lowercase hex digits). These hashes are
opaque identifiers for hardware counter configurations.

This module provides the known mappings from hash → human-readable label,
derived from:
  1. Cross-GPU-family counter plists (G13X, G14X have explicit hash→name files)
  2. Binary string analysis of GTShaderProfiler framework
  3. Partition+Select register matching across GPU generations

The mappings here were extracted from an M4 Pro (G17P) but many hashes are
shared across Apple Silicon generations. For newer GPUs (G15+), the formula
system generates JavaScript at runtime via _SetAndEvaluateRawCounterValues,
so no static hash→name binding exists in the binary.

Hashes are organized into 4 hardware source groups:
  - APS_USC:    Apple Programmable Shader / Unified Shader Core (Partition 32)
  - BMPR_RDE_0: Bandwidth Memory Performance Registers (Partitions 15, 17)
  - RDE_0:      Render Data Engine (Partitions 5, 24, 27, 28, 29)
  - Firmware:   Firmware-level counters (Partition 0)
"""
from __future__ import annotations


def label_for_hash(hash_str: str) -> str:
    """Return a human-readable label for a GPU counter hash.

    Falls back to a group+index label (e.g. "USC Counter 3") for unmapped
    hashes, and to the truncated hash itself for completely unknown hashes.
    """
    entry = COUNTER_MAP.get(hash_str)
    if entry is not None:
        return entry["label"]
    # Unknown hash — truncate for display
    return hash_str[:16] + "..."


def vendor_names_for_hash(hash_str: str) -> list[str]:
    """Return vendor counter names that use this hash in their formulas.

    A single raw hardware counter may contribute to multiple derived counters
    (e.g., bytes-read-from-main-memory feeds into FSMainMemoryThroughput,
    FSArithmeticIntensity, GPUReadBandwidth, etc.).

    Returns an empty list for unmapped or unknown hashes.
    """
    entry = COUNTER_MAP.get(hash_str)
    if entry is not None:
        return list(entry.get("vendor_names", []))
    return []


# ---------------------------------------------------------------------------
# Hash → counter info mapping
#
# Each entry:
#   group:        Hardware source group (APS_USC, BMPR_RDE_0, RDE_0, Firmware)
#   index:        Index within the group's Limiter Counter List Map
#   label:        Best human-readable name for this raw counter
#   vendor_names: Derived counter names that reference this hash (may be empty)
#   hw_partition: Hardware partition ID from AGXMetalPerfCountersExternal.plist
#   hw_select:    Hardware select register value
#   source:       How the mapping was determined
# ---------------------------------------------------------------------------
COUNTER_MAP: dict[str, dict] = {
    # ── APS_USC (Partition 32) ─────────────────────────────────────────────
    # Unified Shader Core counters. On older GPUs (G14X), partition 32 held
    # L2 cache, MMU, main memory, and miss buffer counters.
    # All 10 are unmapped on M4 Pro — runtime JS formula generation only.
    "_bc236dcf3744edf6bcf329ae47088528b35ea5e0beac601cf6781545eb2327cc": {
        "group": "APS_USC", "index": 0, "label": "USC Counter 0",
        "vendor_names": [], "hw_partition": 32, "hw_select": 35184372088832,
        "source": "unmapped",
    },
    "_0f214de435fcac756d84217b6540743235c7b1d500f32f8d0cb68aaaa75ac3e2": {
        "group": "APS_USC", "index": 1, "label": "USC Counter 1",
        "vendor_names": [], "hw_partition": 32, "hw_select": 70368744177664,
        "source": "unmapped",
    },
    "_5fa064796fa00e51a16682635d496690f5bb01777755209762a8752a444bde93": {
        "group": "APS_USC", "index": 2, "label": "USC Counter 2",
        "vendor_names": [], "hw_partition": 32, "hw_select": 8589934592,
        "source": "unmapped",
    },
    "_f30d812c264ead364d91a31eacd9fc8714703d9da79f2df8a51edc8b4e53bff3": {
        "group": "APS_USC", "index": 3, "label": "USC Counter 3",
        "vendor_names": [], "hw_partition": 32, "hw_select": 4294967296,
        "source": "unmapped",
    },
    "_fa953797d6623765218f4f921adcc6a5994bbdbeffb8972600a1055083292e8a": {
        "group": "APS_USC", "index": 4, "label": "USC Counter 4",
        "vendor_names": [], "hw_partition": 32, "hw_select": 549755813888,
        "source": "unmapped",
    },
    "_ae23bde9704f5d8ed4f6fe1bd07d71d94edccfc3853db7d008dce330be2b3844": {
        "group": "APS_USC", "index": 5, "label": "USC Counter 5",
        "vendor_names": [], "hw_partition": 32, "hw_select": 34359738368,
        "source": "unmapped",
    },
    "_f1f19f9b5367da3da259d84460e29e1286ac04d08bcde9ec343fab5a43ce7386": {
        "group": "APS_USC", "index": 6, "label": "USC Counter 6",
        "vendor_names": [], "hw_partition": 32, "hw_select": 137438953472,
        "source": "unmapped",
    },
    "_dc9d2c02b3df41ad60f6512a3823b2d0f9b735ae9c41650c4396bac13a7a3c5e": {
        "group": "APS_USC", "index": 7, "label": "USC Counter 7",
        "vendor_names": [], "hw_partition": 32, "hw_select": 17592186044416,
        "source": "unmapped",
    },
    "_3ee42b964ae05a0daaab1ed4621a4811bc2fe4e672fe5e3c84717c5b98184827": {
        "group": "APS_USC", "index": 8, "label": "USC Counter 8",
        "vendor_names": [], "hw_partition": 32, "hw_select": 274877906944,
        "source": "unmapped",
    },
    "_b08194796a2cb35a8699c8d23b129c582951d9d1941fbc8e36dbaafa02d474e7": {
        "group": "APS_USC", "index": 9, "label": "USC Counter 9",
        "vendor_names": [], "hw_partition": 32, "hw_select": 9223372105574252544,
        "source": "unmapped",
    },

    # ── BMPR_RDE_0 (Partitions 15, 17) ────────────────────────────────────
    # Bandwidth/Memory Performance Registers.
    "_f539a056fbd45e39d61b9d34af0ea1a305dd8af6f005b5aa29488f2c966e7f21": {
        "group": "BMPR_RDE_0", "index": 0,
        "label": "MainMemory Read Bytes (VS/FS)",
        "vendor_names": [
            "FSArithmeticIntensity", "FSBytesReadFromMainMemory",
            "FSMainMemoryThroughput", "VSArithmeticIntensity",
            "VSBytesReadFromMainMemory", "VSMainMemoryThroughput",
        ],
        "hw_partition": 17, "hw_select": 537461504,
        "source": "cross_gpu_plist",
    },
    "_c420303ee3c2ea941491fa0a071ec1553251898524fce318c3635bc711160395": {
        "group": "BMPR_RDE_0", "index": 1,
        "label": "MainMemory Read Bytes (VS/FS) [alt]",
        "vendor_names": [
            "FSArithmeticIntensity", "FSBytesReadFromMainMemory",
            "FSMainMemoryThroughput", "VSArithmeticIntensity",
            "VSBytesReadFromMainMemory", "VSMainMemoryThroughput",
        ],
        "hw_partition": 17, "hw_select": 852224,
        "source": "cross_gpu_plist",
    },
    "_c89f25b2a31116ea6540fc6b89b623da7d4cbc9b2349ce7dbe4423a1395da60a": {
        "group": "BMPR_RDE_0", "index": 2,
        "label": "MainMemory Write Bytes",
        "vendor_names": [
            "BytesWrittenToMainMemory", "CSArithmeticIntensity",
            "FSArithmeticIntensity", "FSBytesWrittenToMainMemory",
            "FSMainMemoryThroughput", "GPUWriteBandwidth",
            "MainMemoryThroughput", "MainMemoryTraffic",
            "VSArithmeticIntensity", "VSBytesWrittenToMainMemory",
            "VSMainMemoryThroughput",
        ],
        "hw_partition": 17, "hw_select": 590848,
        "source": "cross_gpu_plist",
    },
    "_7eba0657c66c2437c8d6952fb82117b9399d920ea84fa32c5bdc9d1dee49e290": {
        "group": "BMPR_RDE_0", "index": 3,
        "label": "MainMemory Write Bytes [alt]",
        "vendor_names": [
            "BytesWrittenToMainMemory", "CSArithmeticIntensity",
            "FSArithmeticIntensity", "FSBytesWrittenToMainMemory",
            "FSMainMemoryThroughput", "GPUWriteBandwidth",
            "MainMemoryThroughput", "MainMemoryTraffic",
            "VSArithmeticIntensity", "VSBytesWrittenToMainMemory",
            "VSMainMemoryThroughput",
        ],
        "hw_partition": 17, "hw_select": 852480,
        "source": "cross_gpu_plist",
    },
    "_5c5c55d05fb355aa5be61ac63c88eb4a2a521a47dd8f79c18b5c1df163d5cb55": {
        "group": "BMPR_RDE_0", "index": 4,
        "label": "L2 Cache Utilization",
        "vendor_names": ["L2CacheLimiter", "L2CacheMissRate", "L2CacheUtilization"],
        "hw_partition": 15, "hw_select": 65537,
        "source": "cross_gpu_plist",
    },
    "_c9bcd5df6397dc8477a12ddf9358bccbbb3d8e52fc3dadab320be9bbb14fe157": {
        "group": "BMPR_RDE_0", "index": 5,
        "label": "L2 Cache Limiter",
        "vendor_names": ["L2CacheLimiter"],
        "hw_partition": 15, "hw_select": 200704,
        "source": "cross_gpu_plist",
    },
    "_fdc48a2370f6885da6ac169661812057de2cf71fbbbcb5df8348a78f112992dc": {
        "group": "BMPR_RDE_0", "index": 6,
        "label": "MMU Limiter",
        "vendor_names": ["MMULimiter", "MissBufferFullStallRatio"],
        "hw_partition": 15, "hw_select": 262148,
        "source": "cross_gpu_plist",
    },
    "_6d6a7c8efb15986fa71f8bf4a6a06f8942199b36680e516766e92490607c958d": {
        "group": "BMPR_RDE_0", "index": 7,
        "label": "MMU Utilization",
        "vendor_names": ["MMULimiter", "MMUUtilization", "MainMemoryTraffic"],
        "hw_partition": 15, "hw_select": 131328,
        "source": "cross_gpu_plist",
    },
    "_44b570b50d07ac48e94ccb4a92c113afe3626e472a1ab87b8f5374302c9c5f34": {
        "group": "BMPR_RDE_0", "index": 8,
        "label": "BMPR Counter 8",
        "vendor_names": [], "hw_partition": 15, "hw_select": 0,
        "source": "unmapped",
    },
    "_61f7db669b241a51962b25096fa7f4bdb11deb4d851c51ed91c34b4f6b086d75": {
        "group": "BMPR_RDE_0", "index": 9,
        "label": "GPU Cycles (BMPR)",
        "vendor_names": ["GPUCycles"],
        "hw_partition": 28, "hw_select": 2,
        "source": "binary_analysis",
    },

    # ── Firmware (Partition 0) ─────────────────────────────────────────────
    "_53c82a25ac54f8ecd1e94581a4020f0a20529b4813cab97e3977346ad0e270a8": {
        "group": "Firmware", "index": 0, "label": "Firmware Counter 0",
        "vendor_names": [], "hw_partition": 0, "hw_select": 10,
        "source": "unmapped",
    },
    "_0965349a7930ddeeae0312bca50ac5672d7ecdb1e38a915536c0e7b8a1a3c321": {
        "group": "Firmware", "index": 1, "label": "Firmware Counter 1",
        "vendor_names": [], "hw_partition": 0, "hw_select": 9,
        "source": "unmapped",
    },

    # ── RDE_0 (Partitions 5, 24, 27, 28, 29) ──────────────────────────────
    # Render Data Engine counters.
    "_04d4411374e68233627aa77e33b97414d97097b7d3599dc0555f05e8ba0c27ad": {
        "group": "RDE_0", "index": 0,
        "label": "Texture Memory Read Compression",
        "vendor_names": ["CompressionRatioTextureMemoryRead"],
        "hw_partition": 5, "hw_select": 66048,
        "source": "cross_gpu_plist",
    },
    "_8788387fa2f782d31f4553bc55eb34284415d12b986df376e394838d5075f058": {
        "group": "RDE_0", "index": 1,
        "label": "MMU TLB Requests",
        "vendor_names": ["MMUTLBRequests"],
        "hw_partition": 5, "hw_select": 851969,
        "source": "partition_select_match",
    },
    "_7c42e99464b33ee51de11bdd9f8cf11a14473f7061e75f7589a3578a7757abfd": {
        "group": "RDE_0", "index": 2,
        "label": "Texture Filtering Limiter",
        "vendor_names": ["TextureFilteringLimiter"],
        "hw_partition": 29, "hw_select": 537067519,
        "source": "cross_gpu_plist",
    },
    "_8ff5f6e1c2e52558354049aef96f7abf429f223a3fc4e626292d894456e02fc2": {
        "group": "RDE_0", "index": 3, "label": "RDE Counter 3",
        "vendor_names": [], "hw_partition": 24, "hw_select": 131074,
        "source": "unmapped",
    },
    "_eb1b52d1bee0dd7bb1d4f6345a6a34e9d9f75340093e425454be8205d717c2ba": {
        "group": "RDE_0", "index": 4, "label": "RDE Counter 4",
        "vendor_names": [], "hw_partition": 27, "hw_select": 537067519,
        "source": "unmapped",
    },
    "_ae304fc8bce5708ffef30935687e442d6bea78f814055a5fe6e3380013d7e507": {
        "group": "RDE_0", "index": 5,
        "label": "Texture Samples",
        "vendor_names": [
            "CSTextureSamples", "CSTextureSamplesPerInvocation",
            "FSTextureSamples", "FSTextureSamplesPerInvocation",
            "PredicatedTextureReadPercentage", "TextureAccesses",
            "TextureSamples", "UnCompressedSamplesPercent",
            "UncompressedSamples", "VSTextureSamples",
            "VSTextureSamplesPerInvocation",
        ],
        "hw_partition": 29, "hw_select": 538509504,
        "source": "cross_gpu_plist",
    },
    "_8922765bce9a86586c4e9f2d8c17967bf71fc42fefdc94b60ac069f686424044": {
        "group": "RDE_0", "index": 6, "label": "RDE Counter 6",
        "vendor_names": [], "hw_partition": 27, "hw_select": 65538,
        "source": "unmapped",
    },
    "_d856a24dfeb33f1dad922753efc16618fdf12ef3544115f2f5fffd93affab8d5": {
        "group": "RDE_0", "index": 7, "label": "RDE Counter 7",
        "vendor_names": [], "hw_partition": 29, "hw_select": 538575232,
        "source": "unmapped",
    },
    "_7646a8523871192073a29fb3af219f4dbddae3339e969e0da8ef8d84a3d46ec5": {
        "group": "RDE_0", "index": 8,
        "label": "TPU/Sample Limiter",
        "vendor_names": [
            "FragmentSampleLimiter", "FragmentTPULimiter",
            "SampleLimiter", "TPULimiter",
            "VertexSampleLimiter", "VertexTPULimiter",
        ],
        "hw_partition": 29, "hw_select": 537198652,
        "source": "cross_gpu_plist",
    },
    "_fec93bc804b85a65d152fdd4747b95f2b6633ea518b6bb44a7bc87186198c2a8": {
        "group": "RDE_0", "index": 9, "label": "RDE Counter 9",
        "vendor_names": [], "hw_partition": 27, "hw_select": 65537,
        "source": "unmapped",
    },
    "_da0afb5d20fd710a2f7ce18da42b2a53dc3d3fcfe45ac35c28e75d4402986d37": {
        "group": "RDE_0", "index": 10, "label": "RDE Counter 10",
        "vendor_names": [], "hw_partition": 24, "hw_select": 131073,
        "source": "unmapped",
    },
    "_1827ca25b7318e2df60eb0fe4f0c290b43054021ec3233e1fcdcf7b622fe4589": {
        "group": "RDE_0", "index": 11,
        "label": "Texture Memory Read Compression [alt]",
        "vendor_names": ["CompressionRatioTextureMemoryRead"],
        "hw_partition": 5, "hw_select": 65600,
        "source": "cross_gpu_plist",
    },
    "_28b593c4df5283c263b57e9a6d113d841094eb62c8e83c152a3d733a17af7b80": {
        "group": "RDE_0", "index": 12,
        "label": "GPU Cycles (RDE)",
        "vendor_names": ["GPUCycles"],
        "hw_partition": 28, "hw_select": 2,
        "source": "binary_analysis",
    },
}
