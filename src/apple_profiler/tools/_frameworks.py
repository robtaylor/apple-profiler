"""Shared Apple GPU framework loading utilities.

All GPU tool scripts need to:
1. Ensure DYLD_FRAMEWORK_PATH is set (re-exec if not)
2. Load the Apple private GPU frameworks from Xcode

This module centralizes that logic so each tool script doesn't duplicate it.
"""
from __future__ import annotations

import os
import sys

SHARED_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"

_FRAMEWORK_NAMES = [
    "GPUToolsCore",
    "GPUTools",
    "GPUToolsPlatform",
    "GLToolsCore",
    "GPUToolsServices",
]


def ensure_dyld_framework_path() -> None:
    """Re-exec with DYLD_FRAMEWORK_PATH if not set.

    dyld reads this variable at process startup to resolve @rpath references,
    so it must be set before any GPU framework is loaded. When missing, we
    set it and os.execv() to restart the process.
    """
    if os.environ.get("DYLD_FRAMEWORK_PATH") != SHARED_FW:
        os.environ["DYLD_FRAMEWORK_PATH"] = SHARED_FW
        os.execv(sys.executable, [sys.executable] + sys.argv)


def load_frameworks() -> None:
    """Load Apple private GPU frameworks in dependency order."""
    from Foundation import NSBundle  # type: ignore[import-untyped]

    for name in _FRAMEWORK_NAMES:
        bundle = NSBundle.bundleWithPath_(f"{SHARED_FW}/{name}.framework")
        if bundle is not None:
            bundle.load()
    # System private framework
    sys_bundle = NSBundle.bundleWithPath_(
        "/System/Library/PrivateFrameworks/GPUToolsCapture.framework"
    )
    if sys_bundle is not None:
        sys_bundle.load()
