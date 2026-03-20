// xcode_gputrace_automation.js — JXA automation for Xcode .gputrace replay
//
// Usage: osascript -l JavaScript tools/xcode_gputrace_automation.js <action> [args...]
//
// Actions:
//   close-window <stem>   — Close the gputrace window matching <stem>
//   ensure-replay <stem>  — Check Profile checkbox, click Replay
//   poll-activity         — Check Activity View for GPU profiling status

"use strict";

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
var args = [];
// $.NSProcessInfo args: [0]=osascript [1]=-l [2]=JavaScript [3]=<script> [4..]=user args
var nsArgs = $.NSProcessInfo.processInfo.arguments;
for (var i = 4; i < nsArgs.count; i++) {
    args.push(nsArgs.objectAtIndex(i).js);
}

var action = args[0] || "";
var stem = args[1] || "";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
var se = Application("System Events");
se.includeStandardAdditions = true;
var xcode = se.processes["Xcode"];

function jsonResult(obj) {
    return JSON.stringify(obj);
}

/**
 * Find the gputrace window whose title contains `stem`.
 */
function findGpuTraceWindow(stem) {
    var wins;
    try { wins = xcode.windows(); } catch (e) { return null; }
    for (var i = 0; i < wins.length; i++) {
        try {
            var name = wins[i].name();
            if (name && name.indexOf(stem) >= 0 && name.indexOf(".gputrace") >= 0) {
                return wins[i];
            }
        } catch (e) { /* skip inaccessible windows */ }
    }
    // Fallback: match just the stem (window title may not include extension)
    for (var i = 0; i < wins.length; i++) {
        try {
            var name = wins[i].name();
            if (name && name.indexOf(stem) >= 0) {
                return wins[i];
            }
        } catch (e) {}
    }
    return null;
}

/**
 * Navigate the deterministic accessibility path from the gputrace window
 * to the editor context split view that contains Replay/Profile controls.
 *
 * Path (from Accessibility Inspector):
 *   window → splitterGroups[0] → groups("editor area") → splitterGroups[0]
 *   → groups("New Editor") → splitterGroups[0] → splitterGroups[0]
 *   → groups(stem) → splitterGroups[0]
 */
function navigateToEditorControls(win, stem) {
    try {
        var sg1 = win.splitterGroups()[0];

        // Find "editor area" group
        var editorArea = null;
        var groups1 = sg1.groups();
        for (var i = 0; i < groups1.length; i++) {
            try {
                if (groups1[i].description() === "editor area") {
                    editorArea = groups1[i];
                    break;
                }
            } catch (e) {}
        }
        if (!editorArea) return null;

        var sg2 = editorArea.splitterGroups()[0];

        // Find "New Editor" group
        var newEditor = null;
        var groups2 = sg2.groups();
        for (var i = 0; i < groups2.length; i++) {
            try {
                if (groups2[i].description() === "New Editor") {
                    newEditor = groups2[i];
                    break;
                }
            } catch (e) {}
        }
        if (!newEditor) return null;

        var sg3 = newEditor.splitterGroups()[0];
        var sg4 = sg3.splitterGroups()[0];

        // Find group whose description contains the gputrace stem
        var editorCtx = null;
        var groups3 = sg4.groups();
        for (var i = 0; i < groups3.length; i++) {
            try {
                var desc = groups3[i].description();
                if (desc && desc.indexOf(stem) >= 0) {
                    editorCtx = groups3[i];
                    break;
                }
            } catch (e) {}
        }
        if (!editorCtx) return null;

        return editorCtx.splitterGroups()[0];
    } catch (e) {
        return null;
    }
}

/**
 * Fallback: recursive search for a UI element by role and name.
 * Only used if deterministic navigation fails.
 */
function findElementRecursive(element, role, name, depth) {
    if (depth > 12) return null;
    try {
        var items;
        if (role === "button") items = element.buttons.whose({name: name})();
        else if (role === "checkbox") items = element.checkboxes.whose({name: name})();
        if (items && items.length > 0) return items[0];
    } catch (e) {}
    try {
        var sgs = element.splitterGroups();
        for (var i = 0; i < sgs.length; i++) {
            var found = findElementRecursive(sgs[i], role, name, depth + 1);
            if (found) return found;
        }
    } catch (e) {}
    try {
        var gs = element.groups();
        for (var i = 0; i < gs.length; i++) {
            var found = findElementRecursive(gs[i], role, name, depth + 1);
            if (found) return found;
        }
    } catch (e) {}
    return null;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/**
 * close-window: Find and close the gputrace window.
 */
function closeWindow(stem) {
    Application("Xcode").activate();
    delay(0.3);
    var win = findGpuTraceWindow(stem);
    if (!win) {
        return jsonResult({closed: false, reason: "not-found"});
    }
    try {
        // Click the close button (first button in window, which is the red close widget)
        var buttons = win.buttons();
        for (var i = 0; i < buttons.length; i++) {
            try {
                var sr = buttons[i].subrole();
                if (sr === "AXCloseButton") {
                    buttons[i].click();
                    return jsonResult({closed: true});
                }
            } catch (e) {}
        }
        // Fallback: look for button with close-related description
        for (var i = 0; i < buttons.length; i++) {
            try {
                var desc = buttons[i].description();
                if (desc && desc.toLowerCase().indexOf("close") >= 0) {
                    buttons[i].click();
                    return jsonResult({closed: true});
                }
            } catch (e) {}
        }
        // Last resort: click first button (standard macOS close button position)
        if (buttons.length > 0) {
            buttons[0].click();
            return jsonResult({closed: true, method: "first-button"});
        }
        return jsonResult({closed: false, reason: "no-close-button"});
    } catch (e) {
        return jsonResult({closed: false, reason: "error", error: e.message});
    }
}

/**
 * ensure-replay: Navigate to Replay/Profile controls, enable profiling, click Replay.
 */
function ensureReplay(stem) {
    Application("Xcode").activate();
    delay(0.3);
    var win = findGpuTraceWindow(stem);
    if (!win) {
        return jsonResult({replayed: false, profiled: false, error: "window-not-found"});
    }

    var profileCb = null;
    var replayBtn = null;

    // Try deterministic path first
    var container = navigateToEditorControls(win, stem);
    if (container) {
        try {
            var cbs = container.checkboxes.whose({name: "Profile after replay"})();
            if (cbs.length > 0) profileCb = cbs[0];
        } catch (e) {}
        try {
            var btns = container.buttons.whose({name: "Replay"})();
            if (btns.length > 0) replayBtn = btns[0];
        } catch (e) {}
    }

    // Fallback to recursive search if needed
    if (!replayBtn) {
        replayBtn = findElementRecursive(win, "button", "Replay", 0);
    }
    if (!profileCb) {
        profileCb = findElementRecursive(win, "checkbox", "Profile after replay", 0);
    }

    if (!replayBtn) {
        return jsonResult({replayed: false, profiled: false, error: "replay-button-not-found"});
    }

    // Check and enable profiling checkbox
    var profiled = false;
    if (profileCb) {
        try {
            if (profileCb.value() !== 1) {
                profileCb.click();
                profiled = true;
            } else {
                profiled = true;  // already enabled
            }
        } catch (e) {
            // Checkbox interaction failed — continue anyway
        }
    }

    // Click Replay
    try {
        replayBtn.click();
        return jsonResult({replayed: true, profiled: profiled, error: null});
    } catch (e) {
        return jsonResult({replayed: false, profiled: profiled, error: "replay-click-failed: " + e.message});
    }
}

/**
 * poll-activity: Check Xcode's Activity View for GPU profiling status.
 */
function pollActivity() {
    var wins;
    try { wins = xcode.windows(); } catch (e) {
        return jsonResult({active: false, status: "no-windows"});
    }

    for (var wi = 0; wi < wins.length; wi++) {
        var toolbars;
        try { toolbars = wins[wi].toolbars(); } catch (e) { continue; }
        for (var ti = 0; ti < toolbars.length; ti++) {
            var tGroups;
            try { tGroups = toolbars[ti].groups(); } catch (e) { continue; }
            for (var gi = 0; gi < tGroups.length; gi++) {
                try {
                    var desc = tGroups[gi].description();
                    if (!desc || desc.indexOf("Activity View") < 0) continue;

                    // Found Activity View — check for GPU-related status
                    var texts = [];
                    try {
                        var staticTexts = tGroups[gi].staticTexts();
                        for (var si = 0; si < staticTexts.length; si++) {
                            try { texts.push(staticTexts[si].value()); } catch (e) {}
                        }
                    } catch (e) {}

                    var statusText = texts.join(" | ");
                    var gpuKeywords = ["Debugging GPU", "Profiling", "GPU Workload", "Replaying"];
                    var isActive = false;
                    for (var ki = 0; ki < gpuKeywords.length; ki++) {
                        if (statusText.indexOf(gpuKeywords[ki]) >= 0) {
                            isActive = true;
                            break;
                        }
                    }

                    // Also check for MultiActionIndicator (activity spinner)
                    var hasIndicator = false;
                    try {
                        var btns = tGroups[gi].buttons();
                        for (var bi = 0; bi < btns.length; bi++) {
                            try {
                                var bName = btns[bi].name();
                                if (bName && bName.indexOf("Actions") >= 0) {
                                    hasIndicator = true;
                                    break;
                                }
                            } catch (e) {}
                        }
                    } catch (e) {}

                    return jsonResult({
                        active: isActive || hasIndicator,
                        status: statusText || "idle",
                        hasIndicator: hasIndicator
                    });
                } catch (e) { continue; }
            }
        }
    }
    return jsonResult({active: false, status: "activity-view-not-found"});
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
function run(_argv) {
    switch (action) {
        case "close-window":
            return closeWindow(stem);
        case "ensure-replay":
            return ensureReplay(stem);
        case "poll-activity":
            return pollActivity();
        default:
            return jsonResult({error: "unknown-action: " + action});
    }
}
