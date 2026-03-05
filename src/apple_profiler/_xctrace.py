"""Thin subprocess wrapper around `xcrun xctrace`."""

from __future__ import annotations

import subprocess
from pathlib import Path


class XctraceError(Exception):
    """Raised when xctrace returns a non-zero exit code."""

    def __init__(self, returncode: int, stderr: str):
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"xctrace failed (rc={returncode}): {stderr}")


def _run(args: list[str], timeout: float | None = 60) -> str:
    """Run an xctrace command and return stdout."""
    cmd = ["xcrun", "xctrace", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise XctraceError(result.returncode, result.stderr.strip())
    return result.stdout


def export_toc(trace_path: Path | str) -> str:
    """Run `xctrace export --toc`, return raw XML string."""
    return _run(["export", "--input", str(trace_path), "--toc"])


def export_xpath(trace_path: Path | str, xpath: str) -> str:
    """Run `xctrace export --xpath`, return raw XML string."""
    return _run(["export", "--input", str(trace_path), "--xpath", xpath])


def export_table(
    trace_path: Path | str,
    schema: str,
    *,
    run: int = 1,
    target_pid: str | None = None,
    **extra_attrs: str,
) -> str:
    """Build an xpath for a table schema and export it.

    Args:
        trace_path: Path to the .trace file.
        schema: The schema name (e.g., "cpu-profile", "potential-hangs").
        run: The run number (default 1).
        target_pid: If set, adds target-pid attribute to the xpath.
        **extra_attrs: Additional attributes to filter the table element.

    Returns:
        Raw XML string from xctrace export.
    """
    predicates = [f'@schema="{schema}"']
    if target_pid is not None:
        predicates.append(f'@target-pid="{target_pid}"')
    for key, value in extra_attrs.items():
        predicates.append(f'@{key}="{value}"')

    pred_str = " and ".join(predicates)
    xpath = f'/trace-toc/run[@number="{run}"]/data/table[{pred_str}]'
    return export_xpath(trace_path, xpath)


def list_instruments() -> list[str]:
    """Run `xctrace list instruments` and return instrument names."""
    output = _run(["list", "instruments"])
    lines = output.strip().splitlines()
    # Skip header line(s) and return instrument names
    instruments: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("=="):
            instruments.append(stripped)
    return instruments


def record(
    instrument: str,
    *,
    output: Path | str,
    attach: str | None = None,
    pid: int | None = None,
    all_processes: bool = False,
    time_limit: str | None = None,
    device: str | None = None,
    no_prompt: bool = True,
    template: str | None = None,
    timeout: float | None = None,
) -> Path:
    """Run `xctrace record`.

    Args:
        instrument: Instrument name to record with.
        output: Output .trace file path.
        attach: Process name to attach to.
        pid: Process ID to attach to.
        all_processes: Record all processes.
        time_limit: Time limit string (e.g., "3s", "10s").
        device: Device UUID or name.
        no_prompt: Don't prompt for confirmation.
        template: Template name (default: uses instrument directly).
        timeout: Subprocess timeout in seconds.

    Returns:
        Path to the output .trace file.
    """
    args = ["record", "--output", str(output)]

    if template:
        args.extend(["--template", template])

    args.extend(["--instrument", instrument])

    if attach is not None:
        args.extend(["--attach", attach])
    elif pid is not None:
        args.extend(["--attach", str(pid)])
    elif all_processes:
        args.append("--all-processes")

    if time_limit:
        args.extend(["--time-limit", time_limit])
    if device:
        args.extend(["--device", device])
    if no_prompt:
        args.append("--no-prompt")

    _run(args, timeout=timeout)
    return Path(output)
