"""Fail-closed readiness checks for the source-linked realistic-agent catalog.

The catalog describes many environments that cannot be executed by the small text-first WebGPU
model without an emulator, VM, MCP server, or visual adapter.  This module reports those gaps
without downloading anything or treating an installed Python package as an official benchmark
run.  A row is ``runnable`` only when its catalog integration status is already supported by a
local adapter; pending rows remain blocked even if a related dependency happens to be installed.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

from localagent.data.realistic_catalog import load_catalog

_SUPPORTED_STATUSES = frozenset({"supported", "supported_text_first_pilot"})


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _dependency_probes() -> dict[str, bool]:
    """Probe optional dependencies without importing or mutating their environments."""

    modules = (
        "android_world",
        "browsergym",
        "datasets",
        "gymnasium",
        "huggingface_hub",
        "mcpmark",
        "osworld",
        "playwright",
    )
    commands = ("adb", "docker", "osascript", "qemu-system-x86_64")
    return {
        **{f"module:{name}": _module_available(name) for name in modules},
        **{f"command:{name}": shutil.which(name) is not None for name in commands},
    }


def _row_blockers(row: dict[str, Any], probes: dict[str, bool]) -> list[str]:
    status = row["integration"]["status"]
    blockers: list[str] = []
    if status not in _SUPPORTED_STATUSES:
        blockers.append(f"integration_status:{status}")

    family = row["family"]
    if status == "prompt_capture_supported":
        # The capture command also requires pristine upstream checkouts and a pinned browser
        # executable; package presence alone is intentionally insufficient.
        blockers.extend(
            missing
            for missing, present in (
                ("module:playwright", probes["module:playwright"]),
                ("browsergym_checkout", False),
                ("miniwob_checkout", False),
                ("pinned_playwright_chromium", False),
            )
            if not present
        )
    elif family == "mobile" and status in {
        "environment_runner_pending",
        "simulator_runner_pending",
        "emulator_runner_pending",
        "Android_runner_pending",
        "APK_runner_pending",
    }:
        if status in {
            "environment_runner_pending",
            "emulator_runner_pending",
            "Android_runner_pending",
            "APK_runner_pending",
        }:
            if not probes["command:adb"]:
                blockers.append("command:adb")
        else:
            blockers.append("mobile_simulator_adapter")
    elif family == "browser" and status in {
        "environment_runner_pending",
        "BrowserGym_runner_pending",
    }:
        if not probes["module:browsergym"]:
            blockers.append("module:browsergym")
        if not probes["module:playwright"]:
            blockers.append("module:playwright")
    elif family == "computer" and status in {
        "vm_runner_pending",
        "VM_runner_pending",
        "Docker_runner_pending",
    }:
        if not (probes["command:docker"] or probes["command:qemu-system-x86_64"]):
            blockers.append("command:docker_or_qemu-system-x86_64")
    elif family == "tool_api" and status in {"MCP_runner_pending", "scenario_runner_pending"}:
        if not probes["command:docker"]:
            blockers.append("command:docker")
    elif status == "container_runner_pending" and not probes["command:docker"]:
        blockers.append("command:docker")
    elif status == "macos_runner_pending" and not probes["command:osascript"]:
        blockers.append("command:osascript")
    return sorted(set(blockers))


def preflight_catalog(path: str | Path) -> dict[str, Any]:
    """Return a deterministic readiness report for every catalog row."""

    catalog, fingerprint = load_catalog(path)
    probes = _dependency_probes()
    rows: list[dict[str, Any]] = []
    for row in catalog["entries"]:
        row_dict = dict(row)
        blockers = _row_blockers(row_dict, probes)
        rows.append(
            {
                "id": row_dict["id"],
                "family": row_dict["family"],
                "train_policy": row_dict["train_policy"],
                "integration_status": row_dict["integration"]["status"],
                "runnable": not blockers,
                "blockers": blockers,
                "source_url": row_dict["source_url"],
            }
        )
    runnable = [row["id"] for row in rows if row["runnable"]]
    blocked = [row["id"] for row in rows if not row["runnable"]]
    return {
        "kind": "localagent_realistic_agent_preflight",
        "schema_version": 1,
        "catalog_sha256": fingerprint,
        "catalog_entries": len(rows),
        "dependency_probes": probes,
        "runnable_ids": runnable,
        "blocked_ids": blocked,
        "counts": {
            "runnable": len(runnable),
            "blocked": len(blocked),
            "train_rows": sum(row["train_policy"] == "train" for row in rows),
            "evaluation_or_restricted_rows": sum(row["train_policy"] != "train" for row in rows),
        },
        "rows": rows,
    }


def json_report(path: str | Path) -> str:
    """Render a stable JSON report suitable for a receipt or CI artifact."""

    return json.dumps(preflight_catalog(path), indent=2, sort_keys=True) + "\n"
