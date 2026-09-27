#!/usr/bin/env python3
"""Assemble a matched native BrowserGym canary for the AgentNet warm child."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BROWSERGYM_REVISION = "9e779f087de9a65668b6974d11f9ce9816026e96"
MINIWOB_REVISION = "7fd85d71a4b60325c6585396ec4f48377d049838"
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    cases = report["cases"]
    steps = [step for case in cases for step in case["steps"]]
    return {
        "task_count": report["task_count"],
        "success_rate": report["success_rate"],
        "successful_episodes": sum(bool(case["success"]) for case in cases),
        "steps": len(steps),
        "grounded_steps": sum(bool(step["grounded"]) for step in steps),
        "noop_actions": sum(step["action"] == "noop(0)" for step in steps),
        "tasks": [{"task": case["task"], "seed": case["seed"], "success": case["success"]} for case in cases],
    }


def _validate(label: str, report: dict[str, Any]) -> None:
    if report["kind"] != "localagent_browsergym_native_eval":
        raise ValueError(f"unexpected {label} report kind")
    if report["benchmark_id"] != "browsergym_miniwob":
        raise ValueError(f"{label} benchmark mismatch")
    if report["browsergym"]["revision"] != BROWSERGYM_REVISION:
        raise ValueError(f"{label} BrowserGym revision mismatch")
    if report["runtime"]["miniwob_revision"] != MINIWOB_REVISION:
        raise ValueError(f"{label} MiniWoB revision mismatch")
    if report["task_plan"]["fixed_seeds"] != [11, 17, 23, 29]:
        raise ValueError(f"{label} seed plan mismatch")
    if report["task_count"] != 4 or report["task_plan"]["selected_limit"] != 4:
        raise ValueError(f"{label} canary size mismatch")
    if report["official_split_verified"] or report["coordinate_fallback"]:
        raise ValueError(f"{label} must remain a non-official limit-4 diagnostic")
    if report["tool_pool"] != "standard" or not report["environment_executed"]:
        raise ValueError(f"{label} tool/environment contract mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.parent_report.read_text(encoding="utf-8"))
    warm = json.loads(args.warm_report.read_text(encoding="utf-8"))
    _validate("parent", parent)
    _validate("warm", warm)
    if parent["checkpoint"]["sha256"] != PARENT_SHA256:
        raise ValueError("parent checkpoint mismatch")
    parent_summary = _summary(parent)
    warm_summary = _summary(warm)
    body: dict[str, Any] = {
        "kind": "localagent_browsergym_agentnet_warm_parent_canary",
        "schema_version": 1,
        "benchmark_id": "browsergym_miniwob",
        "source": {
            "browsergym_revision": BROWSERGYM_REVISION,
            "miniwob_revision": MINIWOB_REVISION,
            "chromium_revision": parent["runtime"]["chromium_revision"],
            "chromium_version": parent["runtime"]["chromium_version"],
            "playwright_version": parent["runtime"]["playwright_version"],
            "browser_executable_sha256": parent["runtime"]["browser_executable_sha256"],
        },
        "protocol": {
            "task_count": 4,
            "fixed_seeds": [11, 17, 23, 29],
            "max_steps": 10,
            "tool_pool": "standard",
            "coordinate_fallback": False,
            "observations": "accessibility_tree_and_DOM_text",
            "official_split_verified": False,
        },
        "parent": {"checkpoint": parent["checkpoint"], "result": parent_summary},
        "warm": {"checkpoint": warm["checkpoint"], "result": warm_summary},
        "comparison": {
            "parent_success_rate": parent_summary["success_rate"],
            "warm_success_rate": warm_summary["success_rate"],
            "parent_grounded_steps": parent_summary["grounded_steps"],
            "warm_grounded_steps": warm_summary["grounded_steps"],
            "parent_noop_actions": parent_summary["noop_actions"],
            "warm_noop_actions": warm_summary["noop_actions"],
            "warm_minus_parent_success_rate": warm_summary["success_rate"] - parent_summary["success_rate"],
        },
        "decision": {
            "adoption": "retain_as_offline_initialization_candidate_but_not_native_browser_promotion",
            "native_replay_required": True,
            "webgpu_export_allowed": False,
            "reason": (
                "The AgentNet warm child executes all four pinned BrowserGym/MiniWoB canary episodes, "
                "but matches the current parent at 0/4 success, 0 grounded steps, and 40/40 no-op "
                "actions. Its offline text gain does not transfer to native browser control."
            ),
        },
        "source_artifacts": {
            "parent_report": _identity(args.parent_report),
            "warm_report": _identity(args.warm_report),
            "runner": _identity(args.runner),
        },
        "claim_boundary": (
            "Matched native BrowserGym/MiniWoB limit-4 diagnostic over accessibility/DOM text. "
            "It is not the official 240-episode score, visual computer-use result, WebArena result, "
            "or live email/Notion account execution."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(body["comparison"], indent=2, sort_keys=True))
    print(json.dumps(body["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
