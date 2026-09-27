#!/usr/bin/env python3
"""Assemble a matched native MobileGym canary for the AgentNet warm child."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MOBILEGYM_REVISION = "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    tasks = report["task_results"]
    return {
        "task_count": report["task_count"],
        "success_rate": report["success_rate"],
        "successful_tasks": sum(bool(task["passed"]) for task in tasks),
        "progress_sum": sum(float(task["progress"]) for task in tasks),
        "model_invocations": sum(int(task["model_invocations"]) for task in tasks),
        "tool_counts": report["tool_counts"],
        "tasks": [
            {
                "task_id": task["task_id"],
                "suite": task["suite"],
                "passed": task["passed"],
                "progress": task["progress"],
                "steps": task["steps"],
                "tool_names": task["tool_names"],
                "judge_issue_fields": task["judge"]["issue_fields"],
            }
            for task in tasks
        ],
    }


def _validate(label: str, report: dict[str, Any]) -> None:
    if report["kind"] != "localagent_mobilegym_native_text_eval":
        raise ValueError(f"unexpected {label} report kind")
    if report["benchmark_id"] != "mobilegym":
        raise ValueError(f"{label} benchmark mismatch")
    if report["source"]["revision"] != MOBILEGYM_REVISION:
        raise ValueError(f"{label} MobileGym revision mismatch")
    if report["official_split"] != "test" or not report["official_split_verified"]:
        raise ValueError(f"{label} is not bound to the public MobileGym test split")
    if report["task_count"] != 4 or report["run"]["limit"] != 4:
        raise ValueError(f"{label} canary size mismatch")
    if report["run"]["start"] != 0 or report["run"]["max_steps"] != 2:
        raise ValueError(f"{label} protocol mismatch")
    if report["run"]["selector_top_m"] != 1 or report["run"]["selector_first"]:
        raise ValueError(f"{label} selector policy mismatch")
    if report["native_receipt_eligible"] or report["errors"]:
        raise ValueError(f"{label} must remain a clean non-eligible canary")


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
    if parent["checkpoint_sha256"] != PARENT_SHA256:
        raise ValueError("parent checkpoint mismatch")
    parent_summary = _summary(parent)
    warm_summary = _summary(warm)
    body: dict[str, Any] = {
        "kind": "localagent_mobilegym_agentnet_warm_parent_canary",
        "schema_version": 1,
        "benchmark_id": "mobilegym",
        "source": {
            "repository": parent["source"]["repository"],
            "revision": MOBILEGYM_REVISION,
            "test_split_sha256": parent["source"]["test_split_sha256"],
            "train_split_sha256": parent["source"]["train_split_sha256"],
            "test_ids_sha256": parent["source"]["test_ids_sha256"],
            "official_test_task_count": parent["official_test_task_count"],
        },
        "protocol": {
            "task_count": 4,
            "start": 0,
            "limit": 4,
            "max_steps": 2,
            "selector_top_m": 1,
            "selector_first": False,
            "official_split": "test",
            "official_split_verified": True,
            "native_receipt_eligible": False,
            "observations": "bounded_DOM_text_projection",
            "vision_used": False,
        },
        "parent": {"checkpoint_sha256": parent["checkpoint_sha256"], "result": parent_summary},
        "warm": {"checkpoint_sha256": warm["checkpoint_sha256"], "result": warm_summary},
        "comparison": {
            "parent_success_rate": parent_summary["success_rate"],
            "warm_success_rate": warm_summary["success_rate"],
            "parent_progress_sum": parent_summary["progress_sum"],
            "warm_progress_sum": warm_summary["progress_sum"],
            "parent_tool_counts": parent_summary["tool_counts"],
            "warm_tool_counts": warm_summary["tool_counts"],
            "warm_minus_parent_success_rate": warm_summary["success_rate"] - parent_summary["success_rate"],
        },
        "decision": {
            "adoption": "retain_as_offline_initialization_candidate_but_not_native_mobile_promotion",
            "native_replay_required": True,
            "webgpu_export_allowed": False,
            "reason": (
                "The AgentNet warm child executes cleanly on all four pinned MobileGym canary tasks, "
                "but matches the current parent at 0/4 success, 0.0 aggregate progress, and only "
                "mobile_input_text actions. Its desktop text-projection gain does not transfer to "
                "native mobile state completion."
            ),
        },
        "source_artifacts": {
            "parent_report": _identity(args.parent_report),
            "warm_report": _identity(args.warm_report),
            "runner": _identity(args.runner),
        },
        "claim_boundary": (
            "Matched native MobileGym limit-4 diagnostic over pinned public test IDs using a bounded "
            "DOM/text observation projection. It is not the complete 256-task score, visual mobile "
            "agent result, Android emulator result, or real-account execution."
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
