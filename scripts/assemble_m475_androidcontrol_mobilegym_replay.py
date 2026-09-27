#!/usr/bin/env python3
"""Assemble the bounded native MobileGym replay for Android-Control transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
WARM_SHA256 = "939547bd3955488794efca2dfb994ef5fd793df80d9aeebbd33b45aeb97e6ff9"
MOBILEGYM_REVISION = "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
TASK_IDS = [
    "account.Railway12306ChangePassword",
    "account.Railway12306ForgotPasswordReset",
    "account.Railway12306LoginWithAccount",
    "account.Railway12306RegisterThenLogin",
]


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _task_summary(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tasks = report.get("task_results", [])
    if [task.get("task_id") for task in tasks] != TASK_IDS:
        raise ValueError("MobileGym canary task order mismatch")
    return {
        task["task_id"]: {
            "passed": task["passed"],
            "progress": task["progress"],
            "steps": task["steps"],
            "tool_names": task["tool_names"],
            "trace_sha256": task["trace_sha256"],
        }
        for task in tasks
    }


def assemble(parent: dict[str, Any], warm: dict[str, Any], *, parent_path: Path, warm_path: Path) -> dict[str, Any]:
    if parent["checkpoint_sha256"] != PARENT_SHA256:
        raise ValueError("parent replay is not bound to the current checkpoint")
    if warm["checkpoint_sha256"] != WARM_SHA256:
        raise ValueError("warm replay is not bound to the Android-Control child")
    for label, report in (("parent", parent), ("warm", warm)):
        if report["source"]["revision"] != MOBILEGYM_REVISION:
            raise ValueError(f"{label} MobileGym revision mismatch")
        if report["task_count"] != 4 or report["official_split"] != "test":
            raise ValueError(f"{label} replay is not the bounded official-test canary")
        if report["run"]["max_steps"] != 2 or report["run"]["limit"] != 4:
            raise ValueError(f"{label} replay protocol mismatch")
        if report["vision_used"] or report["observation_mode"] != "text_projection":
            raise ValueError("replay must retain the text-only observation boundary")
    parent_tasks = _task_summary(parent)
    warm_tasks = _task_summary(warm)
    body: dict[str, Any] = {
        "kind": "localagent_androidcontrol_mobilegym_replay_receipt",
        "schema_version": 1,
        "benchmark_id": "mobilegym",
        "source": warm["source"],
        "protocol": {
            "official_split": "test",
            "official_split_verified": True,
            "task_ids": TASK_IDS,
            "task_count": 4,
            "max_steps": 2,
            "observation_mode": "text_projection",
            "vision_used": False,
            "native_receipt_eligible": False,
        },
        "inputs": {
            "parent_report": _identity(parent_path),
            "warm_report": _identity(warm_path),
        },
        "parent": {
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "success_rate": parent["success_rate"],
            "progress_sum": sum(float(item["progress"]) for item in parent["task_results"]),
            "tool_counts": parent["tool_counts"],
            "tasks": parent_tasks,
        },
        "warm": {
            "checkpoint_sha256": warm["checkpoint_sha256"],
            "success_rate": warm["success_rate"],
            "progress_sum": sum(float(item["progress"]) for item in warm["task_results"]),
            "tool_counts": warm["tool_counts"],
            "tasks": warm_tasks,
        },
        "decision": {
            "adopt_androidcontrol_child_for_native_mobile": False,
            "webgpu_export_allowed": False,
            "reason": (
                "The Android-Control warm child matches the current parent at 0/4 MobileGym "
                "success, zero progress, and the same mobile_input_text-only collapse. The "
                "offline text-projection gain therefore does not transfer to native state "
                "completion; keep the child as an initialization diagnostic only."
            ),
        },
        "claim_boundary": (
            "Bounded native MobileGym simulator/state-diff replay over four official test IDs. "
            "This is not the complete 256-task score, visual grounding, Android emulator "
            "evaluation, or real-account execution."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body = assemble(
        _load(args.parent_report),
        _load(args.warm_report),
        parent_path=args.parent_report,
        warm_path=args.warm_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(body["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
