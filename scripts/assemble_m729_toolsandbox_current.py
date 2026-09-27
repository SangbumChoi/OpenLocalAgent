#!/usr/bin/env python3
"""Seal paired current-checkpoint ToolSandbox native smoke and continuation receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_REVISION = "165848b9a78cead7ca7fe7c89c688b58e6501219"


def identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path: Path, checkpoint_sha: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("kind") != "localagent_toolsandbox_native_smoke":
        raise ValueError(f"unexpected ToolSandbox receipt kind: {path}")
    if report.get("source_revision") != SOURCE_REVISION:
        raise ValueError("ToolSandbox source revision mismatch")
    if report.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("ToolSandbox receipt is not bound to the current checkpoint")
    if report.get("environment_executed") is not True or report.get("verifier_executed") is not True:
        raise ValueError("ToolSandbox simulator/verifier did not execute")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--interactive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = identity(args.checkpoint)
    single = load(args.single, checkpoint["sha256"])
    interactive = load(args.interactive, checkpoint["sha256"])
    payload: dict[str, Any] = {
        "kind": "localagent_m729_current_toolsandbox_native",
        "schema_version": 1,
        "benchmark_id": "toolsandbox",
        "checkpoint": checkpoint,
        "source": {
            "url": single.get("source_url"),
            "revision": single.get("source_revision"),
        },
        "environment_executed": True,
        "verifier_executed": True,
        "external_api_called": bool(single.get("external_api_called") or interactive.get("external_api_called")),
        "official_split_verified": False,
        "single_step": {
            "task_count": single.get("task_count"),
            "success_count": single.get("success_count"),
            "success_rate": single.get("success_rate"),
            "protocol": single.get("protocol"),
            "raw_report": identity(args.single),
        },
        "interactive": {
            "task_count": interactive.get("task_count"),
            "success_count": interactive.get("success_count"),
            "success_rate": interactive.get("success_rate"),
            "protocol": interactive.get("protocol"),
            "max_agent_turns": interactive.get("max_agent_turns"),
            "raw_report": identity(args.interactive),
        },
        "decision": {
            "native_tool_use_verified": single.get("success_rate") == 1.0,
            "multi_turn_tool_use_verified": interactive.get("success_rate") == 1.0,
            "promote_checkpoint": False,
            "reason": (
                "Single-step fixtures pass, but bounded continuation reaches only 0/3 exact "
                "milestones; official split and model-based user simulator were not executed."
            ),
        },
        "claim_boundary": (
            "Current-checkpoint native ToolSandbox simulator/verifier smoke only: three simple "
            "single-step fixtures and a bounded scripted continuation. No official split, upstream "
            "user simulator, full scenario matrix, RapidAPI service, real email/Notion account, or "
            "leaderboard score is claimed."
        ),
    }
    payload["receipt_self_sha256"] = self_hash(payload)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"single_step": payload["single_step"]["success_rate"], "interactive": payload["interactive"]["success_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
