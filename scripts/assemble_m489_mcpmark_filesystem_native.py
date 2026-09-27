#!/usr/bin/env python3
"""Seal the targeted MCPMark filesystem head-adaptation and native verifier replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assemble(*, continuation: Path, heads: Path, native: Path) -> dict[str, Any]:
    continuation_payload = _load(continuation)
    heads_payload = _load(heads)
    native_payload = _load(native)
    if continuation_payload.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("continuation report kind mismatch")
    if heads_payload.get("kind") != "localagent_mcpmark_filesystem_head_adaptation":
        raise ValueError("head report kind mismatch")
    if native_payload.get("kind") != "localagent_mcpmark_native_filesystem_current_checkpoint":
        raise ValueError("native report kind mismatch")
    if native_payload.get("benchmark_id") != "mcpmark":
        raise ValueError("native benchmark mismatch")
    if native_payload.get("dataset", {}).get("revision") != SOURCE_REVISION:
        raise ValueError("native source revision mismatch")
    if heads_payload.get("source", {}).get("revision") != SOURCE_REVISION:
        raise ValueError("head source revision mismatch")
    if heads_payload["parent"]["sha256"] != continuation_payload["child"]["sha256"]:
        raise ValueError("head adaptation is not based on the continuation child")
    if native_payload["model"]["sha256"] != heads_payload["child"]["sha256"]:
        raise ValueError("native replay is not bound to the head-adapted child")
    if native_payload["environment"].get("mcp_server_executed") is not True:
        raise ValueError("MCP server was not executed")
    if native_payload["rollout"].get("verifier_exit_code") != 0:
        raise ValueError("native verifier did not pass")
    turns = native_payload["rollout"].get("turns", [])
    if [turn.get("tool") for turn in turns] != ["directory_tree", "write_file"]:
        raise ValueError("unexpected native tool sequence")
    payload: dict[str, Any] = {
        "kind": "localagent_mcpmark_filesystem_native_head_adaptation_receipt",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "environment_executed": True,
        "official_split_verified": False,
        "task_count": 1,
        "success_rate": 1.0,
        "source": {
            "dataset": "MCPMark",
            "url": "https://github.com/eval-sys/mcpmark",
            "revision": SOURCE_REVISION,
            "suite": native_payload["dataset"].get("suite"),
            "task": native_payload["dataset"].get("task"),
            "official_split_verified": False,
        },
        "training": {
            "continuation_report": _identity(continuation),
            "continuation": {
                "parent": continuation_payload["parent"],
                "child": continuation_payload["child"],
                "rows": continuation_payload["rows"],
                "hyperparameters": continuation_payload["hyperparameters"],
                "before_eval": continuation_payload["before"]["eval"],
                "after_eval": continuation_payload["after"]["eval"],
            },
            "head_report": _identity(heads),
            "head_adaptation": {
                "parent": heads_payload["parent"],
                "child": heads_payload["child"],
                "hyperparameters": heads_payload["hyperparameters"],
                "before_eval": heads_payload["before_eval"],
                "after_eval": heads_payload["after_eval"],
                "head_movement": heads_payload["head_movement"],
            },
        },
        "native": {
            "receipt": _identity(native),
            "model": native_payload["model"],
            "environment": native_payload["environment"],
            "task_source": native_payload["task_source"],
            "turns": turns,
            "verifier_exit_code": native_payload["rollout"]["verifier_exit_code"],
            "model_completed_task": native_payload["rollout"]["model_completed_task"],
        },
        "decision": {
            "native_task_pass": True,
            "official_split_verified": False,
            "promotion": "blocked_pending_official_mcpmark_split_and_broader_tasks",
            "claim_boundary": "One isolated MCPMark easy filesystem task passed its real stdio-server and independent verifier replay after public filesystem continuation, frozen-backbone head adaptation, and generic path/operation grounding. This is not an official MCPMark split, leaderboard score, multi-service result, email/Notion account side effect, or publication approval.",
        },
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = assemble(continuation=args.continuation, heads=args.heads, native=args.native)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": payload["receipt_self_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
