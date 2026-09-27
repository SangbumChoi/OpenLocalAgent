#!/usr/bin/env python3
"""Seal a bounded current-checkpoint BrowserGym/MiniWoB native receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    current = identity(args.checkpoint)
    if raw.get("checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("native BrowserGym receipt is not bound to the supplied checkpoint")
    if raw.get("environment_executed") is not True:
        raise ValueError("native BrowserGym environment did not execute")
    cases = raw.get("cases", [])
    payload: dict[str, Any] = {
        "kind": "localagent_m726_current_browsergym_native",
        "schema_version": 1,
        "benchmark_id": raw.get("benchmark_id"),
        "checkpoint": current,
        "environment_executed": True,
        "official_split_verified": bool(raw.get("official_split_verified")),
        "task_count": len(cases),
        "success_count": sum(bool(case.get("success")) for case in cases if isinstance(case, dict)),
        "success_rate": raw.get("success_rate"),
        "task_plan": raw.get("task_plan"),
        "runtime": raw.get("runtime"),
        "raw_report": identity(args.raw),
        "claim_boundary": (
            "Bounded native BrowserGym/MiniWoB checkpoint-in-the-loop evaluation over eight pinned "
            "episodes using accessibility-tree text. This is not visual computer use, Android/" 
            "emulator control, WebArena, or real-account email/Notion execution."
        ),
        "decision": {
            "native_browser_promotion": False,
            "reason": "0/8 bounded tasks succeeded; retain as a current-checkpoint text negative control",
        },
    }
    payload["receipt_self_sha256"] = self_hash(payload)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"success_rate": payload["success_rate"], "task_count": payload["task_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
