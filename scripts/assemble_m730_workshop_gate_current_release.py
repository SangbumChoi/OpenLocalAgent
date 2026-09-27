#!/usr/bin/env python3
"""Refresh the release gate with current-checkpoint ToolSandbox native evidence."""

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
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--toolsandbox", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.prior.read_text(encoding="utf-8"))
    evidence = json.loads(args.toolsandbox.read_text(encoding="utf-8"))
    current = identity(args.checkpoint)
    if evidence.get("checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("ToolSandbox evidence is not bound to current checkpoint")
    blocked = dict(gate.get("blocked_requirements", {}))
    blocked["native:toolsandbox"] = "single_step_3_of_3_but_bounded_interactive_0_of_3_and_official_split_unverified"
    records = dict(gate.get("evidence", {}))
    records["current_checkpoint_toolsandbox_native"] = identity(args.toolsandbox)
    payload: dict[str, Any] = {
        "kind": "localagent_m730_workshop_gate_current_release",
        "schema_version": 1,
        "ready": False,
        "current_checkpoint": current,
        "passed_requirements": sorted(set(gate.get("passed_requirements", []))),
        "blocked_requirements": dict(sorted(blocked.items())),
        "evidence": records,
        "claim_boundary": (
            "Fail-closed current-release gate after current WebGPU visual probing, native BrowserGym "
            "text diagnostics, and current-checkpoint ToolSandbox simulator evidence. ToolSandbox "
            "single-step fixtures pass 3/3, but bounded continuation is 0/3; Android/desktop visual "
            "verifiers, official task scores, and HF publication remain absent."
        ),
        "source_gate": identity(args.prior),
    }
    payload["receipt_self_sha256"] = self_hash(payload)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ready": payload["ready"], "blockers": len(blocked)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
