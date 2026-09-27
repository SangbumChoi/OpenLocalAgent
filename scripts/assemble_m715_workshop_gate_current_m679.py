#!/usr/bin/env python3
"""Refresh the fail-closed m679 gate with structured visual action evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.prior.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    current = identity(args.checkpoint)
    if gate.get("current_checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("prior gate checkpoint mismatch")
    if pilot.get("kind") != "localagent_m714_androidcontrol_structured_visual_pilot":
        raise ValueError("unexpected structured visual pilot kind")
    blocked = dict(gate.get("blocked_requirements", {}))
    blocked["native:androidcontrol_structured_visual"] = "structured_action_head_native_emulator_unverified"
    evidence = dict(gate.get("evidence", {}))
    evidence["androidcontrol_structured_visual_pilot"] = identity(args.pilot)
    payload: dict[str, Any] = {
        "kind": "localagent_m715_workshop_gate_current_m679",
        "schema_version": 1,
        "ready": False,
        "current_checkpoint": current,
        "passed_requirements": sorted(set(gate.get("passed_requirements", []))),
        "blocked_requirements": dict(sorted(blocked.items())),
        "evidence": evidence,
        "claim_boundary": (
            "Fail-closed gate after structured screenshot-conditioned action-type and pointer pilot. "
            "Held-out action accuracy is diagnostic only; no native emulator verifier, visual WebGPU "
            "export, or official benchmark score is bound."
        ),
        "source_gate": identity(args.prior),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ready": payload["ready"], "blockers": len(blocked)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
