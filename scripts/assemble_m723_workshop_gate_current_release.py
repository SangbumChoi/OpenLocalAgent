#!/usr/bin/env python3
"""Refresh the release gate with current-checkpoint visual transfer evidence."""

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
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate = json.loads(args.prior.read_text(encoding="utf-8"))
    evidence_receipt = json.loads(args.evidence.read_text(encoding="utf-8"))
    current = identity(args.checkpoint)
    if evidence_receipt.get("checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("visual evidence is not bound to the supplied current checkpoint")
    if evidence_receipt.get("kind") != "localagent_m722_current_checkpoint_structured_visual":
        raise ValueError("unexpected visual evidence receipt kind")
    blocked = dict(gate.get("blocked_requirements", {}))
    blocked["artifacts:visual_webgpu_export"] = (
        "current_checkpoint_visual_cpu_parity_but_no_warm_action_gain_pointer_regression_and_browser_native_verifier_missing"
    )
    evidence = dict(gate.get("evidence", {}))
    evidence["current_checkpoint_structured_visual"] = identity(args.evidence)
    payload: dict[str, Any] = {
        "kind": "localagent_m723_workshop_gate_current_release",
        "schema_version": 1,
        "ready": False,
        "current_checkpoint": current,
        "passed_requirements": sorted(set(gate.get("passed_requirements", []))),
        "blocked_requirements": dict(sorted(blocked.items())),
        "evidence": evidence,
        "claim_boundary": (
            "Fail-closed release gate after binding the current checkpoint to a trained visual sidecar "
            "and CPU ONNX parity. The visual warm arm ties random on action accuracy and regresses "
            "pointer MAE; browser execution of this current graph and native mobile verification remain absent."
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
