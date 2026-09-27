#!/usr/bin/env python3
"""Bind the fail-closed workshop gate to the current m624 WebGPU candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CHECKPOINT_SHA256 = "984152a802357e18387a6a28c93c9d30f43c5b0c9e9fede48caa24157716b43b"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def assemble(gate_path: Path, adoption_path: Path, transfer_path: Path) -> dict[str, Any]:
    gate = _load(gate_path)
    adoption = _load(adoption_path)
    transfer = _load(transfer_path)
    if gate.get("kind") != "localagent_workshop_publication_gate":
        raise ValueError("unexpected workshop gate kind")
    if gate.get("ready") is not False:
        raise ValueError("current gate must remain fail-closed")
    if gate.get("current_checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("gate is not bound to the m624 checkpoint")
    if adoption.get("kind") != "localagent_m625_mcp_trajectory_webgpu_adoption":
        raise ValueError("unexpected WebGPU adoption receipt kind")
    if adoption.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("adoption receipt checkpoint mismatch")
    if transfer.get("kind") != "localagent_m626_androidcontrol_warm_random_transfer":
        raise ValueError("unexpected AndroidControl transfer receipt kind")
    if transfer.get("parent_checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("transfer receipt parent mismatch")
    checks = {row["requirement"]: row for row in gate.get("checks", [])}
    payload: dict[str, Any] = {
        "kind": "localagent_m627_workshop_gate_current_m624",
        "schema_version": 1,
        "current_checkpoint": gate["current_checkpoint"],
        "gate": gate,
        "gate_input": _identity(gate_path),
        "adoption_receipt": _identity(adoption_path),
        "androidcontrol_transfer_receipt": _identity(transfer_path),
        "decision": {
            "ready": False,
            "passed_checks": sorted(
                requirement for requirement, check in checks.items() if check.get("status") == "pass"
            ),
            "blocking_requirements": gate["blocking_requirements"],
            "webgpu_and_weight_evidence_current": (
                checks.get("webgpu:native_capability_and_latency", {}).get("status") == "pass"
                and checks.get("weights:transfer_and_no_transfer_ablation", {}).get("status") == "pass"
            ),
        },
        "claim_boundary": (
            "This is a current-checkpoint fail-closed publication gate. The m624 WebGPU capability "
            "and m626 transfer ablation pass, but missing native mobile/browser/desktop/tool suites, "
            "RL preflight, and authenticated public model/demo artifacts prevent workshop approval."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    receipt = assemble(args.gate, args.adoption, args.transfer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
