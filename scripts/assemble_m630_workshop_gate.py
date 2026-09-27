#!/usr/bin/env python3
"""Bind a current m626 workshop gate with native MobileGym evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CHECKPOINT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


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


def assemble(gate_path: Path, adoption_path: Path, mobilegym_path: Path, transfer_path: Path) -> dict[str, Any]:
    gate = _load(gate_path)
    adoption = _load(adoption_path)
    mobilegym = _load(mobilegym_path)
    transfer = _load(transfer_path)
    if gate.get("kind") != "localagent_workshop_publication_gate" or gate.get("ready") is not False:
        raise ValueError("gate must be a fail-closed workshop gate")
    if gate.get("current_checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("gate is not bound to m626")
    if adoption.get("kind") != "localagent_m628_androidcontrol_webgpu_adoption":
        raise ValueError("unexpected WebGPU adoption receipt")
    if adoption.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("WebGPU adoption checkpoint mismatch")
    if mobilegym.get("benchmark_id") != "mobilegym":
        raise ValueError("MobileGym benchmark id missing")
    if mobilegym.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("MobileGym checkpoint mismatch")
    if transfer.get("kind") not in {
        "localagent_m626_androidcontrol_warm_random_transfer",
        "localagent_m631_m626_mcp_warm_random_transfer",
    }:
        raise ValueError("unexpected transfer receipt")
    if transfer.get("parent_checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("transfer parent checkpoint mismatch")
    checks = {row["requirement"]: row for row in gate.get("checks", [])}
    payload: dict[str, Any] = {
        "kind": "localagent_m630_workshop_gate_current_m626",
        "schema_version": 1,
        "current_checkpoint": gate["current_checkpoint"],
        "gate": gate,
        "gate_input": _identity(gate_path),
        "webgpu_adoption_receipt": _identity(adoption_path),
        "mobilegym_receipt": _identity(mobilegym_path),
        "transfer_receipt": _identity(transfer_path),
        "decision": {
            "ready": False,
            "passed_checks": sorted(
                requirement for requirement, check in checks.items() if check.get("status") == "pass"
            ),
            "blocking_requirements": gate["blocking_requirements"],
            "native_mobile_evidence_current": checks.get("native:mobilegym", {}).get("status") == "pass",
            "webgpu_evidence_current": checks.get("webgpu:native_capability_and_latency", {}).get("status") == "pass",
        },
        "claim_boundary": (
            "Current m626 gate with a complete native MobileGym test-split receipt. MobileGym, WebGPU, "
            "and catalog checks are checkpoint-bound, but the remaining native suites, fresh weight "
            "ablation, RL preflight, and public artifact manifest still prevent workshop approval."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--mobilegym", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    receipt = assemble(args.gate, args.adoption, args.mobilegym, args.transfer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
