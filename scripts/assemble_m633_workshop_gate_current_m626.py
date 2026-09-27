#!/usr/bin/env python3
"""Bind the current m626 gate after the RL and BrowserGym refreshes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CURRENT_CHECKPOINT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def assemble(
    gate_path: Path,
    adoption_path: Path,
    mobilegym_path: Path,
    browsergym_path: Path,
    transfer_path: Path,
    rl_path: Path,
) -> dict[str, Any]:
    gate = _load(gate_path)
    adoption = _load(adoption_path)
    mobilegym = _load(mobilegym_path)
    browsergym = _load(browsergym_path)
    transfer = _load(transfer_path)
    rl = _load(rl_path)
    if gate.get("kind") != "localagent_workshop_publication_gate" or gate.get("ready") is not False:
        raise ValueError("gate must remain fail-closed")
    if gate.get("current_checkpoint", {}).get("sha256") != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("gate is not bound to m626")
    if adoption.get("checkpoint", {}).get("sha256") != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("WebGPU adoption checkpoint mismatch")
    for payload, benchmark_id in ((mobilegym, "mobilegym"), (browsergym, "browsergym_miniwob")):
        if payload.get("benchmark_id") != benchmark_id:
            raise ValueError(f"unexpected benchmark id: {benchmark_id}")
        if payload.get("checkpoint", {}).get("sha256") != CURRENT_CHECKPOINT_SHA256:
            raise ValueError(f"{benchmark_id} checkpoint mismatch")
    if transfer.get("parent_checkpoint", {}).get("sha256") != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("transfer parent checkpoint mismatch")
    if rl.get("status") != "passed" or rl.get("lineage", {}).get("parent_checkpoint_sha256") != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("RL preflight is not a passed current-checkpoint receipt")
    checks = {item["requirement"]: item for item in gate.get("checks", [])}
    payload: dict[str, Any] = {
        "kind": "localagent_m633_workshop_gate_current_m626",
        "schema_version": 1,
        "current_checkpoint": gate["current_checkpoint"],
        "gate": gate,
        "gate_input": _identity(gate_path),
        "webgpu_adoption_receipt": _identity(adoption_path),
        "mobilegym_receipt": _identity(mobilegym_path),
        "browsergym_receipt": _identity(browsergym_path),
        "transfer_receipt": _identity(transfer_path),
        "rl_preflight_receipt": _identity(rl_path),
        "decision": {
            "ready": False,
            "passed_checks": sorted(
                requirement for requirement, check in checks.items() if check.get("status") == "pass"
            ),
            "blocking_requirements": gate["blocking_requirements"],
            "native_mobile_evidence_current": checks.get("native:mobilegym", {}).get("status") == "pass",
            "native_browser_evidence_current": checks.get("native:browsergym_miniwob", {}).get("status") == "pass",
            "rl_preflight_current": checks.get("training:rl_preflight", {}).get("status") == "pass",
        },
        "claim_boundary": (
            "Current m626 gate with checkpoint-bound native MobileGym/BrowserGym receipts and a "
            "strict isolated RL preflight. Readiness remains fail-closed until every required "
            "native suite and authenticated public model/demo manifest are supplied."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gate", "adoption", "mobilegym", "browsergym", "transfer", "rl"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    payload = assemble(args.gate, args.adoption, args.mobilegym, args.browsergym, args.transfer, args.rl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
