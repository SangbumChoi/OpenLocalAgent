#!/usr/bin/env python3
"""Refresh the fail-closed m679 gate with the browser visual sidecar probe."""

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
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate = json.loads(args.prior.read_text(encoding="utf-8"))
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    current = identity(args.checkpoint)
    if gate.get("current_checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("prior gate checkpoint mismatch")
    if probe.get("kind") != "localagent_m718_visual_webgpu_probe":
        raise ValueError("unexpected visual probe receipt kind")
    if probe.get("runtime", {}).get("webgpu", {}).get("session_ready") is not True:
        raise ValueError("visual probe did not create a WebGPU session")

    blocked = dict(gate.get("blocked_requirements", {}))
    blocked["artifacts:visual_webgpu_export"] = "browser_visual_probe_bound_but_native_mobile_verifier_missing"
    evidence = dict(gate.get("evidence", {}))
    evidence["visual_webgpu_probe"] = identity(args.probe)
    payload: dict[str, Any] = {
        "kind": "localagent_m719_workshop_gate_current_m679",
        "schema_version": 1,
        "ready": False,
        "current_checkpoint": current,
        "passed_requirements": sorted(set(gate.get("passed_requirements", []))),
        "blocked_requirements": dict(sorted(blocked.items())),
        "evidence": evidence,
        "claim_boundary": (
            "Fail-closed gate after browser execution of the pinned visual sidecar. The graph now "
            "has an explicit onnxruntime-web WebGPU probe, but native Android emulator validation, "
            "official AndroidControl scoring, and current-checkpoint release binding remain absent."
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
