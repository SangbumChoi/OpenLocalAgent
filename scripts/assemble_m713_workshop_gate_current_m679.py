#!/usr/bin/env python3
"""Refresh the fail-closed m679 gate with visual action-transfer evidence."""

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


def assemble(*, prior: Path, checkpoint: Path, smoke: Path, pilot: Path, output: Path) -> dict[str, Any]:
    gate = json.loads(prior.read_text(encoding="utf-8"))
    current = identity(checkpoint)
    if gate.get("current_checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("prior gate checkpoint mismatch")
    smoke_payload = json.loads(smoke.read_text(encoding="utf-8"))
    pilot_payload = json.loads(pilot.read_text(encoding="utf-8"))
    if smoke_payload.get("kind") != "localagent_m711_androidcontrol_visual_bridge_smoke":
        raise ValueError("unexpected visual bridge receipt kind")
    if pilot_payload.get("kind") != "localagent_m712_androidcontrol_visual_action_pilot":
        raise ValueError("unexpected visual action pilot receipt kind")
    blocked = dict(gate.get("blocked_requirements", {}))
    blocked["native:androidcontrol_visual"] = "visual_pilot_sequence_exact_zero_and_no_emulator_verifier"
    evidence = dict(gate.get("evidence", {}))
    evidence.update({"androidcontrol_visual_bridge": identity(smoke), "androidcontrol_visual_action_pilot": identity(pilot)})
    payload: dict[str, Any] = {
        "kind": "localagent_m713_workshop_gate_current_m679",
        "schema_version": 1,
        "ready": False,
        "current_checkpoint": current,
        "passed_requirements": sorted(set(gate.get("passed_requirements", []))),
        "blocked_requirements": dict(sorted(blocked.items())),
        "evidence": evidence,
        "claim_boundary": (
            "Fail-closed gate after the first bounded AndroidControl screenshot-to-action pilot. "
            "Visual bridge wiring and teacher-forced action transfer are evidenced, but sequence "
            "exactness is zero and no emulator/native/WebGPU visual evaluator ran; no publication or "
            "checkpoint promotion is authorized."
        ),
        "source_gate": identity(prior),
    }
    payload["receipt_self_sha256"] = self_hash(payload)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        prior=args.prior,
        checkpoint=args.checkpoint,
        smoke=args.smoke,
        pilot=args.pilot,
        output=args.output,
    )
    print(json.dumps({"ready": payload["ready"], "blockers": len(payload["blocked_requirements"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
