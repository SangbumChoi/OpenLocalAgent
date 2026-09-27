#!/usr/bin/env python3
"""Refresh the fail-closed m679 workshop gate with current AgentNet evidence."""

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


def assemble(*, prior: Path, checkpoint: Path, visual: Path, transfer: Path, output: Path) -> dict[str, Any]:
    gate = json.loads(prior.read_text(encoding="utf-8"))
    current = identity(checkpoint)
    if gate.get("current_checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("prior gate checkpoint mismatch")
    visual_payload = json.loads(visual.read_text(encoding="utf-8"))
    if visual_payload.get("kind") != "localagent_agentnet_visual_source_audit":
        raise ValueError("unexpected AgentNet visual receipt kind")
    transfer_payload = json.loads(transfer.read_text(encoding="utf-8"))
    if transfer_payload.get("parent", {}).get("sha256") != current["sha256"]:
        raise ValueError("AgentNet transfer parent mismatch")
    blocked = dict(gate.get("blocked_requirements", {}))
    blocked["native:agentnet"] = "visual_runtime_and_native_verifier_missing; text_projection_success_zero"
    evidence = dict(gate.get("evidence", {}))
    evidence.update(
        {
            "agentnet_visual_source": identity(visual),
            "agentnet_selector_transfer": identity(transfer),
        }
    )
    payload: dict[str, Any] = {
        "kind": "localagent_m710_workshop_gate_current_m679",
        "schema_version": 1,
        "ready": False,
        "current_checkpoint": current,
        "passed_requirements": sorted(set(gate.get("passed_requirements", []))),
        "blocked_requirements": dict(sorted(blocked.items())),
        "evidence": evidence,
        "claim_boundary": (
            "Fail-closed gate after pinned AgentNet visual provenance and matched selector transfer. "
            "AgentNet screenshots remain unconsumed, native desktop verification is absent, and the "
            "text projection has zero task success; no visual, native, or public-artifact promotion "
            "is authorized."
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
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        prior=args.prior,
        checkpoint=args.checkpoint,
        visual=args.visual,
        transfer=args.transfer,
        output=args.output,
    )
    print(json.dumps({"ready": payload["ready"], "blockers": len(payload["blocked_requirements"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
