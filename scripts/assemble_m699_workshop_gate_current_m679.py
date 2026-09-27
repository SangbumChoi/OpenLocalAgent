#!/usr/bin/env python3
"""Refresh the fail-closed workshop gate with m698 native transfer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assemble(*, prior: Path, checkpoint: Path, native: Path, continuation: Path, output: Path) -> dict[str, Any]:
    gate = json.loads(prior.read_text())
    native_receipt = json.loads(native.read_text())
    continuation_receipt = json.loads(continuation.read_text())
    current = _identity(checkpoint)
    if gate.get("current_checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("prior gate is not bound to supplied checkpoint")
    if native_receipt.get("checkpoint_sha256") != current["sha256"]:
        raise ValueError("m697 native receipt checkpoint mismatch")
    if continuation_receipt.get("parent", {}).get("sha256") != current["sha256"]:
        raise ValueError("m698 continuation parent mismatch")
    blocked = dict(gate.get("blocked_requirements", {}))
    blocked.update(
        {
            "native:mcpmark": "native_verifier_zero",
            "native:appworld": "native_completion_zero",
            "native:enterpriseopsgym": "no_server_or_verifier_execution",
        }
    )
    passed = [item for item in gate.get("passed_requirements", []) if item != "native:mcpmark"]
    payload: dict[str, Any] = {
        "kind": "localagent_m699_workshop_gate_current_m679",
        "schema_version": 1,
        "ready": False,
        "current_checkpoint": current,
        "passed_requirements": sorted(set(passed)),
        "blocked_requirements": dict(sorted(blocked.items())),
        "evidence": {
            **gate.get("evidence", {}),
            "mcpmark_native_filesystem_standard": _identity(native),
            "mcpmark_trajectory_continuation_native": _identity(continuation),
            "appworld_native": "docs/paper/results/raw/m693-m679-appworld-native-v1.json",
            "appworld_free": "docs/paper/results/raw/m694-m679-appworld-free-v1.json",
            "appworld_sft_native": "docs/paper/results/raw/m696-m679-appworld-sft-native-v1.json",
        },
        "claim_boundary": (
            "Fail-closed workshop/publication gate after current native MCPMark and AppWorld "
            "execution. WebGPU capability, MobileGym/BrowserGym diagnostics, RL preflight, and "
            "weight lineage are evidenced, but native filesystem/AppWorld completion is zero, "
            "visual inputs and real accounts are absent, and the public HF/Space artifacts are "
            "not bound to this checkpoint."
        ),
        "source_gate": _identity(prior),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        prior=args.prior,
        checkpoint=args.checkpoint,
        native=args.native,
        continuation=args.continuation,
        output=args.output,
    )
    print(json.dumps({"output": str(args.output), "ready": payload["ready"], "blockers": len(payload["blocked_requirements"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
