#!/usr/bin/env python3
"""Refresh the m679 gate with browser-specific SFT/native transfer evidence."""

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


def assemble(*, prior: Path, checkpoint: Path, transfer: Path, output: Path) -> dict[str, Any]:
    gate = json.loads(prior.read_text())
    receipt = json.loads(transfer.read_text())
    current = _identity(checkpoint)
    if gate.get("current_checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("prior gate checkpoint mismatch")
    if receipt.get("parent", {}).get("sha256") != current["sha256"]:
        raise ValueError("transfer parent checkpoint mismatch")
    blocked = dict(gate.get("blocked_requirements", {}))
    blocked["native:mcpmark_playwright"] = "browser_sft_native_verifier_zero"
    payload: dict[str, Any] = {
        "kind": "localagent_m706_workshop_gate_current_m679",
        "schema_version": 1,
        "ready": False,
        "current_checkpoint": current,
        "passed_requirements": sorted(set(gate.get("passed_requirements", []))),
        "blocked_requirements": dict(sorted(blocked.items())),
        "evidence": {**gate.get("evidence", {}), "mcpmark_browser_transfer_native": _identity(transfer)},
        "claim_boundary": (
            "Fail-closed gate after browser-specific public SFT and native Playwright replay. "
            "Held-out text accuracy improves, but native verifier success remains zero; no visual, "
            "official-split, external-account, or checkpoint-bound public artifact claim is made."
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
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(prior=args.prior, checkpoint=args.checkpoint, transfer=args.transfer, output=args.output)
    print(json.dumps({"output": str(args.output), "ready": payload["ready"], "blockers": len(payload["blocked_requirements"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
