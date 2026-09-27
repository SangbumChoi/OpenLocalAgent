#!/usr/bin/env python3
"""Bind a workshop-gate refresh to the current m607 transfer receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def assemble(gate_path: Path, transfer_path: Path) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    if gate.get("kind") != "localagent_workshop_publication_gate":
        raise ValueError("unexpected gate kind")
    if transfer.get("kind") != "localagent_m607_current_policy_transfer":
        raise ValueError("unexpected m607 transfer kind")
    checks = {row["requirement"]: row for row in gate.get("checks", [])}
    if checks.get("weights:transfer_and_no_transfer_ablation", {}).get("status") != "pass":
        raise ValueError("m607 transfer did not satisfy the weight gate")
    body: dict[str, Any] = {
        "kind": "localagent_m608_workshop_gate_current_m607",
        "schema_version": 1,
        "gate": gate,
        "gate_input": _identity(gate_path),
        "transfer_receipt": {
            "path": str(transfer_path),
            "sha256": transfer["receipt_self_sha256"],
            "parent_checkpoint": transfer["parent_checkpoint"],
            "warm_minus_random_after_pp": transfer["decision"]["warm_minus_random_after_pp"],
        },
        "claim_boundary": (
            "This refresh changes only the checkpoint-bound transfer evidence in the fail-closed "
            "gate. It does not synthesize native benchmark receipts, public Hub publication, or "
            "real email/Notion/browser side effects."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--transfer-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    receipt = assemble(args.gate_report, args.transfer_receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
