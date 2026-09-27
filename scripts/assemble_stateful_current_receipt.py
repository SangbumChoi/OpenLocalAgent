#!/usr/bin/env python3
"""Bind a compact, fail-closed receipt for a current stateful transfer probe.

The trainer writes a verbose local report.  This assembler keeps only aggregate metrics, source
inventory, checkpoint identities, and weight movement in the tracked result; prompts and model
outputs remain outside the repository.  The probe is a local synthetic diagnostic, never an
official mobile/browser/MCP score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(report: dict[str, Any], label: str) -> dict[str, Any]:
    arm = report["arms"][label]
    return {
        "label": arm.get("label"),
        "random_backbone": arm.get("random_backbone"),
        "warm_start_heads": arm.get("warm_start_heads"),
        "closed_loop": arm.get("closed_loop"),
        "selector": arm.get("selector"),
        "training": arm.get("training"),
        "weight_movement": arm.get("weight_movement"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")

    report = _load(args.report)
    source = report.get("source", {})
    if source.get("kind") != "local_synthetic_state_machine":
        raise ValueError("stateful receipt requires the canonical local synthetic state machine")
    if source.get("native_runtime_executed") or source.get("tools_executed"):
        raise ValueError("stateful receipt refuses native/tool side effects")
    receipt: dict[str, Any] = {
        "kind": "localagent_stateful_productivity_current_transfer_receipt",
        "schema_version": 1,
        "suite": report.get("suite"),
        "parent": report.get("parent"),
        "output": report.get("output"),
        "report": _identity(args.report),
        "source": {
            "kind": source.get("kind"),
            "suite": report.get("suite"),
            "train_inventory": source.get("train_inventory"),
            "eval_inventory": source.get("eval_inventory"),
            "train_task_hash": source.get("train_task_hash"),
            "eval_task_hash": source.get("eval_task_hash"),
            "public_benchmark_text_used": source.get("public_benchmark_text_used"),
            "native_runtime_executed": source.get("native_runtime_executed"),
            "tools_executed": source.get("tools_executed"),
            "external_accounts_used": source.get("external_accounts_used"),
        },
        "configuration": report.get("configuration"),
        "comparison": report.get("comparison"),
        "arms": {
            "pretrained_frozen_backbone": _arm(report, "pretrained_frozen_backbone"),
            "pretrained_lowrate_unfrozen_backbone": _arm(
                report, "pretrained_lowrate_unfrozen_backbone"
            ),
            "matched_random_backbone": _arm(report, "matched_random_backbone"),
        },
        "decision": "diagnostic_only",
        "claim_boundary": (
            "Matched local synthetic state-machine transfer diagnostic on the current 10.52M "
            "checkpoint. It is not AndroidWorld, BrowserGym, OSWorld, MCPMark, EnterpriseOps-Gym, "
            "real email or Notion access, screenshot grounding, trusted computer control, or a "
            "native WebGPU capability result."
        ),
    }
    receipt["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
