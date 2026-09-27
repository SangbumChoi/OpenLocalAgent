#!/usr/bin/env python3
"""Seal a public-train AppWorld API-head routing diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assemble(
    *,
    head_report: Path,
    native_report: Path,
    checkpoint: Path,
    train_manifest: Path,
    eval_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    head = _load(head_report)
    native = _load(native_report)
    if head.get("kind") != "localagent_appworld_api_head_training_report":
        raise ValueError("API-head report kind mismatch")
    if native.get("kind") != "localagent_appworld_checkpoint_native_probe":
        raise ValueError("native report kind mismatch")
    checkpoint_identity = _identity(checkpoint)
    if native.get("checkpoint", {}).get("sha256") != checkpoint_identity["sha256"]:
        raise ValueError("native report is not bound to current checkpoint")
    summary = native.get("summary")
    metrics = head.get("metrics")
    if not isinstance(summary, dict) or not isinstance(metrics, dict):
        raise ValueError("head/native metrics missing")
    if summary.get("tasks") != 6:
        raise ValueError("expected six public dev tasks")
    payload: dict[str, Any] = {
        "kind": "localagent_m669_appworld_api_head_diagnostic_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "data_version": "0.2.0",
            "train_rows": 90,
            "eval_rows": 6,
            "protected_test_used": False,
        },
        "checkpoint": checkpoint_identity,
        "api_head": {
            "training_report": _identity(head_report),
            "sidecar": head["child"],
            "classes": head["classes"],
            "train": metrics["train"],
            "eval": metrics["eval"],
        },
        "native_replay": {
            "report": _identity(native_report),
            "configuration": native["configuration"],
            "summary": summary,
            "tasks": native["tasks"],
        },
        "dataset_inputs": {
            "train": _identity(train_manifest),
            "eval": _identity(eval_manifest),
        },
        "decision": {
            "retain_frozen_body": True,
            "promote_api_head": False,
            "promote_to_webgpu": False,
            "reason": (
                "The API head memorizes the 90 public train labels (100% train exact) but reaches "
                "only 2/6 disjoint dev labels. With selector-first schema replay, three actions are "
                "replayed but native task success remains 0/6; the learned routing head therefore "
                "does not replace the executor-side planner."
            ),
        },
        "claim_boundary": (
            "Frozen-backbone AppWorld app.api head trained on public train rows and evaluated on six "
            "disjoint public dev tasks. Native replay uses resettable local AppWorld databases only; "
            "this is not an official leaderboard score, external-account result, email/Notion result, "
            "or WebGPU deployment claim."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        head_report=args.head_report,
        native_report=args.native_report,
        checkpoint=args.checkpoint,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        output=args.out,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
