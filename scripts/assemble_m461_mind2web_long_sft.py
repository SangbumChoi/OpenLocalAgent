#!/usr/bin/env python3
"""Assemble the longer current-parent Mind2Web SFT and weight audit receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_URL = "https://huggingface.co/datasets/osunlp/Mind2Web"
DATASET_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--weight-report", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    weight = json.loads(args.weight_report.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if training["kind"] != "localagent_public_agent_continuation_report":
        raise ValueError("unexpected public-agent training report kind")
    if weight["kind"] != "localagent_weight_transfer_analysis":
        raise ValueError("unexpected weight report kind")
    if training["parent"]["sha256"] != PARENT_SHA256:
        raise ValueError("training parent is not the current checkpoint")
    if training["source"]["revision"] != DATASET_REVISION:
        raise ValueError("Mind2Web source revision mismatch")
    groups = weight["groups"]
    payload = {
        "kind": "localagent_mind2web_long_sft_and_weight_receipt",
        "schema_version": 1,
        "dataset": {
            "name": "osunlp/Mind2Web",
            "url": DATASET_URL,
            "revision": DATASET_REVISION,
            "train_rows": training["rows"]["train"],
            "eval_rows": training["rows"]["eval"],
            "train_identity": training["train_inputs"],
            "eval_identity": training["eval_inputs"],
            "source_manifest": identity(args.source_manifest),
            "manifest_self_sha256": source_manifest.get("manifest_self_sha256"),
        },
        "parent_checkpoint": training["parent"],
        "child_checkpoint": training["child"],
        "protocol": training["hyperparameters"],
        "teacher_forced_before": training["before"],
        "teacher_forced": training["after"],
        "heads": training["heads"],
        "weight_transfer": {
            "compatibility": weight["compatibility"],
            "groups": groups,
            "recommendation": weight["recommendation"],
        },
        "decision": {
            "adoption": "retain_as_low_rate_initialization_candidate",
            "reason": (
                "Held-out token accuracy improves on a source-disjoint public Mind2Web split, "
                "but the 24-step continuation moves the embedding/attention/FFN groups by "
                "0.328%/0.173%/0.206% and exact sequence accuracy remains 0%. Action heads "
                "are frozen, so native replay and a matched no-transfer control are required "
                "before any WebGPU export or promotion."
            ),
            "native_replay_required": True,
            "webgpu_export_allowed": False,
        },
        "loss_history": training["loss_history"],
        "token_accounting": training["token_accounting"],
        "source_artifacts": {
            "training_report": identity(args.training_report),
            "weight_report": identity(args.weight_report),
        },
        "claim_boundary": (
            "This is a longer public train-only Mind2Web DOM/action continuation with a "
            "source-disjoint teacher-forced evaluation and a matched checkpoint weight audit. "
            "It is not an official Mind2Web test score, BrowserGym score, visual grounding "
            "result, live browser result, MCP result, or real-account side-effect claim."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
