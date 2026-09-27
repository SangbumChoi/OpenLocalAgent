#!/usr/bin/env python3
"""Assemble the bounded public xLAM-derived warm/random continuation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET = "product-science/xlam-function-calling-60k-raw"
DATASET_URL = "https://huggingface.co/datasets/product-science/xlam-function-calling-60k-raw"
DATASET_REVISION = "dfbd3c669354c27f2727870d39a4d86c32381448"
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _source(report: dict, label: str) -> dict:
    for source in report["train_sources"] + report["eval_sources"]:
        if source["label"] == label:
            return source
    raise ValueError(f"source label not present: {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    warm = json.loads(args.warm_report.read_text(encoding="utf-8"))
    random = json.loads(args.random_report.read_text(encoding="utf-8"))
    train_manifest = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    eval_manifest = json.loads(args.eval_manifest.read_text(encoding="utf-8"))
    for label, report in (("warm", warm), ("random", random)):
        if report["kind"] != "localagent_cross_surface_public_continuation_report":
            raise ValueError(f"unexpected {label} report kind")
        if report["parent"]["sha256"] != PARENT_SHA256:
            raise ValueError(f"{label} parent is not the current checkpoint")
        if report["hyperparameters"]["steps"] != 24:
            raise ValueError(f"{label} step count mismatch")
        if report["hyperparameters"]["learning_rate"] != 1e-5:
            raise ValueError(f"{label} learning rate mismatch")
    if random["hyperparameters"]["backbone_init"] != "random":
        raise ValueError("random report is not the random-backbone arm")
    if warm["hyperparameters"]["backbone_init"] != "parent":
        raise ValueError("warm report is not the parent-backbone arm")
    for report in (warm, random):
        for source in report["train_sources"] + report["eval_sources"]:
            if source["datasets"] != [DATASET] or source["revisions"] != [DATASET_REVISION]:
                raise ValueError("xLAM source identity mismatch")
    if train_manifest["source"]["revision"] != DATASET_REVISION:
        raise ValueError("train manifest revision mismatch")
    if eval_manifest["source"]["revision"] != DATASET_REVISION:
        raise ValueError("eval manifest revision mismatch")
    if train_manifest["source"]["official_salesforce_split"]:
        raise ValueError("derived train source must not be labeled as official Salesforce")
    if eval_manifest["source"]["official_salesforce_split"]:
        raise ValueError("derived eval source must not be labeled as official Salesforce")

    warm_after = warm["after"]["eval"]
    warm_before = warm["before"]["eval"]
    random_after = random["after"]["eval"]
    random_before = random["before"]["eval"]
    warm_gain = warm_after["assistant_token_accuracy"] - warm_before["assistant_token_accuracy"]
    random_gain = random_after["assistant_token_accuracy"] - random_before["assistant_token_accuracy"]
    advantage = warm_after["assistant_token_accuracy"] - random_after["assistant_token_accuracy"]
    warm_wins = advantage > 0
    payload = {
        "kind": "localagent_xlam_derived_warm_random_continuation_receipt",
        "schema_version": 1,
        "dataset": {
            "name": DATASET,
            "url": DATASET_URL,
            "revision": DATASET_REVISION,
            "license": "apache-2.0",
            "official_salesforce_split_verified": False,
            "train": {
                "source_label": "xlam_train",
                "rows": warm["rows"]["train"],
                "source": _source(warm, "xlam_train"),
                "manifest": train_manifest,
            },
            "eval": {
                "source_label": "xlam_eval",
                "rows": warm["rows"]["eval"],
                "source": _source(warm, "xlam_eval"),
                "manifest": eval_manifest,
            },
            "split_boundary": (
                "The derivative's explicit train/test directories are retained as separate source "
                "labels. Generic slot-value overlap exists across the derivative shards, so the "
                "generic slot-disjoint guard is not used as evidence of independence."
            ),
        },
        "parent_checkpoint": warm["parent"],
        "protocol": {
            "steps": warm["hyperparameters"]["steps"],
            "batch_size": warm["hyperparameters"]["batch_size"],
            "learning_rate": warm["hyperparameters"]["learning_rate"],
            "max_seq_len": warm["hyperparameters"]["max_seq_len"],
            "seed": warm["hyperparameters"]["seed"],
            "device": warm["hyperparameters"]["device"],
            "warm_backbone_init": "parent",
            "random_backbone_init": "random",
            "random_backbone_seed": random["hyperparameters"]["random_backbone_seed"],
        },
        "warm": {
            "child_checkpoint": warm["child"],
            "before": warm["before"],
            "after": warm["after"],
            "weight_transfer": warm["weight_transfer"],
        },
        "random": {
            "child_checkpoint": random["child"],
            "before": random["before"],
            "after": random["after"],
            "weight_transfer": random["weight_transfer"],
        },
        "comparison": {
            "warm_eval_token_accuracy_before": warm_before["assistant_token_accuracy"],
            "warm_eval_token_accuracy_after": warm_after["assistant_token_accuracy"],
            "warm_eval_token_accuracy_gain": warm_gain,
            "random_eval_token_accuracy_before": random_before["assistant_token_accuracy"],
            "random_eval_token_accuracy_after": random_after["assistant_token_accuracy"],
            "random_eval_token_accuracy_gain": random_gain,
            "warm_minus_random_eval_token_accuracy_after": advantage,
            "warm_eval_sequence_accuracy_after": warm_after["assistant_sequence_accuracy"],
            "random_eval_sequence_accuracy_after": random_after["assistant_sequence_accuracy"],
            "warm_wins_teacher_forced_tokens": warm_wins,
        },
        "decision": {
            "adoption": (
                "retain_as_low_rate_initialization_candidate"
                if warm_wins
                else "reject_warm_initialization_candidate"
            ),
            "native_replay_required": True,
            "webgpu_export_allowed": False,
            "reason": (
                "This is a bounded derivative-source continuation, not an official Salesforce "
                "xLAM split. Warm held-out teacher-forced token accuracy is "
                f"{warm_after['assistant_token_accuracy']:.6f} versus "
                f"{random_after['assistant_token_accuracy']:.6f} random "
                f"({advantage * 100:.2f} percentage points); exact sequence accuracy is 0% "
                "for both. The source may inform low-rate initialization only after the public "
                "derivative terms, slot-overlap limitation, native replay, and a full tool-use "
                "evaluation are addressed."
            ),
        },
        "source_artifacts": {
            "warm_report": identity(args.warm_report),
            "random_report": identity(args.random_report),
            "train_manifest": identity(args.train_manifest),
            "eval_manifest": identity(args.eval_manifest),
        },
        "claim_boundary": (
            "This is a source-bound teacher-forced continuation on a public Apache-2.0 derivative "
            "of xLAM function-calling data. It is not an official Salesforce xLAM score, BFCL "
            "score, multi-call tool-use score, native MCP result, live API execution, or proof "
            "of browser/email/Notion side-effect competence."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
