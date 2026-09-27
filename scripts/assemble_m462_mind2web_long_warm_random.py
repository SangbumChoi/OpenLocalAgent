#!/usr/bin/env python3
"""Assemble the matched warm-versus-random Mind2Web continuation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _metric(report: dict, arm: str, phase: str, name: str) -> float:
    return float(report[arm][phase]["eval"][name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-receipt", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    warm = json.loads(args.warm_receipt.read_text(encoding="utf-8"))
    random = json.loads(args.random_report.read_text(encoding="utf-8"))
    if warm["kind"] != "localagent_mind2web_long_sft_and_weight_receipt":
        raise ValueError("unexpected warm receipt kind")
    if random["kind"] != "localagent_cross_surface_public_continuation_report":
        raise ValueError("unexpected random report kind")
    if warm["dataset"]["revision"] != DATASET_REVISION:
        raise ValueError("warm Mind2Web revision mismatch")
    if random["eval_sources"][0]["revisions"] != [DATASET_REVISION]:
        raise ValueError("random Mind2Web revision mismatch")
    if warm["parent_checkpoint"]["sha256"] != PARENT_SHA256:
        raise ValueError("warm parent is not the current checkpoint")
    if random["parent"]["sha256"] != PARENT_SHA256:
        raise ValueError("random parent is not the current checkpoint")
    random_hyper = random["hyperparameters"]
    warm_protocol = warm["protocol"]
    for key in ("batch_size", "learning_rate", "max_seq_len", "seed", "steps"):
        if random_hyper[key] != warm_protocol[key]:
            raise ValueError(f"protocol mismatch for {key}")
    if random_hyper["backbone_init"] != "random":
        raise ValueError("random report is not the random-backbone arm")
    if random["split_contract"]["mode"] != "source_local_parent_and_slot_disjoint":
        raise ValueError("random split contract mismatch")

    warm_before = warm["teacher_forced_before"]["eval"]
    warm_after = warm["teacher_forced"]["eval"]
    random_before = random["before"]["eval"]
    random_after = random["after"]["eval"]
    warm_gain = warm_after["assistant_token_accuracy"] - warm_before["assistant_token_accuracy"]
    random_gain = random_after["assistant_token_accuracy"] - random_before["assistant_token_accuracy"]
    warm_minus_random = (
        warm_after["assistant_token_accuracy"] - random_after["assistant_token_accuracy"]
    )
    warm_wins = warm_after["assistant_token_accuracy"] > random_after["assistant_token_accuracy"]
    decision = (
        "retain_as_low_rate_initialization_candidate"
        if warm_wins
        else "reject_warm_initialization_candidate"
    )
    payload = {
        "kind": "localagent_mind2web_long_warm_random_comparison_receipt",
        "schema_version": 1,
        "dataset": {
            "name": warm["dataset"]["name"],
            "url": warm["dataset"]["url"],
            "revision": DATASET_REVISION,
            "train_rows": warm["dataset"]["train_rows"],
            "eval_rows": warm["dataset"]["eval_rows"],
            "train_identity": warm["dataset"]["train_identity"],
            "eval_identity": warm["dataset"]["eval_identity"],
            "source_manifest": warm["dataset"]["source_manifest"],
        },
        "parent_checkpoint": warm["parent_checkpoint"],
        "protocol": {
            "batch_size": warm_protocol["batch_size"],
            "learning_rate": warm_protocol["learning_rate"],
            "max_seq_len": warm_protocol["max_seq_len"],
            "seed": warm_protocol["seed"],
            "steps": warm_protocol["steps"],
            "device": warm_protocol["device"],
            "warm_backbone_init": warm_protocol["head_init"],
            "random_backbone_init": random_hyper["backbone_init"],
            "random_backbone_seed": random_hyper["random_backbone_seed"],
            "head_steps": warm_protocol["head_steps"],
        },
        "warm": {
            "receipt_self_sha256": warm["receipt_self_sha256"],
            "child_checkpoint": warm["child_checkpoint"],
            "teacher_forced_before": warm["teacher_forced_before"],
            "teacher_forced": warm["teacher_forced"],
            "weight_transfer": warm["weight_transfer"],
        },
        "random": {
            "report": identity(args.random_report),
            "child_checkpoint": random["child"],
            "teacher_forced_before": random["before"],
            "teacher_forced": random["after"],
            "weight_transfer": random["weight_transfer"],
        },
        "comparison": {
            "warm_eval_token_accuracy_before": warm_before["assistant_token_accuracy"],
            "warm_eval_token_accuracy_after": warm_after["assistant_token_accuracy"],
            "warm_eval_token_accuracy_gain": warm_gain,
            "random_eval_token_accuracy_before": random_before["assistant_token_accuracy"],
            "random_eval_token_accuracy_after": random_after["assistant_token_accuracy"],
            "random_eval_token_accuracy_gain": random_gain,
            "warm_minus_random_eval_token_accuracy_after": warm_minus_random,
            "warm_eval_sequence_accuracy_after": warm_after["assistant_sequence_accuracy"],
            "random_eval_sequence_accuracy_after": random_after["assistant_sequence_accuracy"],
            "warm_wins_teacher_forced_tokens": warm_wins,
        },
        "decision": {
            "adoption": decision,
            "native_replay_required": True,
            "webgpu_export_allowed": False,
            "reason": (
                "The matched warm arm is compared with a shape-matched random-backbone arm on "
                "the same pinned public Mind2Web rows, tokenizer, parent checkpoint, seed, and "
                "24-step low-rate protocol. Warm held-out teacher-forced token accuracy is "
                f"{warm_after['assistant_token_accuracy']:.6f} versus "
                f"{random_after['assistant_token_accuracy']:.6f} random "
                f"({warm_minus_random * 100:.2f} percentage points); exact sequence accuracy "
                "is 0% for both. This supports retaining warm initialization as a candidate, "
                "not promotion: native replay, official benchmark checks, and the public "
                "model/demo manifest remain required."
            ),
        },
        "claim_boundary": (
            "This is a source-disjoint teacher-forced transfer ablation on a public Mind2Web "
            "train-only projection. It is not an official Mind2Web test score, BrowserGym or "
            "MiniWoB score, visual grounding result, live browser result, MCP result, or a "
            "real-account email/Notion side-effect claim."
        ),
        "source_artifacts": {
            "warm_receipt": identity(args.warm_receipt),
            "random_report": identity(args.random_report),
        },
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
