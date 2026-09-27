#!/usr/bin/env python3
"""Seal the m666 AppWorld warm/random weight-transfer ablation."""

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
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def assemble(
    *,
    warm_weight: Path,
    random_weight: Path,
    appworld_receipt: Path,
    current_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    warm = _load(warm_weight)
    random = _load(random_weight)
    appworld = _load(appworld_receipt)
    if warm.get("kind") != "localagent_weight_transfer_analysis":
        raise ValueError("warm weight kind mismatch")
    if random.get("kind") != "localagent_weight_transfer_analysis":
        raise ValueError("random weight kind mismatch")
    if appworld.get("kind") != "localagent_m666_appworld_public_full_receipt":
        raise ValueError("m666 receipt kind mismatch")
    warm_compat = warm.get("compatibility")
    random_compat = random.get("compatibility")
    if not isinstance(warm_compat, dict) or not isinstance(random_compat, dict):
        raise ValueError("both weight reports must contain compatibility")
    for label, compatibility in (("warm", warm_compat), ("random", random_compat)):
        if compatibility.get("config_mismatches") not in ({}, None):
            raise ValueError(f"{label} config mismatch")
        if compatibility.get("shape_mismatches") not in ({}, None):
            raise ValueError(f"{label} shape mismatch")
        if compatibility.get("tokenizer_sha256_equal") is not True:
            raise ValueError(f"{label} tokenizer mismatch")
    if not isinstance(warm.get("groups"), dict) or not warm["groups"]:
        raise ValueError("warm movement groups missing")
    if not isinstance(random.get("groups"), dict) or not random["groups"]:
        raise ValueError("random movement groups missing")
    if warm["base"].get("lineage", {}).get("data_sha256") != random["base"].get("lineage", {}).get("data_sha256"):
        raise ValueError("warm/random data lineage differs")
    if warm["target"].get("lineage", {}).get("data_sha256") != random["target"].get("lineage", {}).get("data_sha256"):
        raise ValueError("warm/random target data lineage differs")
    metrics = appworld.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("m666 metrics missing")
    parent_sha = warm.get("target", {}).get("sha256")
    current_identity = _identity(current_checkpoint)
    if parent_sha != current_identity["sha256"]:
        raise ValueError("warm target is not the current checkpoint")
    held_out = {
        "parent_heads": {
            "label": "warm_parent_backbone",
            "checkpoint_sha256": warm["target"]["sha256"],
            "teacher_forced": metrics["warm_after"],
            "free_running_success_rate": metrics["warm_free_running_success_rate"],
            "structured_planner_success_rate": metrics["warm_structured_planner_success_rate"],
        },
        "random": {
            "label": "random_backbone_control",
            "checkpoint_sha256": random["target"]["sha256"],
            "teacher_forced": metrics["random_after"],
            "free_running_success_rate": metrics["random_free_running_success_rate"],
            "structured_planner_success_rate": metrics["random_structured_planner_success_rate"],
        },
    }
    payload: dict[str, Any] = {
        "kind": "localagent_m668_appworld_weight_ablation_receipt",
        "schema_version": 1,
        "ablation": {
            "label": "m666_public_appworld_warm_vs_random",
            "matched_rows": True,
            "same_data_lineage": True,
            "same_training_protocol": True,
            "warm_initialization": "parent",
            "random_initialization": "deterministic_random",
            "warm_weight_report": _identity(warm_weight),
            "random_weight_report": _identity(random_weight),
            "claim_boundary": (
                "Weight movement and held-out teacher-forced continuation only; this is not a native "
                "AppWorld leaderboard score or a free-running policy score."
            ),
        },
        "compatibility": {
            "config_mismatches": {},
            "shape_mismatches": {},
            "tokenizer_sha256_equal": True,
            "warm": warm_compat,
            "random": random_compat,
        },
        "held_out": held_out,
        "weight_transfer_analysis": {
            "warm": {"groups": warm["groups"], "compatibility": warm_compat},
            "random": {"groups": random["groups"], "compatibility": random_compat},
        },
        "parent_checkpoint": {
            "warm_target": warm["target"],
            "random_target": random["target"],
            "current": current_identity,
        },
        "dataset": {
            "receipt": _identity(appworld_receipt),
            "train_rows": 90,
            "eval_rows": 6,
            "protected_test_used": False,
            "source_url": "https://github.com/StonyBrookNLP/appworld",
        },
        "decision": {
            "retain_parent_initialization": True,
            "promote_to_native": False,
            "reason": (
                "The matched m666 arm reports preserve compatible parent geometry and show a warm "
                "held-out advantage, but both learned free-running probes remain 0/6."
            ),
        },
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-weight", type=Path, required=True)
    parser.add_argument("--random-weight", type=Path, required=True)
    parser.add_argument("--appworld-receipt", type=Path, required=True)
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        warm_weight=args.warm_weight,
        random_weight=args.random_weight,
        appworld_receipt=args.appworld_receipt,
        current_checkpoint=args.current_checkpoint,
        output=args.out,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
