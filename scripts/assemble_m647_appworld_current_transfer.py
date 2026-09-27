#!/usr/bin/env python3
"""Seal the matched current-checkpoint AppWorld first-action continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CURRENT_WARM_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
CURRENT_RANDOM_SHA256 = "390f1414260e118cd621af735fe6e87b01e8641b1cff650d594585e39b212e45"
DATASET_URL = "https://github.com/StonyBrookNLP/appworld"
DATA_VERSION = "0.2.0"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _check_arm(label: str, report: dict[str, Any], parent_sha: str) -> None:
    if report.get("kind") != "localagent_public_agent_continuation_report":
        raise ValueError(f"{label} continuation kind mismatch")
    if report.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError(f"{label} parent mismatch")
    if report.get("source", {}).get("dataset") != "appworld_action_step":
        raise ValueError(f"{label} dataset mismatch")
    if report.get("rows") != {"train": 64, "eval": 18}:
        raise ValueError(f"{label} row selection mismatch")
    if report.get("hyperparameters", {}).get("steps") != 32:
        raise ValueError(f"{label} continuation steps mismatch")


def assemble(
    warm_path: Path,
    random_path: Path,
    warm_weight_path: Path,
    random_weight_path: Path,
    train_manifest: Path,
    eval_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    warm = _load(warm_path)
    random = _load(random_path)
    warm_weight = _load(warm_weight_path)
    random_weight = _load(random_weight_path)
    _check_arm("warm", warm, CURRENT_WARM_SHA256)
    _check_arm("random", random, CURRENT_RANDOM_SHA256)
    if warm["hyperparameters"] != random["hyperparameters"]:
        raise ValueError("warm/random hyperparameters differ")
    if warm["train_inputs"] != random["train_inputs"] or warm["eval_inputs"] != random["eval_inputs"]:
        raise ValueError("warm/random input identities differ")
    if warm_weight.get("kind") != "localagent_weight_transfer_analysis":
        raise ValueError("warm weight report kind mismatch")
    if random_weight.get("kind") != "localagent_weight_transfer_analysis":
        raise ValueError("random weight report kind mismatch")
    if warm_weight.get("base", {}).get("sha256") != CURRENT_WARM_SHA256:
        raise ValueError("warm weight base mismatch")
    if random_weight.get("base", {}).get("sha256") != CURRENT_RANDOM_SHA256:
        raise ValueError("random weight base mismatch")
    warm_after = warm["after"]["eval"]
    random_after = random["after"]["eval"]
    warm_before = warm["before"]["eval"]
    random_before = random["before"]["eval"]
    payload: dict[str, Any] = {
        "kind": "localagent_m647_appworld_current_transfer_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": DATA_VERSION,
            "source_url": DATASET_URL,
            "source_revision": warm["source"]["revision"],
            "train_rows": 64,
            "eval_rows": 18,
            "official_split": "public train / public dev",
            "protected_test_used": False,
        },
        "parent_checkpoints": {
            "warm": warm["parent"],
            "random": random["parent"],
        },
        "children": {"warm": warm["child"], "random": random["child"]},
        "inputs": {
            "train": warm["train_inputs"],
            "eval": warm["eval_inputs"],
            "train_manifest": _identity(train_manifest),
            "eval_manifest": _identity(eval_manifest),
            "warm_report": _identity(warm_path),
            "random_report": _identity(random_path),
            "warm_weight": _identity(warm_weight_path),
            "random_weight": _identity(random_weight_path),
        },
        "metrics": {
            "warm_before": warm_before,
            "warm_after": warm_after,
            "random_before": random_before,
            "random_after": random_after,
            "warm_after_minus_random_after_token_accuracy_pp": 100.0
            * (warm_after["assistant_token_accuracy"] - random_after["assistant_token_accuracy"]),
            "warm_continuation_delta_pp": 100.0
            * (warm_after["assistant_token_accuracy"] - warm_before["assistant_token_accuracy"]),
            "random_continuation_delta_pp": 100.0
            * (random_after["assistant_token_accuracy"] - random_before["assistant_token_accuracy"]),
        },
        "weight_movement": {
            "warm": warm_weight["groups"],
            "random": random_weight["groups"],
            "compatibility": {
                "warm": warm_weight["compatibility"],
                "random": random_weight["compatibility"],
            },
        },
        "decision": {
            "retain_warm_initialization": warm_after["assistant_token_accuracy"] > random_after["assistant_token_accuracy"],
            "freeze_or_low_rate_backbone": True,
            "adapt_action_heads_separately": True,
            "promote_to_native_appworld_or_webgpu_success": False,
            "reason": (
                "Warm transfer remains ahead after matched continuation, while sequence exactness and "
                "route/selector accuracy remain zero; the next bottleneck is free-run schema/action "
                "grounding rather than shared-body initialization."
            ),
        },
        "claim_boundary": (
            "Public AppWorld 0.2.0 train/dev first non-bootstrap API-action projection with source-local "
            "disjoint rows and matched warm/random continuation from the current m626 checkpoints. "
            "This is teacher-forced token evidence only: no protected test split, complete AppWorld "
            "task success, external account, email side effect, or WebGPU/native promotion is claimed."
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
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--warm-weight", type=Path, required=True)
    parser.add_argument("--random-weight", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        args.warm_report,
        args.random_report,
        args.warm_weight,
        args.random_weight,
        args.train_manifest,
        args.eval_manifest,
        args.out,
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
