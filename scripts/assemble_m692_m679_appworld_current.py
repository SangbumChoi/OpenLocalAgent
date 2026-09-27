#!/usr/bin/env python3
"""Seal a current m679 public AppWorld trajectory continuation and control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET = "appworld"
PARENT = "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
TRAIN_SHA = "4484056c9df0c1c31eb3178fd4f6245fd2b743c718e3a234b23149995f48b6bd"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate(report: dict[str, Any], *, name: str) -> None:
    if report.get("kind") != "localagent_public_agent_continuation_report":
        raise ValueError(f"{name} report kind mismatch")
    if report.get("parent", {}).get("sha256") != PARENT:
        raise ValueError(f"{name} parent mismatch")
    if report.get("source", {}).get("dataset") != DATASET:
        raise ValueError(f"{name} dataset mismatch")
    if report.get("rows") != {"train": 90, "eval": 6}:
        raise ValueError(f"{name} split mismatch")
    if report.get("train_inputs", [{}])[0].get("sha256") != TRAIN_SHA:
        raise ValueError(f"{name} train input mismatch")
    expected = {"steps": 32, "batch_size": 2, "max_seq_len": 2048, "learning_rate": 1.0e-5}
    if any(report.get("hyperparameters", {}).get(k) != v for k, v in expected.items()):
        raise ValueError(f"{name} hyperparameters mismatch")
    compatibility = report.get("weight_transfer", {}).get("compatibility", {})
    if compatibility.get("config_mismatches") or compatibility.get("shape_mismatches"):
        raise ValueError(f"{name} tensor compatibility mismatch")
    if compatibility.get("tokenizer_sha256_equal") is not True:
        raise ValueError(f"{name} tokenizer mismatch")


def _arm(report: dict[str, Any], path: Path) -> dict[str, Any]:
    transfer = report["weight_transfer"]
    return {
        "report": _identity(path),
        "parent": report["parent"],
        "child": report["child"],
        "before": report["before"]["eval"],
        "after": report["after"]["eval"],
        "heads": {"before": report["heads"]["before"], "after": report["heads"]["after"]},
        "weight_groups": transfer["groups"],
        "compatibility": transfer["compatibility"],
    }


def assemble(warm_path: Path, random_path: Path, output: Path) -> dict[str, Any]:
    warm_report, random_report = _load(warm_path), _load(random_path)
    _validate(warm_report, name="warm")
    _validate(random_report, name="random")
    warm, random = _arm(warm_report, warm_path), _arm(random_report, random_path)
    warm_before = float(warm["before"]["assistant_token_accuracy"])
    warm_after = float(warm["after"]["assistant_token_accuracy"])
    random_before = float(random["before"]["assistant_token_accuracy"])
    random_after = float(random["after"]["assistant_token_accuracy"])
    payload: dict[str, Any] = {
        "kind": "localagent_m692_m679_appworld_current",
        "schema_version": 1,
        "benchmark_id": "appworld_text_trajectory_projection",
        "environment_executed": False,
        "official_split_verified": False,
        "source": {
            "dataset": "AppWorld",
            "url": "https://github.com/StonyBrookNLP/appworld",
            "data_version": "0.2.0",
            "train_tasks": 90,
            "dev_tasks": 6,
            "train_manifest": warm_report["source"]["manifest"],
            "train_input": warm_report["train_inputs"],
            "eval_input": warm_report["eval_inputs"],
            "bootstrap_credentials_removed": True,
            "rich_observations": True,
            "max_actions": 64,
        },
        "protocol": {
            "steps": 32,
            "batch_size": 2,
            "learning_rate": 1.0e-5,
            "max_seq_len": 2048,
            "split_policy": "public train tasks only; separate public dev tasks for evaluation",
            "observation_policy": "bounded redacted API summaries; no credentials or task databases",
            "official_native_score": False,
        },
        "parent_checkpoint": warm["parent"],
        "arms": {"warm": warm, "random": random},
        "comparison": {
            "warm_before_eval_token_accuracy": warm_before,
            "warm_after_eval_token_accuracy": warm_after,
            "warm_gain_pp": 100.0 * (warm_after - warm_before),
            "random_before_eval_token_accuracy": random_before,
            "random_after_eval_token_accuracy": random_after,
            "random_gain_pp": 100.0 * (random_after - random_before),
            "warm_minus_random_after_pp": 100.0 * (warm_after - random_after),
            "warm_start_better_after": warm_after > random_after,
            "exact_sequence_accuracy": {
                "warm": warm["after"]["assistant_sequence_accuracy"],
                "random": random["after"]["assistant_sequence_accuracy"],
            },
        },
        "weight_adoption": {
            "config_compatible": True,
            "tokenizer_compatible": True,
            "warm_action_heads_frozen": warm["weight_groups"]["action_heads"]["delta_l2"] == 0.0,
            "warm_body_relative_delta_max": max(
                warm["weight_groups"][name]["relative_delta_l2"]
                for name in ("embedding", "attention_or_mixer", "ffn", "normalization")
            ),
            "random_action_head_relative_delta": random["weight_groups"]["action_heads"]["relative_delta_l2"],
            "recommendation": "retain m679 backbone and train API heads separately; require native AppWorld and real email/Notion service validation before promotion",
        },
        "claim_boundary": (
            "Public AppWorld train/dev ground-truth API trajectory continuation with redacted, "
            "bounded observations. No AppWorld environment, email service, Notion service, emulator, "
            "browser, or external account executed; this is not an official AppWorld score."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.warm_report, args.random_report, args.out)["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
