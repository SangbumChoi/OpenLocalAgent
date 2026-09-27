#!/usr/bin/env python3
"""Seal a current-checkpoint AndroidControl warm/random continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET = "OfficerChul/Android-Control-84k"
REVISION = "0248027f747c9d57bd09c14e8f044f9a8103dddd"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected regular file: {path}")
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate(report: dict[str, Any], *, name: str, parent_sha: str) -> None:
    if report.get("kind") != "localagent_public_agent_continuation_report":
        raise ValueError(f"{name} report kind mismatch")
    if report.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError(f"{name} parent checkpoint mismatch")
    if report.get("source", {}).get("dataset") != DATASET:
        raise ValueError(f"{name} dataset mismatch")
    if report.get("source", {}).get("revision") != REVISION:
        raise ValueError(f"{name} revision mismatch")
    if report.get("rows") != {"train": 512, "eval": 256}:
        raise ValueError(f"{name} row counts are not the declared public split")
    expected = {"steps": 32, "batch_size": 4, "max_seq_len": 512, "learning_rate": 1.0e-5}
    for key, value in expected.items():
        if report.get("hyperparameters", {}).get(key) != value:
            raise ValueError(f"{name} hyperparameter mismatch: {key}")
    compatibility = report.get("weight_transfer", {}).get("compatibility", {})
    if compatibility.get("config_mismatches") != {} or compatibility.get("shape_mismatches") != {}:
        raise ValueError(f"{name} tensor compatibility failed")
    if compatibility.get("tokenizer_sha256_equal") is not True:
        raise ValueError(f"{name} tokenizer compatibility failed")


def _arm(report: dict[str, Any]) -> dict[str, Any]:
    transfer = report["weight_transfer"]
    return {
        "parent": report["parent"],
        "child": report["child"],
        "before": report["before"]["eval"],
        "after": report["after"]["eval"],
        "heads": {"before": report["heads"]["before"], "after": report["heads"]["after"]},
        "weight_transfer": {
            "compatibility": transfer["compatibility"],
            "groups": transfer["groups"],
            "recommendation": transfer["recommendation"],
        },
    }


def assemble(warm_path: Path, random_path: Path, output: Path) -> dict[str, Any]:
    warm_report, random_report = _load(warm_path), _load(random_path)
    parent_sha = warm_report.get("parent", {}).get("sha256")
    if not isinstance(parent_sha, str) or parent_sha != random_report.get("parent", {}).get("sha256"):
        raise ValueError("warm/random reports must share one parent checkpoint")
    _validate(warm_report, name="warm", parent_sha=parent_sha)
    _validate(random_report, name="random", parent_sha=parent_sha)
    warm, random = _arm(warm_report), _arm(random_report)
    warm_before = float(warm["before"]["assistant_token_accuracy"])
    warm_after = float(warm["after"]["assistant_token_accuracy"])
    random_before = float(random["before"]["assistant_token_accuracy"])
    random_after = float(random["after"]["assistant_token_accuracy"])
    payload: dict[str, Any] = {
        "kind": "localagent_m691_m679_androidcontrol_current",
        "schema_version": 1,
        "benchmark_id": "androidcontrol_text_projection",
        "environment_executed": False,
        "official_split_verified": False,
        "source": {
            "dataset": DATASET,
            "url": "https://huggingface.co/datasets/OfficerChul/Android-Control-84k",
            "original_project": "https://github.com/google-research/google-research/tree/master/android_control",
            "revision": REVISION,
            "train_manifest": warm_report["source"]["manifest"],
            "train_rows": warm_report["train_inputs"],
            "eval_rows": warm_report["eval_inputs"],
            "screenshots": "omitted_from_projection",
        },
        "protocol": {
            "train_rows": 512,
            "eval_rows": 256,
            "steps": 32,
            "batch_size": 4,
            "learning_rate": 1.0e-5,
            "max_seq_len": 512,
            "split_policy": "pinned public train/eval mirror; no eval rows used for SFT",
            "official_native_score": False,
        },
        "parent_checkpoint": warm["parent"],
        "arms": {"warm": warm, "random": random},
        "comparison": {
            "warm_before_eval_token_accuracy": warm_before,
            "warm_after_eval_token_accuracy": warm_after,
            "warm_gain_pp": (warm_after - warm_before) * 100.0,
            "random_before_eval_token_accuracy": random_before,
            "random_after_eval_token_accuracy": random_after,
            "random_gain_pp": (random_after - random_before) * 100.0,
            "warm_minus_random_after_pp": (warm_after - random_after) * 100.0,
            "warm_start_better_after": warm_after > random_after,
            "exact_sequence_accuracy": {
                "warm": warm["after"]["assistant_sequence_accuracy"],
                "random": random["after"]["assistant_sequence_accuracy"],
            },
        },
        "weight_adoption": {
            "config_compatible": True,
            "tokenizer_compatible": True,
            "warm_action_heads_frozen": warm["weight_transfer"]["groups"]["action_heads"]["delta_l2"] == 0.0,
            "warm_body_relative_delta_max": max(
                warm["weight_transfer"]["groups"][name]["relative_delta_l2"]
                for name in ("embedding", "attention_or_mixer", "ffn", "normalization")
            ),
            "random_action_head_relative_delta": random["weight_transfer"]["groups"]["action_heads"]["relative_delta_l2"],
            "random_body_relative_delta_max": max(
                random["weight_transfer"]["groups"][name]["relative_delta_l2"]
                for name in ("embedding", "attention_or_mixer", "ffn", "normalization")
            ),
            "recommendation": "retain the m679 backbone for mobile text continuation, but add visual grounding and native emulator validation before promotion",
        },
        "raw_reports": {"warm": _identity(warm_path), "random": _identity(random_path)},
        "claim_boundary": (
            "Current m679 public AndroidControl mirror continuation with teacher-forced text/action "
            "metrics. Screenshots were omitted and no emulator was launched; this is not visual "
            "mobile control, AndroidWorld or MobileGym success, or external-account behavior."
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
    payload = assemble(args.warm_report, args.random_report, args.out)
    print(json.dumps(payload["comparison"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
