#!/usr/bin/env python3
"""Seal a longer AppWorld SFT continuation with native free-run controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PARENT_SHA = "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
TASKS = ["6bdbc26_1", "6bdbc26_2", "6bdbc26_3", "396c5a2_1", "396c5a2_2", "396c5a2_3"]


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


def _validate_sft(report: dict[str, Any], name: str, init: str) -> None:
    if report.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError(f"{name} SFT kind mismatch")
    if report.get("parent", {}).get("sha256") != PARENT_SHA:
        raise ValueError(f"{name} parent mismatch")
    if report.get("hyperparameters", {}).get("steps") != 128:
        raise ValueError(f"{name} step mismatch")
    if report.get("hyperparameters", {}).get("batch_size") != 2:
        raise ValueError(f"{name} batch mismatch")
    if report.get("hyperparameters", {}).get("learning_rate") != 1.0e-5:
        raise ValueError(f"{name} learning-rate mismatch")
    if report.get("hyperparameters", {}).get("backbone_init") != init:
        raise ValueError(f"{name} initialization mismatch")
    if report.get("rows") != {"train": 90, "eval": 6}:
        raise ValueError(f"{name} split mismatch")
    compatibility = report.get("weight_transfer", {}).get("compatibility", {})
    if compatibility.get("config_mismatches") or compatibility.get("shape_mismatches"):
        raise ValueError(f"{name} tensor compatibility mismatch")
    if compatibility.get("tokenizer_sha256_equal") is not True:
        raise ValueError(f"{name} tokenizer mismatch")


def _validate_native(report: dict[str, Any], name: str) -> None:
    if report.get("kind") != "localagent_appworld_checkpoint_free_running_trajectory_probe":
        raise ValueError(f"{name} native kind mismatch")
    config = report.get("configuration", {})
    if config.get("tasks") != TASKS or config.get("max_steps") != 4:
        raise ValueError(f"{name} native protocol mismatch")
    if report.get("environment", {}).get("native_runtime_executed") is not True:
        raise ValueError(f"{name} native runtime missing")
    if [row.get("task_id") for row in report.get("tasks", [])] != TASKS:
        raise ValueError(f"{name} native task rows mismatch")


def _arm(sft: dict[str, Any], native: dict[str, Any], sft_path: Path, native_path: Path) -> dict[str, Any]:
    groups = sft["weight_transfer"]["groups"]
    return {
        "sft_report": _identity(sft_path),
        "native_report": _identity(native_path),
        "parent": sft["parent"],
        "child": sft["child"],
        "before": sft["before"]["eval"],
        "after": sft["after"]["eval"],
        "weight_groups": groups,
        "native_summary": native["summary"],
        "native_tasks": [
            {
                "task_id": row["task_id"],
                "action_replayed": row["action_replayed"],
                "success": row["evaluation"]["success"],
                "apis": [step.get("api") for step in row["steps"] if step.get("api")],
                "stop_reasons": [step.get("stop_reason") for step in row["steps"] if step.get("stop_reason")],
            }
            for row in native["tasks"]
        ],
    }


def assemble(warm_sft: Path, random_sft: Path, warm_native: Path, random_native: Path, out: Path) -> dict[str, Any]:
    warm_report, random_report = _load(warm_sft), _load(random_sft)
    warm_native_report, random_native_report = _load(warm_native), _load(random_native)
    _validate_sft(warm_report, "warm", "parent")
    _validate_sft(random_report, "random", "random")
    _validate_native(warm_native_report, "warm")
    _validate_native(random_native_report, "random")
    warm = _arm(warm_report, warm_native_report, warm_sft, warm_native)
    random = _arm(random_report, random_native_report, random_sft, random_native)
    warm_before = float(warm["before"]["assistant_token_accuracy"])
    warm_after = float(warm["after"]["assistant_token_accuracy"])
    random_before = float(random["before"]["assistant_token_accuracy"])
    random_after = float(random["after"]["assistant_token_accuracy"])
    payload: dict[str, Any] = {
        "kind": "localagent_m696_m679_appworld_sft_native",
        "schema_version": 1,
        "benchmark_id": "appworld_public_train_sft_native_free_run",
        "source": {
            "dataset": "AppWorld",
            "url": "https://github.com/StonyBrookNLP/appworld",
            "data_version": "0.2.0",
            "train_tasks": 90,
            "dev_tasks": 6,
            "split_policy": "public train tasks only; separate public dev tasks for native evaluation",
            "observations": "bounded redacted API summaries; no credentials or task databases",
        },
        "protocol": {
            "sft_steps": 128,
            "batch_size": 2,
            "learning_rate": 1.0e-5,
            "max_seq_len": 2048,
            "native_max_steps": 4,
            "native_retrieve_k": 100,
            "native_schema_adapter": "strict one literal API call per step",
            "native_environment_reset_per_task": True,
            "native_external_accounts": False,
            "native_screenshots": False,
        },
        "parent_checkpoint": {"sha256": PARENT_SHA},
        "arms": {"warm": warm, "random": random},
        "comparison": {
            "warm_before_eval_token_accuracy": warm_before,
            "warm_after_eval_token_accuracy": warm_after,
            "warm_gain_pp": 100.0 * (warm_after - warm_before),
            "random_before_eval_token_accuracy": random_before,
            "random_after_eval_token_accuracy": random_after,
            "random_gain_pp": 100.0 * (random_after - random_before),
            "warm_minus_random_after_pp": 100.0 * (warm_after - random_after),
            "native_successes": {"warm": warm["native_summary"]["native_successes"], "random": random["native_summary"]["native_successes"]},
            "native_success_rate": {"warm": warm["native_summary"]["native_success_rate"], "random": random["native_summary"]["native_success_rate"]},
            "native_action_replayed": {"warm": warm["native_summary"]["action_replayed"], "random": random["native_summary"]["action_replayed"]},
        },
        "weight_adoption": {
            "config_compatible": True,
            "tokenizer_compatible": True,
            "warm_body_relative_delta_max": max(groups["relative_delta_l2"] for name, groups in warm["weight_groups"].items() if name != "action_heads"),
            "random_body_relative_delta_max": max(groups["relative_delta_l2"] for name, groups in random["weight_groups"].items() if name != "action_heads"),
            "warm_action_heads_frozen": warm["weight_groups"]["action_heads"]["delta_l2"] == 0.0,
            "random_action_heads_frozen": random["weight_groups"]["action_heads"]["delta_l2"] == 0.0,
            "decision": "retain warm shared weights only as an initialization candidate; do not promote the child because native task success remains 0/6",
        },
        "claim_boundary": (
            "Public AppWorld train-only 128-step multi-turn SFT with matched random initialization, followed by "
            "native resettable free-running evaluation on six public dev tasks. Warm token accuracy improves, "
            "but both arms score 0/6 native completion. This is not an official AppWorld/AppWorld-UL score, does "
            "not cover Gmail or Notion, and uses no external accounts, screenshots, or irreversible side effects."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-sft", type=Path, required=True)
    parser.add_argument("--random-sft", type=Path, required=True)
    parser.add_argument("--warm-native", type=Path, required=True)
    parser.add_argument("--random-native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.warm_sft, args.random_sft, args.warm_native, args.random_native, args.out)["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
