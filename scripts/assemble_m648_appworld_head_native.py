#!/usr/bin/env python3
"""Seal AppWorld route/selector head adaptation and its bounded native replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


M647_WARM_CHILD = "039386d75721a12bfe9da636347302a51245c8e1571245711b66d4e13e1fe4a0"
M647_RANDOM_CHILD = "64329be82c3f8e963159b022c215985c12ba989759ae58e25f90b23367afdef9"
DATA_VERSION = "0.2.0"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _validate_arm(label: str, report: dict[str, Any], parent_sha: str) -> None:
    if report.get("kind") != "localagent_public_agent_continuation_report":
        raise ValueError(f"{label} continuation kind mismatch")
    if report.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError(f"{label} parent mismatch")
    if report.get("rows") != {"train": 64, "eval": 18}:
        raise ValueError(f"{label} row count mismatch")
    if report.get("hyperparameters", {}).get("head_steps") != 256:
        raise ValueError(f"{label} head-step mismatch")
    if report.get("source", {}).get("dataset") != "appworld_action_step_head":
        raise ValueError(f"{label} dataset mismatch")


def assemble(
    warm_report_path: Path,
    random_report_path: Path,
    warm_weight_path: Path,
    random_weight_path: Path,
    native_baseline_path: Path,
    native_head_path: Path,
    output: Path,
) -> dict[str, Any]:
    warm = _load(warm_report_path)
    random = _load(random_report_path)
    warm_weight = _load(warm_weight_path)
    random_weight = _load(random_weight_path)
    native_baseline = _load(native_baseline_path)
    native_head = _load(native_head_path)
    _validate_arm("warm", warm, M647_WARM_CHILD)
    _validate_arm("random", random, M647_RANDOM_CHILD)
    if warm["hyperparameters"] != random["hyperparameters"]:
        raise ValueError("warm/random head hyperparameters differ")
    for label, report in (("native baseline", native_baseline), ("native head", native_head)):
        if report.get("kind") != "localagent_appworld_checkpoint_native_probe":
            raise ValueError(f"{label} native kind mismatch")
        if report.get("runner", {}).get("data_version") != DATA_VERSION:
            raise ValueError(f"{label} data version mismatch")
        if report.get("summary", {}).get("tasks") != 6:
            raise ValueError(f"{label} native task count mismatch")
        if report.get("environment", {}).get("native_runtime_executed") is not True:
            raise ValueError(f"{label} was not native")
    payload: dict[str, Any] = {
        "kind": "localagent_m648_appworld_head_native_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": DATA_VERSION,
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "head_train_rows": 64,
            "head_eval_rows": 18,
            "native_tasks": 6,
            "native_split": "public dev subset",
            "protected_test_used": False,
        },
        "head_adaptation": {
            "warm": {
                "parent": warm["parent"],
                "child": warm["child"],
                "heads_before": warm["heads"]["before"],
                "heads_after": warm["heads"]["after"],
                "weight_movement": warm_weight["groups"],
            },
            "random": {
                "parent": random["parent"],
                "child": random["child"],
                "heads_before": random["heads"]["before"],
                "heads_after": random["heads"]["after"],
                "weight_movement": random_weight["groups"],
            },
            "matched": {
                "same_rows": True,
                "same_hyperparameters": True,
                "warm_route_after": warm["heads"]["after"]["route_accuracy"],
                "warm_selector_after": warm["heads"]["after"]["selector_top1_accuracy"],
                "random_route_after": random["heads"]["after"]["route_accuracy"],
                "random_selector_after": random["heads"]["after"]["selector_top1_accuracy"],
            },
        },
        "native_replay": {
            "baseline": {
                "report": _identity(native_baseline_path),
                "checkpoint": native_baseline["checkpoint"],
                "summary": native_baseline["summary"],
            },
            "head_adapted": {
                "report": _identity(native_head_path),
                "checkpoint": native_head["checkpoint"],
                "summary": native_head["summary"],
            },
            "paired_task_ids": native_baseline["configuration"]["tasks"],
            "same_task_order": native_baseline["configuration"]["tasks"]
            == native_head["configuration"]["tasks"],
        },
        "decision": {
            "retain_warm_body": True,
            "train_schema_action_heads": True,
            "promote_head_to_native_success": False,
            "reason": (
                "Head training reaches 100% route and selector accuracy on the held-out projection, "
                "but the paired native six-task replay remains 0/6 with zero API calls; the metric is "
                "a candidate-ranking diagnostic and does not establish free-run AppWorld success."
            ),
        },
        "claim_boundary": (
            "AppWorld 0.2.0 route/selector head adaptation on public train/dev first-action projections "
            "plus a six-task resettable native replay. The replay used no external accounts and no "
            "protected test tasks. This is not an AppWorld leaderboard score, complete task success, "
            "real email/SMS side effects, or WebGPU deployment evidence."
        ),
        "inputs": {
            "warm_report": _identity(warm_report_path),
            "random_report": _identity(random_report_path),
            "warm_weight": _identity(warm_weight_path),
            "random_weight": _identity(random_weight_path),
        },
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
    parser.add_argument("--native-baseline", type=Path, required=True)
    parser.add_argument("--native-head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        args.warm_report,
        args.random_report,
        args.warm_weight,
        args.random_weight,
        args.native_baseline,
        args.native_head,
        args.out,
    )
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
