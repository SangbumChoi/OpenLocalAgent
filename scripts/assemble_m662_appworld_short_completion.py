"""Seal the short public AppWorld completion continuation and native probe."""

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
    warm_report_path: Path,
    random_report_path: Path,
    warm_weight_path: Path,
    random_weight_path: Path,
    warm_native_path: Path,
    random_native_path: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    warm, random = _load(warm_report_path), _load(random_report_path)
    warm_weight, random_weight = _load(warm_weight_path), _load(random_weight_path)
    warm_native, random_native = _load(warm_native_path), _load(random_native_path)
    for report in (warm, random):
        if report.get("kind") != "localagent_public_agent_continuation_report":
            raise ValueError("continuation kind mismatch")
        if report.get("source", {}).get("dataset") != "appworld":
            raise ValueError("AppWorld source mismatch")
        if report.get("rows") != {"train": 6, "eval": 6}:
            raise ValueError("short split mismatch")
        if report.get("hyperparameters", {}).get("steps") != 48:
            raise ValueError("unexpected continuation steps")
    for report in (warm_weight, random_weight):
        if report.get("kind") != "localagent_weight_transfer_analysis":
            raise ValueError("weight report kind mismatch")
        if report.get("compatibility", {}).get("config_mismatches"):
            raise ValueError("checkpoint config mismatch")
        if not report.get("compatibility", {}).get("tokenizer_sha256_equal"):
            raise ValueError("checkpoint tokenizer mismatch")
    for report in (warm_native, random_native):
        if report.get("kind") != "localagent_appworld_checkpoint_free_running_trajectory_probe":
            raise ValueError("native kind mismatch")
        if report.get("summary", {}).get("tasks") != 6:
            raise ValueError("native task count mismatch")
        if not report.get("configuration", {}).get("allow_completion"):
            raise ValueError("completion probe was not enabled")
    warm_after = warm["after"]["eval"]["assistant_token_accuracy"]
    random_after = random["after"]["eval"]["assistant_token_accuracy"]
    warm_before = warm["before"]["eval"]["assistant_token_accuracy"]
    payload: dict[str, Any] = {
        "kind": "localagent_m662_appworld_short_completion_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "train_rows": 6,
            "eval_rows": 6,
            "max_actions_per_row": 4,
            "includes_supervisor_completion": True,
            "native_tasks": 6,
            "native_max_steps": 5,
            "retrieve_k": 100,
            "protected_test_used": False,
        },
        "metrics": {
            "warm_before": warm["before"]["eval"],
            "warm_after": warm["after"]["eval"],
            "random_before": random["before"]["eval"],
            "random_after": random["after"]["eval"],
            "warm_delta_pp": 100 * (warm_after - warm_before),
            "random_delta_pp": 100 * (random_after - random["before"]["eval"]["assistant_token_accuracy"]),
            "warm_after_minus_random_after_pp": 100 * (warm_after - random_after),
            "warm_exact_sequence_accuracy": warm["after"]["eval"]["assistant_sequence_accuracy"],
            "random_exact_sequence_accuracy": random["after"]["eval"]["assistant_sequence_accuracy"],
        },
        "weight_movement": {"warm": warm_weight["groups"], "random": random_weight["groups"]},
        "native_replay": {
            "warm": {"report": _identity(warm_native_path), "summary": warm_native["summary"]},
            "random": {"report": _identity(random_native_path), "summary": random_native["summary"]},
        },
        "decision": {
            "promote_to_native_success": False,
            "retain_short_completion_probe": True,
            "reason": (
                "The six-row held-out continuation raises warm teacher-forced token accuracy from "
                f"{100 * warm_before:.2f}% to "
                f"{100 * warm_after:.2f}% (random reaches {100 * random_after:.2f}%), but exact "
                "sequence accuracy and resettable native success remain 0/6. The strict completion "
                "candidate therefore exposes an action-selection/state-planning gap rather than a "
                "publishable agent success claim."
            ),
        },
        "claim_boundary": (
            "Public AppWorld train/dev short-task continuation with supervisor completion retained, "
            "paired warm/random weight movement, and resettable native replay. This is not an official "
            "leaderboard score, complete task success, external-account side effect, or WebGPU email/"
            "Notion promotion."
        ),
        "inputs": {
            "warm_report": _identity(warm_report_path),
            "random_report": _identity(random_report_path),
            "warm_weight": _identity(warm_weight_path),
            "random_weight": _identity(random_weight_path),
            "warm_native": _identity(warm_native_path),
            "random_native": _identity(random_native_path),
            "train_manifest": _identity(train_manifest_path),
            "eval_manifest": _identity(eval_manifest_path),
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
    for name in (
        "warm-report", "random-report", "warm-weight", "random-weight", "warm-native",
        "random-native", "train-manifest", "eval-manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        args.warm_report, args.random_report, args.warm_weight, args.random_weight,
        args.warm_native, args.random_native, args.train_manifest, args.eval_manifest, args.out,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
