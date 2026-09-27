"""Seal AppWorld observation-grounding, completion-prefix, and persisted native evidence."""

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
    warm_completion_path: Path,
    random_completion_path: Path,
    api_head_report_path: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    warm, random = _load(warm_report_path), _load(random_report_path)
    warm_weight, random_weight = _load(warm_weight_path), _load(random_weight_path)
    warm_native, random_native = _load(warm_native_path), _load(random_native_path)
    warm_completion, random_completion = _load(warm_completion_path), _load(random_completion_path)
    api_head = _load(api_head_report_path)
    for report in (warm, random):
        if report.get("kind") != "localagent_public_agent_continuation_report":
            raise ValueError("continuation kind mismatch")
        if report.get("rows") != {"train": 64, "eval": 6}:
            raise ValueError("split mismatch")
        if report.get("hyperparameters", {}).get("steps") != 24:
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
        if report.get("summary", {}).get("native_successes") != 0:
            raise ValueError("unexpected free-running success")
    for report in (warm_completion, random_completion):
        if report.get("kind") != "localagent_appworld_completion_prefix_probe":
            raise ValueError("completion prefix kind mismatch")
        if not report.get("configuration", {}).get("ground_truth_prefix_injected"):
            raise ValueError("completion prefix was not explicitly marked as injected")
        if report.get("summary", {}).get("native_successes") != 6:
            raise ValueError("completion prefix must pass all six tasks")
    if api_head.get("kind") != "localagent_appworld_trajectory_api_head_training_report":
        raise ValueError("API-head report kind mismatch")
    warm_before = warm["before"]["eval"]["assistant_token_accuracy"]
    warm_after = warm["after"]["eval"]["assistant_token_accuracy"]
    random_after = random["after"]["eval"]["assistant_token_accuracy"]
    payload: dict[str, Any] = {
        "kind": "localagent_m663_appworld_grounding_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "train_rows": 64,
            "eval_rows": 6,
            "native_tasks": 6,
            "max_native_steps": 5,
            "completion_prefix_tasks": 6,
            "protected_test_used": False,
        },
        "metrics": {
            "warm_before": warm["before"]["eval"],
            "warm_after": warm["after"]["eval"],
            "random_before": random["before"]["eval"],
            "random_after": random["after"]["eval"],
            "warm_delta_pp": 100 * (warm_after - warm_before),
            "warm_after_minus_random_after_pp": 100 * (warm_after - random_after),
            "warm_exact_sequence_accuracy": warm["after"]["eval"]["assistant_sequence_accuracy"],
            "random_exact_sequence_accuracy": random["after"]["eval"]["assistant_sequence_accuracy"],
            "api_head_seen_label_accuracy": api_head["metrics"]["eval"]["accuracy"],
            "free_running_warm_success_rate": warm_native["summary"]["native_success_rate"],
            "free_running_random_success_rate": random_native["summary"]["native_success_rate"],
            "completion_prefix_warm_success_rate": warm_completion["summary"]["native_success_rate"],
            "completion_prefix_random_success_rate": random_completion["summary"]["native_success_rate"],
        },
        "weight_movement": {"warm": warm_weight["groups"], "random": random_weight["groups"]},
        "api_head": {
            "report": _identity(api_head_report_path),
            "classes": api_head["classes"],
            "metrics": api_head["metrics"],
        },
        "native_replay": {
            "free_running": {
                "warm": {"report": _identity(warm_native_path), "summary": warm_native["summary"]},
                "random": {"report": _identity(random_native_path), "summary": random_native["summary"]},
            },
            "ground_truth_prefix_completion": {
                "warm": {"report": _identity(warm_completion_path), "summary": warm_completion["summary"]},
                "random": {"report": _identity(random_completion_path), "summary": random_completion["summary"]},
            },
        },
        "decision": {
            "retain_warm_initialization": True,
            "promote_to_native_success": False,
            "reason": (
                "The follower-count state sketch and persisted AppWorld execution contract make the "
                "completion layer measurable: both arms complete all 6/6 tasks after an explicitly "
                "injected public ground-truth API prefix. Fully free-running success remains 0/6, "
                "with wrong first-API choices dominating; the remaining blocker is action selection and "
                "state-conditioned planning, not the final completion call."
            ),
        },
        "claim_boundary": (
            "Public AppWorld trajectory continuation, frozen API-head diagnostic, persisted native "
            "free-running replay, and ground-truth-prefix completion control. The prefix result is not "
            "a free-running benchmark score; no external accounts, email, Notion, or WebGPU side effect "
            "is claimed."
        ),
        "inputs": {
            "warm_report": _identity(warm_report_path),
            "random_report": _identity(random_report_path),
            "warm_weight": _identity(warm_weight_path),
            "random_weight": _identity(random_weight_path),
            "warm_native": _identity(warm_native_path),
            "random_native": _identity(random_native_path),
            "warm_completion": _identity(warm_completion_path),
            "random_completion": _identity(random_completion_path),
            "api_head_report": _identity(api_head_report_path),
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
        "random-native", "warm-completion", "random-completion", "api-head-report",
        "train-manifest", "eval-manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        args.warm_report, args.random_report, args.warm_weight, args.random_weight,
        args.warm_native, args.random_native, args.warm_completion, args.random_completion,
        args.api_head_report, args.train_manifest, args.eval_manifest, args.out,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
