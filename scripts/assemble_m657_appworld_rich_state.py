"""Seal rich-observation AppWorld continuation and native negative control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


WARM_PARENT = "9c760ff270d12c797db937c94366ef33e801cac1f9fbee2c0015184f7837111e"
RANDOM_PARENT = "f39869027d7822f454fcd181d24cc139b4286e7837d349856c497633675559cd"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _check_training(report: dict[str, Any], parent: str) -> None:
    if report.get("kind") != "localagent_public_agent_continuation_report":
        raise ValueError("training report kind mismatch")
    if report.get("parent", {}).get("sha256") != parent:
        raise ValueError("training parent mismatch")
    if report.get("source", {}).get("dataset") != "appworld_ground_truth_api_trajectory_rich":
        raise ValueError("rich trajectory dataset mismatch")
    if report.get("rows") != {"train": 64, "eval": 18}:
        raise ValueError("training split mismatch")


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
    warm = _load(warm_report_path)
    random = _load(random_report_path)
    warm_weight = _load(warm_weight_path)
    random_weight = _load(random_weight_path)
    warm_native = _load(warm_native_path)
    random_native = _load(random_native_path)
    train_manifest = _load(train_manifest_path)
    eval_manifest = _load(eval_manifest_path)
    _check_training(warm, WARM_PARENT)
    _check_training(random, RANDOM_PARENT)
    if warm["hyperparameters"] != random["hyperparameters"]:
        raise ValueError("warm/random hyperparameters differ")
    if warm["train_inputs"] != random["train_inputs"]:
        raise ValueError("warm/random train identities differ")
    for report in (warm_weight, random_weight):
        if report.get("kind") != "localagent_weight_transfer_analysis":
            raise ValueError("weight report kind mismatch")
    for report in (warm_native, random_native):
        if report.get("kind") != "localagent_appworld_checkpoint_free_running_trajectory_probe":
            raise ValueError("native report kind mismatch")
        if report.get("summary", {}).get("tasks") != 6:
            raise ValueError("native task count mismatch")
    warm_after = warm["after"]["eval"]["assistant_token_accuracy"]
    random_after = random["after"]["eval"]["assistant_token_accuracy"]
    payload: dict[str, Any] = {
        "kind": "localagent_m657_appworld_rich_state_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "train_rows": 64,
            "eval_rows": 18,
            "max_actions_per_row": 3,
            "rich_observations": True,
            "protected_test_used": False,
            "train_truncated_rows": sum(bool(t["truncated"]) for t in train_manifest["tasks"]),
            "eval_truncated_rows": sum(bool(t["truncated"]) for t in eval_manifest["tasks"]),
        },
        "metrics": {
            "warm_before": warm["before"]["eval"],
            "warm_after": warm["after"]["eval"],
            "random_before": random["before"]["eval"],
            "random_after": random["after"]["eval"],
            "warm_delta_pp": 100.0 * (warm_after - warm["before"]["eval"]["assistant_token_accuracy"]),
            "random_delta_pp": 100.0 * (random_after - random["before"]["eval"]["assistant_token_accuracy"]),
            "warm_after_minus_random_after_pp": 100.0 * (warm_after - random_after),
            "warm_exact_sequence_accuracy": warm["after"]["eval"]["assistant_sequence_accuracy"],
            "random_exact_sequence_accuracy": random["after"]["eval"]["assistant_sequence_accuracy"],
        },
        "weight_movement": {"warm": warm_weight["groups"], "random": random_weight["groups"]},
        "native_replay": {
            "warm": {"report": _identity(warm_native_path), "summary": warm_native["summary"]},
            "random": {"report": _identity(random_native_path), "summary": random_native["summary"]},
        },
        "decision": {
            "rich_observations_help_teacher_forcing": warm_after > random_after,
            "promote_to_native_success": False,
            "reason": (
                "Safe response values improve the warm teacher-forced score to 63.22%, but all 18 "
                "evaluation rows are context-truncated at the 2,048-token model window and the paired "
                "native probe remains 0/6 for both arms (3 warm versus 10 random actions). Rich state "
                "is therefore a useful training direction, not deployment evidence."
            ),
        },
        "claim_boundary": (
            "Public AppWorld train/dev trajectory continuation with bounded safe response summaries; "
            "sensitive credentials, addresses, canaries, and tokens are redacted. Metrics are "
            "teacher-forced and the native run is a bounded diagnostic, not an official leaderboard "
            "score, complete task success, live email/Notion side effect, or WebGPU promotion."
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
    for name in ("warm-report", "random-report", "warm-weight", "random-weight", "warm-native", "random-native", "train-manifest", "eval-manifest"):
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
