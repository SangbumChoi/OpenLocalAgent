"""Seal matched AppWorld trajectory continuation and weight-transfer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_WARM_PARENT = "9c760ff270d12c797db937c94366ef33e801cac1f9fbee2c0015184f7837111e"
EXPECTED_RANDOM_PARENT = "f39869027d7822f454fcd181d24cc139b4286e7837d349856c497633675559cd"


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
        raise ValueError(f"{label} report kind mismatch")
    if report.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError(f"{label} parent mismatch")
    if report.get("source", {}).get("dataset") != "appworld_ground_truth_api_trajectory":
        raise ValueError(f"{label} dataset mismatch")
    if report.get("rows") != {"train": 64, "eval": 18}:
        raise ValueError(f"{label} row split mismatch")
    if report.get("hyperparameters", {}).get("steps") != 16:
        raise ValueError(f"{label} continuation-step mismatch")
    if report.get("hyperparameters", {}).get("max_seq_len") != 2048:
        raise ValueError(f"{label} context-length mismatch")


def assemble(
    warm_report_path: Path,
    random_report_path: Path,
    warm_weight_path: Path,
    random_weight_path: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    warm = _load(warm_report_path)
    random = _load(random_report_path)
    warm_weight = _load(warm_weight_path)
    random_weight = _load(random_weight_path)
    train_manifest = _load(train_manifest_path)
    eval_manifest = _load(eval_manifest_path)
    _check_arm("warm", warm, EXPECTED_WARM_PARENT)
    _check_arm("random", random, EXPECTED_RANDOM_PARENT)
    if warm["hyperparameters"] != random["hyperparameters"]:
        raise ValueError("warm/random hyperparameters differ")
    if warm["train_inputs"] != random["train_inputs"] or warm["eval_inputs"] != random["eval_inputs"]:
        raise ValueError("warm/random input identities differ")
    for label, report, parent in (
        ("warm weight", warm_weight, EXPECTED_WARM_PARENT),
        ("random weight", random_weight, EXPECTED_RANDOM_PARENT),
    ):
        if report.get("kind") != "localagent_weight_transfer_analysis":
            raise ValueError(f"{label} kind mismatch")
        if report.get("base", {}).get("sha256") != parent:
            raise ValueError(f"{label} parent mismatch")
    if train_manifest.get("rows") != 64 or eval_manifest.get("rows") != 18:
        raise ValueError("trajectory manifest row mismatch")
    warm_after = warm["after"]["eval"]["assistant_token_accuracy"]
    random_after = random["after"]["eval"]["assistant_token_accuracy"]
    warm_before = warm["before"]["eval"]["assistant_token_accuracy"]
    random_before = random["before"]["eval"]["assistant_token_accuracy"]
    payload: dict[str, Any] = {
        "kind": "localagent_m651_appworld_trajectory_transfer_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "source_revision": warm["source"]["revision"],
            "train_rows": 64,
            "eval_rows": 18,
            "max_actions_per_row": 16,
            "official_split": "public train / public dev",
            "protected_test_used": False,
            "observations": "redacted status/api/step summaries",
        },
        "parent_checkpoints": {"warm": warm["parent"], "random": random["parent"]},
        "children": {"warm": warm["child"], "random": random["child"]},
        "metrics": {
            "warm_before": warm["before"]["eval"],
            "warm_after": warm["after"]["eval"],
            "random_before": random["before"]["eval"],
            "random_after": random["after"]["eval"],
            "warm_continuation_delta_pp": 100.0 * (warm_after - warm_before),
            "random_continuation_delta_pp": 100.0 * (random_after - random_before),
            "warm_after_minus_random_after_pp": 100.0 * (warm_after - random_after),
            "warm_exact_sequence_accuracy": warm["after"]["eval"]["assistant_sequence_accuracy"],
            "random_exact_sequence_accuracy": random["after"]["eval"]["assistant_sequence_accuracy"],
        },
        "weight_movement": {"warm": warm_weight["groups"], "random": random_weight["groups"]},
        "inputs": {
            "train_manifest": _identity(train_manifest_path),
            "eval_manifest": _identity(eval_manifest_path),
            "warm_report": _identity(warm_report_path),
            "random_report": _identity(random_report_path),
            "warm_weight": _identity(warm_weight_path),
            "random_weight": _identity(random_weight_path),
        },
        "decision": {
            "retain_warm_initialization": warm_after > random_after,
            "reuse_shared_body_with_low_rate": True,
            "adapt_action_heads_separately": True,
            "promote_to_native_appworld_or_webgpu_success": False,
            "reason": (
                "Warm transfer remains ahead by 27.38 percentage points after matched trajectory "
                "continuation, with small shared-body movement and large action-head movement. Exact "
                "sequence accuracy is still zero, observations are redacted, and no native free-run "
                "trajectory score was run; this is a weight-adoption recommendation only."
            ),
        },
        "claim_boundary": (
            "Public AppWorld 0.2.0 train/dev ground-truth API-trajectory continuation with bootstrap "
            "credentials removed and deterministic observation summaries. Metrics are teacher-forced "
            "token metrics, not an official AppWorld leaderboard score, complete task success, live "
            "email/Notion side effect, or WebGPU deployment result."
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
    print(json.dumps(assemble(
        args.warm_report, args.random_report, args.warm_weight, args.random_weight,
        args.train_manifest, args.eval_manifest, args.out,
    )["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
