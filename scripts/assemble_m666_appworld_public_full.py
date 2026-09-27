"""Seal the full public-train AppWorld continuation and schema-planner control."""

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


def _check_training(report: dict[str, Any], *, parent_sha: str, train_sha: str) -> None:
    if report.get("kind") != "localagent_public_agent_continuation_report":
        raise ValueError("continuation kind mismatch")
    if report.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError("continuation parent mismatch")
    if report.get("rows") != {"train": 90, "eval": 6}:
        raise ValueError("continuation split mismatch")
    if report.get("train_inputs", [{}])[0].get("sha256") != train_sha:
        raise ValueError("continuation train input mismatch")


def _check_native(report: dict[str, Any], *, checkpoint_sha: str, expected_success: int) -> None:
    if report.get("kind") != "localagent_appworld_checkpoint_free_running_trajectory_probe":
        raise ValueError("native report kind mismatch")
    if report.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("native checkpoint mismatch")
    summary = report.get("summary", {})
    if summary.get("tasks") != 6 or summary.get("native_successes") != expected_success:
        raise ValueError("native task or success mismatch")


def assemble(
    *,
    warm_report: Path,
    random_report: Path,
    warm_weight: Path,
    random_weight: Path,
    warm_free: Path,
    random_free: Path,
    warm_planner: Path,
    random_planner: Path,
    train_manifest: Path,
    eval_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    warm, random = _load(warm_report), _load(random_report)
    warm_weight_payload, random_weight_payload = _load(warm_weight), _load(random_weight)
    warm_free_payload, random_free_payload = _load(warm_free), _load(random_free)
    warm_planner_payload, random_planner_payload = _load(warm_planner), _load(random_planner)
    train_manifest_payload, eval_manifest_payload = _load(train_manifest), _load(eval_manifest)

    warm_parent = warm.get("parent", {}).get("sha256")
    random_parent = random.get("parent", {}).get("sha256")
    if not isinstance(warm_parent, str) or not isinstance(random_parent, str):
        raise ValueError("missing continuation parent identity")
    train_sha = _identity(Path(warm["train_inputs"][0]["path"]))["sha256"]
    _check_training(warm, parent_sha=warm_parent, train_sha=train_sha)
    _check_training(random, parent_sha=random_parent, train_sha=train_sha)
    if warm["hyperparameters"] != random["hyperparameters"]:
        raise ValueError("warm/random hyperparameters differ")
    for report in (warm_weight_payload, random_weight_payload):
        if report.get("kind") != "localagent_weight_transfer_analysis":
            raise ValueError("weight report kind mismatch")
        if report.get("compatibility", {}).get("config_mismatches"):
            raise ValueError("weight config mismatch")
        if report.get("compatibility", {}).get("tokenizer_sha256_equal") is not True:
            raise ValueError("weight tokenizer mismatch")
    warm_sha = warm.get("child", {}).get("sha256")
    random_sha = random.get("child", {}).get("sha256")
    if not isinstance(warm_sha, str) or not isinstance(random_sha, str):
        raise ValueError("missing continuation child identity")
    _check_native(warm_free_payload, checkpoint_sha=warm_sha, expected_success=0)
    _check_native(random_free_payload, checkpoint_sha=random_sha, expected_success=0)
    _check_native(warm_planner_payload, checkpoint_sha=warm_sha, expected_success=6)
    _check_native(random_planner_payload, checkpoint_sha=random_sha, expected_success=6)

    warm_before = warm["before"]["eval"]["assistant_token_accuracy"]
    warm_after = warm["after"]["eval"]["assistant_token_accuracy"]
    random_before = random["before"]["eval"]["assistant_token_accuracy"]
    random_after = random["after"]["eval"]["assistant_token_accuracy"]
    payload: dict[str, Any] = {
        "kind": "localagent_m666_appworld_public_full_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld",
            "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "train_tasks": 90,
            "eval_tasks": 6,
            "protected_test_used": False,
            "rich_observations": True,
            "max_actions": 64,
        },
        "metrics": {
            "warm_before": warm["before"]["eval"],
            "warm_after": warm["after"]["eval"],
            "random_before": random["before"]["eval"],
            "random_after": random["after"]["eval"],
            "warm_delta_pp": 100.0 * (warm_after - warm_before),
            "random_delta_pp": 100.0 * (random_after - random_before),
            "warm_after_minus_random_after_pp": 100.0 * (warm_after - random_after),
            "warm_free_running_success_rate": warm_free_payload["summary"]["native_success_rate"],
            "random_free_running_success_rate": random_free_payload["summary"]["native_success_rate"],
            "warm_structured_planner_success_rate": warm_planner_payload["summary"]["native_success_rate"],
            "random_structured_planner_success_rate": random_planner_payload["summary"]["native_success_rate"],
        },
        "weight_movement": {"warm": warm_weight_payload["groups"], "random": random_weight_payload["groups"]},
        "native_replay": {
            "free_running": {
                "warm": {"report": _identity(warm_free), "summary": warm_free_payload["summary"]},
                "random": {"report": _identity(random_free), "summary": random_free_payload["summary"]},
            },
            "structured_planner_control": {
                "warm": {"report": _identity(warm_planner), "summary": warm_planner_payload["summary"]},
                "random": {"report": _identity(random_planner), "summary": random_planner_payload["summary"]},
                "configuration": {
                    "allow_completion": True,
                    "lexical_first": True,
                    "ground_truth_actions_injected": False,
                    "environment_side_schema_planner": True,
                },
            },
        },
        "decision": {
            "retain_warm_initialization": True,
            "promote_to_native_success": False,
            "structured_planner_is_model_score": False,
            "reason": (
                "Using all 90 public train tasks adds held-out API coverage and raises warm token "
                f"accuracy from {100 * warm_before:.2f}% to {100 * warm_after:.2f}% versus "
                f"{100 * random_after:.2f}% for the random control. Fully free-running model ranking "
                "remains 0/6 for both arms. The explicit schema-planner control reaches 6/6 for both "
                "arms, proving the persisted executor, argument extraction, and completion verifier "
                "are live while isolating the remaining learned action-selection gap."
            ),
        },
        "claim_boundary": (
            "Public AppWorld train/dev continuation and paired resettable native diagnostics. The "
            "6/6 structured-planner result is an environment-side schema policy control, not a model "
            "score, official AppWorld leaderboard result, external-account side effect, or WebGPU "
            "email/Notion promotion."
        ),
        "inputs": {
            "planner_source": _identity(Path("scripts/evaluate_appworld_checkpoint.py")),
            "warm_report": _identity(warm_report),
            "random_report": _identity(random_report),
            "warm_weight": _identity(warm_weight),
            "random_weight": _identity(random_weight),
            "warm_free": _identity(warm_free),
            "random_free": _identity(random_free),
            "warm_planner": _identity(warm_planner),
            "random_planner": _identity(random_planner),
            "train_manifest": _identity(train_manifest),
            "eval_manifest": _identity(eval_manifest),
            "train_manifest_self_sha256": train_manifest_payload.get("manifest_self_sha256"),
            "eval_manifest_self_sha256": eval_manifest_payload.get("manifest_self_sha256"),
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
        "warm-report", "random-report", "warm-weight", "random-weight", "warm-free", "random-free",
        "warm-planner", "random-planner", "train-manifest", "eval-manifest",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        warm_report=args.warm_report,
        random_report=args.random_report,
        warm_weight=args.warm_weight,
        random_weight=args.random_weight,
        warm_free=args.warm_free,
        random_free=args.random_free,
        warm_planner=args.warm_planner,
        random_planner=args.random_planner,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        output=args.out,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
