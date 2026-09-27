#!/usr/bin/env python3
"""Compare matched warm and random stateful-productivity GRPO receipts.

The comparison keeps protocol identity separate from checkpoint identity: warm and random arms
must use the same task hashes, rollout budget, seed, reward environment, and deployment-head
contract, while their parent checkpoints are intentionally different.  This is a local RL
simulation ablation, not an official benchmark or native WebGPU result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt must be a JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _training_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    training = receipt["training"]
    accounting = training["rl_accounting"]
    return {
        "mean_reward_pre": float(training["mean_reward_pre"]),
        "mean_reward_post": float(training["mean_reward_post"]),
        "exact_match_accuracy_pre": float(training["exact_match_accuracy_pre"]),
        "exact_match_accuracy_post": float(training["exact_match_accuracy_post"]),
        "tool_exact_match_accuracy_pre": float(training["tool_exact_match_accuracy_pre"]),
        "tool_exact_match_accuracy_post": float(training["tool_exact_match_accuracy_post"]),
        "informative_groups": int(accounting["informative_groups"]),
        "realized_optimizer_updates": int(accounting["realized_optimizer_updates"]),
        "truncated_rollouts": int(accounting["truncated_rollouts"]),
    }


def _protocol(receipt: dict[str, Any]) -> dict[str, Any]:
    configuration = receipt["configuration"]
    source = receipt["source"]
    return {
        "suite": receipt["suite"],
        "train_task_hash": source["train"]["task_hash"],
        "eval_task_hash": source["eval"]["task_hash"],
        "train_tasks": source["train"]["tasks"],
        "eval_tasks": source["eval"]["tasks"],
        "model_config": configuration["model_config"],
        "steps": int(configuration["steps"]),
        "prompts_per_step": int(configuration["prompts_per_step"]),
        "group_size": int(configuration["group_size"]),
        "max_new_tokens": int(configuration["max_new_tokens"]),
        "seed": int(configuration["seed"]),
        "sft_steps": int(configuration["sft"]["steps"]),
        "sft_batch_size": int(configuration["sft"]["batch_size"]),
        "sft_lr": float(configuration["sft"]["lr"]),
        "reward_environment": configuration["reward_environment"],
        "deployment_heads": sorted(configuration["deployment_heads_preserved"]),
        "deployment_heads_trainable": bool(configuration["deployment_heads_trainable"]),
    }


def _assert_matched(warm: dict[str, Any], random: dict[str, Any]) -> dict[str, Any]:
    if warm.get("kind") != "localagent_stateful_productivity_grpo_simulation":
        raise ValueError("warm receipt kind mismatch")
    if random.get("kind") != "localagent_stateful_productivity_grpo_simulation":
        raise ValueError("random receipt kind mismatch")
    warm_protocol = _protocol(warm)
    random_protocol = _protocol(random)
    if warm_protocol != random_protocol:
        mismatches = {
            key: {"warm": warm_protocol.get(key), "random": random_protocol.get(key)}
            for key in sorted(set(warm_protocol) | set(random_protocol))
            if warm_protocol.get(key) != random_protocol.get(key)
        }
        raise ValueError(f"matched GRPO protocol mismatch: {mismatches}")
    for receipt in (warm, random):
        if receipt["source"]["native_runtime_executed"] is not False:
            raise ValueError("native runtime must be false for this local ablation")
        if receipt["source"]["public_benchmark_text_used"] is not False:
            raise ValueError("public benchmark text must be false for this local ablation")
    if warm["parent"]["original"]["sha256"] == random["parent"]["original"]["sha256"]:
        raise ValueError("warm and random parent checkpoints must differ")
    return warm_protocol


def _weight_groups(path: Path) -> dict[str, dict[str, float | int]]:
    payload = _load(path)
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"weight report has no groups: {path}")
    selected = {}
    for name in ("action_heads", "embedding", "attention_or_mixer", "ffn", "normalization"):
        group = groups.get(name)
        if not isinstance(group, dict):
            raise ValueError(f"weight report missing group {name!r}: {path}")
        selected[name] = {
            "relative_delta_l2": float(group["relative_delta_l2"]),
            "delta_l2": float(group["delta_l2"]),
            "parameters": int(group["parameters"]),
        }
    return selected


def _deployment_summary(path: Path) -> dict[str, Any]:
    payload = _load(path)
    summary = payload["summary"]
    return {
        "cases": int(summary["cases"]),
        "exact_tool": int(summary["exact_tool"]),
        "tool_accuracy": float(summary["tool_accuracy"]),
        "environment_executed": bool(payload["environment"]["environment_executed"]),
        "external_accounts": bool(payload["environment"]["external_accounts"]),
    }


def compare(
    warm_receipt: Path,
    random_receipt: Path,
    warm_weight: Path,
    random_weight: Path,
    warm_deployment: Path,
    random_deployment: Path,
) -> dict[str, Any]:
    warm = _load(warm_receipt)
    random = _load(random_receipt)
    protocol = _assert_matched(warm, random)
    warm_training = _training_summary(warm)
    random_training = _training_summary(random)
    warm_weights = _weight_groups(warm_weight)
    random_weights = _weight_groups(random_weight)
    warm_deploy = _deployment_summary(warm_deployment)
    random_deploy = _deployment_summary(random_deployment)
    reward_delta = warm_training["mean_reward_post"] - random_training["mean_reward_post"]
    exact_delta = (
        warm_training["exact_match_accuracy_post"]
        - random_training["exact_match_accuracy_post"]
    )
    deployment_delta = warm_deploy["tool_accuracy"] - random_deploy["tool_accuracy"]
    decision = (
        "warm_reward_signal_advantage_but_no_deployment_adoption"
        if reward_delta > 0.0 and deployment_delta <= 0.0
        else "no_warm_advantage_on_matched_local_grpo_probe"
    )
    body = {
        "kind": "localagent_stateful_productivity_grpo_matched_ablation",
        "schema_version": 1,
        "protocol": protocol,
        "arms": {
            "warm": {
                "receipt": _identity(warm_receipt),
                "parent": warm["parent"],
                "training": warm_training,
                "weight_report": _identity(warm_weight),
                "weight_groups": warm_weights,
                "deployment": warm_deploy,
            },
            "random": {
                "receipt": _identity(random_receipt),
                "parent": random["parent"],
                "training": random_training,
                "weight_report": _identity(random_weight),
                "weight_groups": random_weights,
                "deployment": random_deploy,
            },
        },
        "comparison": {
            "warm_minus_random_mean_reward_post": reward_delta,
            "warm_minus_random_exact_match_post": exact_delta,
            "warm_minus_random_deployment_tool_accuracy": deployment_delta,
            "warm_informative_groups": warm_training["informative_groups"],
            "random_informative_groups": random_training["informative_groups"],
            "warm_realized_optimizer_updates": warm_training["realized_optimizer_updates"],
            "random_realized_optimizer_updates": random_training["realized_optimizer_updates"],
        },
        "decision": decision,
        "claim_boundary": (
            "Matched pure-PyTorch local stateful-productivity GRPO simulation only. Warm and random "
            "arms share task hashes, rollout protocol, seed, and frozen deployment-head contract; "
            "the random arm uses shape-matched random backbone and heads. This is not public "
            "benchmark training, native Android/browser/MCP execution, real-account side effects, "
            "or WebGPU capability evidence."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-receipt", type=Path, required=True)
    parser.add_argument("--random-receipt", type=Path, required=True)
    parser.add_argument("--warm-weight", type=Path, required=True)
    parser.add_argument("--random-weight", type=Path, required=True)
    parser.add_argument("--warm-deployment", type=Path, required=True)
    parser.add_argument("--random-deployment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    report = compare(
        args.warm_receipt,
        args.random_receipt,
        args.warm_weight,
        args.random_weight,
        args.warm_deployment,
        args.random_deployment,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
