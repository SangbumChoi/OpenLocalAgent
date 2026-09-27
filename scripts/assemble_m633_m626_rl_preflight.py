#!/usr/bin/env python3
"""Bind the current m626 RL-preflight report into a tracked receipt.

The runner writes a verbose isolated report under ``/private/tmp``.  This assembler keeps the
checkpoint lineage, split audit, optimizer transition, reward/held-out metrics, and raw-report
identity in the paper results directory without copying the isolated checkpoint or prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CURRENT_CHECKPOINT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RL preflight report must be a JSON object")
    return payload


def assemble(raw_path: Path, output: Path) -> dict[str, Any]:
    raw = _load(raw_path)
    if raw.get("kind") != "localagent_one_update_training_preflight":
        raise ValueError("unexpected RL preflight kind")
    if raw.get("status") != "passed":
        raise ValueError("refusing to bind a non-passed RL preflight")
    parent = raw.get("metrics", {}).get("lineage", {}).get("parent_checkpoint_sha256")
    if parent != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("RL preflight is not bound to the current m626 checkpoint")
    measurement = raw.get("measurement", {})
    heldout = raw.get("metrics", {}).get("heldout_eval", {})
    payload: dict[str, Any] = {
        "kind": raw["kind"],
        "schema_version": raw.get("schema_version"),
        "status": raw["status"],
        "current_checkpoint": raw.get("source", {}).get("sft_parent_checkpoint"),
        "protocol": {
            "environment": raw.get("effective", {}).get("config_payload", {}).get("environment"),
            "train_eval_row_overlap": raw.get("metrics", {})
            .get("data", {})
            .get("split_audit", {})
            .get("row_overlap"),
            "steps": raw.get("effective", {}).get("contract", {}).get("rollout_steps"),
            "prompts_per_step": raw.get("effective", {}).get("contract", {}).get("prompts_per_step"),
            "group_size": raw.get("effective", {}).get("contract", {}).get("group_size"),
            "max_new_tokens": raw.get("effective", {}).get("contract", {}).get("max_new_tokens"),
            "device": raw.get("metrics", {}).get("execution", {}).get("resolved_device"),
            "production_output_untouched": raw.get("source", {}).get("production_rl_output_untouched"),
            "source_artifacts_untouched": raw.get("source", {}).get("source_artifacts_untouched"),
        },
        "lineage": raw.get("metrics", {}).get("lineage"),
        "metrics": {"lineage": raw.get("metrics", {}).get("lineage")},
        "measurement": {
            "realized_optimizer_updates": measurement.get("policy_transition", {}).get(
                "realized_optimizer_updates", raw.get("metrics", {}).get("rl_accounting", {}).get("realized_optimizer_updates")
            ),
            "learning_rate_history": measurement.get("policy_transition", {}).get("actual_learning_rates"),
            "changed_model_parameter_count": measurement.get("policy_transition", {}).get("changed_model_parameter_count"),
            "informative_groups": raw.get("metrics", {}).get("rl_accounting", {}).get("informative_groups"),
            "attempted_rollouts": raw.get("metrics", {}).get("rl_accounting", {}).get("attempted_rollouts"),
            "exact_success_rollouts": raw.get("metrics", {}).get("rl_accounting", {}).get("rollout_observability", {}).get("reward", {}).get("exact_success_rollouts"),
            "reward_distribution": raw.get("metrics", {}).get("rl_accounting", {}).get("rollout_observability", {}).get("reward", {}).get("distribution"),
            "heldout_pre_mean_reward": heldout.get("pre", {}).get("mean_reward"),
            "heldout_post_mean_reward": heldout.get("post", {}).get("mean_reward"),
            "heldout_post_exact_match": heldout.get("post", {}).get("exact_match_accuracy"),
            "heldout_post_tool_format_valid_rate": heldout.get("post", {}).get("tool_format_valid_rate"),
        },
        "weight_transition": {
            "contract": measurement.get("policy_transition", {}).get("contract"),
            "all_named_policy_tensors_changed": measurement.get("policy_transition", {}).get("at_least_one_policy_tensor_changed"),
            "changed_model_parameter_count": measurement.get("policy_transition", {}).get("changed_model_parameter_count"),
            "overall_relative_l2": None,
            "by_component": {},
            "interpretation": "The isolated preflight proves optimizer and checkpoint transition only; it does not establish task success or a production RL checkpoint.",
        },
        "raw_report": _identity(raw_path),
        "claim_boundary": (
            "Strict one-update RL simulation over a deterministic local email/Notion/browser "
            "state machine. It proves current-checkpoint lineage, isolated execution, optimizer "
            "transition, and measured reward only; it is not native API control, a public "
            "benchmark score, real-account execution, or a production RL checkpoint."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.raw, args.out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
