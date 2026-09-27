#!/usr/bin/env python3
"""Assemble the matched MCP trajectory warm/random continuation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET = "obaydata/mcp-agent-trajectory-benchmark"
DATASET_URL = "https://huggingface.co/datasets/obaydata/mcp-agent-trajectory-benchmark"
REVISION = "f4f449d65271abc1e4ccd5157d121a59a1dd38c4"


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def assemble(warm_path: Path, random_path: Path, manifest_path: Path) -> dict[str, Any]:
    warm = json.loads(warm_path.read_text(encoding="utf-8"))
    random = json.loads(random_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    warm_before = warm["before"]["eval"]
    warm_after = warm["after"]["eval"]
    random_before = random["before"]["eval"]
    random_after = random["after"]["eval"]
    report: dict[str, Any] = {
        "kind": "localagent_mcp_trajectory_transfer_v1",
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "url": DATASET_URL,
            "revision": REVISION,
            "license": "Apache-2.0",
            "normalization_manifest": _identity(manifest_path),
            "manifest_sha256": manifest.get("manifest_sha256"),
        },
        "protocol": {
            "public_train_trajectories": 38,
            "internal_train_agents": 30,
            "internal_eval_agents": 8,
            "train_rows": warm["rows"]["train"],
            "eval_rows": warm["rows"]["eval"],
            "steps": warm["hyperparameters"]["steps"],
            "batch_size": warm["hyperparameters"]["batch_size"],
            "lr": warm["hyperparameters"]["learning_rate"],
            "max_seq_len": warm["hyperparameters"]["max_seq_len"],
            "target": "user_to_tool_calls_json",
            "official_test_split": False,
            "split_policy": "sorted_agent_name_first_30_train_last_8_eval",
            "multi_conv_trajectories_seen": 11,
            "multi_conv_invalid_json": 1,
        },
        "arms": {
            "warm": {
                "report": _identity(warm_path),
                "checkpoint": warm["child"],
                "before_eval": warm_before,
                "after_eval": warm_after,
                "weight_transfer": warm["weight_transfer"],
            },
            "random": {
                "report": _identity(random_path),
                "checkpoint": random["child"],
                "before_eval": random_before,
                "after_eval": random_after,
                "weight_transfer": random["weight_transfer"],
            },
        },
        "comparison": {
            "warm_eval_token_accuracy_before": warm_before["assistant_token_accuracy"],
            "warm_eval_token_accuracy_after": warm_after["assistant_token_accuracy"],
            "warm_eval_gain": warm_after["assistant_token_accuracy"] - warm_before["assistant_token_accuracy"],
            "random_eval_token_accuracy_before": random_before["assistant_token_accuracy"],
            "random_eval_token_accuracy_after": random_after["assistant_token_accuracy"],
            "random_eval_gain": random_after["assistant_token_accuracy"] - random_before["assistant_token_accuracy"],
            "warm_minus_random_after": warm_after["assistant_token_accuracy"] - random_after["assistant_token_accuracy"],
            "warm_exact_sequence_accuracy": warm_after["assistant_sequence_accuracy"],
            "random_exact_sequence_accuracy": random_after["assistant_sequence_accuracy"],
        },
        "claim_boundary": (
            "Internal agent-disjoint teacher-forced tool-call projection. The Hub release has no official "
            "held-out split; this 30/8 split is an explicit structural holdout. No MCP server, tool output, "
            "reasoning trace, native evaluator, or external side effect was executed; this is not an official "
            "MCP benchmark score."
        ),
    }
    report["receipt_self_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    report = assemble(args.warm_report, args.random_report, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
