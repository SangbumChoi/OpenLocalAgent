#!/usr/bin/env python3
"""Seal the current-checkpoint MCP trajectory warm/random transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET = "obaydata/mcp-agent-trajectory-benchmark"
DATASET_URL = "https://huggingface.co/datasets/obaydata/mcp-agent-trajectory-benchmark"
REVISION = "f4f449d65271abc1e4ccd5157d121a59a1dd38c4"
WARM_PARENT = "8c3a4ed31d468a8556faf9f1442a3d7cfb7cfcc19788055bb1c501217c65d850"
RANDOM_PARENT = "0cb06efb0037bb96c94846ff42e33063525aa418d8063be07444bb0d27ef1ffc"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _arm(report: dict[str, Any], *, name: str, parent_sha: str) -> dict[str, Any]:
    if report.get("kind") != "localagent_public_agent_continuation_report":
        raise ValueError(f"{name} report kind mismatch")
    if report.get("parent", {}).get("sha256") != parent_sha:
        raise ValueError(f"{name} report is not bound to the expected m666 parent")
    if report.get("rows") != {"train": 86, "eval": 21}:
        raise ValueError(f"{name} row count mismatch")
    hyper = report.get("hyperparameters", {})
    expected = {"steps": 64, "batch_size": 4, "max_seq_len": 512, "learning_rate": 1.0e-5}
    for key, value in expected.items():
        if hyper.get(key) != value:
            raise ValueError(f"{name} hyperparameter {key} mismatch")
    source = report.get("source", {})
    if source.get("dataset") != DATASET or source.get("revision") != REVISION:
        raise ValueError(f"{name} source mismatch")
    transfer = report.get("weight_transfer")
    if not isinstance(transfer, dict) or transfer.get("compatibility", {}).get("config_mismatches"):
        raise ValueError(f"{name} compatibility gate failed")
    if transfer.get("compatibility", {}).get("tokenizer_sha256_equal") is not True:
        raise ValueError(f"{name} tokenizer compatibility gate failed")
    if transfer.get("groups", {}).get("action_heads", {}).get("delta_l2") != 0.0:
        raise ValueError(f"{name} action heads were not frozen")
    if report.get("after", {}).get("eval", {}).get("assistant_sequence_accuracy") != 0.0:
        raise ValueError(f"{name} exact sequence metric unexpectedly changed")
    return {
        "parent": report["parent"],
        "child": report["child"],
        "before": report["before"],
        "after": report["after"],
        "weight_transfer": transfer,
        "heads": report["heads"],
    }


def assemble(warm_path: Path, random_path: Path, output: Path) -> dict[str, Any]:
    warm_report = _load(warm_path)
    random_report = _load(random_path)
    warm = _arm(warm_report, name="warm", parent_sha=WARM_PARENT)
    random = _arm(random_report, name="random", parent_sha=RANDOM_PARENT)
    warm_before = float(warm["before"]["eval"]["assistant_token_accuracy"])
    warm_after = float(warm["after"]["eval"]["assistant_token_accuracy"])
    random_before = float(random["before"]["eval"]["assistant_token_accuracy"])
    random_after = float(random["after"]["eval"]["assistant_token_accuracy"])
    payload: dict[str, Any] = {
        "kind": "localagent_m671_m666_mcp_current_transfer",
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "url": DATASET_URL,
            "revision": REVISION,
            "license": "Apache-2.0",
            "manifest": warm_report["source"]["manifest"],
        },
        "parent_checkpoint": {
            "warm": warm["parent"],
            "random": random["parent"],
        },
        "protocol": {
            "train_rows": 86,
            "eval_rows": 21,
            "steps": 64,
            "batch_size": 4,
            "learning_rate": 1.0e-5,
            "max_seq_len": 512,
            "split_policy": "public train trajectories with deterministic agent-disjoint internal holdout",
            "official_test_split": False,
            "tool_outputs_and_reasoning": "excluded_from_projection",
        },
        "inputs": {
            "warm_report": _identity(warm_path),
            "random_report": _identity(random_path),
            "train": warm_report["train_inputs"],
            "eval": warm_report["eval_inputs"],
        },
        "arms": {"warm": warm, "random": random},
        "comparison": {
            "warm_before_eval_token_accuracy": warm_before,
            "warm_after_eval_token_accuracy": warm_after,
            "warm_gain_pp": (warm_after - warm_before) * 100.0,
            "random_before_eval_token_accuracy": random_before,
            "random_after_eval_token_accuracy": random_after,
            "random_gain_pp": (random_after - random_before) * 100.0,
            "warm_minus_random_after_pp": (warm_after - random_after) * 100.0,
            "warm_start_better_after": warm_after > random_after,
            "exact_sequence_accuracy": {
                "warm": warm["after"]["eval"]["assistant_sequence_accuracy"],
                "random": random["after"]["eval"]["assistant_sequence_accuracy"],
            },
        },
        "weight_adoption": {
            "config_compatible": True,
            "tokenizer_compatible": True,
            "action_heads_frozen": True,
            "reuse_warm_backbone": True,
            "backbone_learning_rate": "low",
            "new_or_domain_heads_learning_rate": "higher_than_backbone",
            "native_promotion": False,
            "reason": (
                "The warm initialization wins the held-out text projection by a large margin with "
                "compatible tensors and frozen action heads. This supports warm-start adoption and "
                "low-rate backbone continuation, but exact sequence accuracy remains zero."
            ),
        },
        "claim_boundary": (
            "Current m666-parent public MCP trajectory continuation with an agent-disjoint internal "
            "holdout. This is teacher-forced text evidence, not an official MCP benchmark score, "
            "native MCP server execution, or real email/Notion/browser side effect."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = assemble(args.warm_report, args.random_report, args.out)
    print(json.dumps(receipt["comparison"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
