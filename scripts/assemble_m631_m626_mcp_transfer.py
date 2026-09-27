#!/usr/bin/env python3
"""Seal a fresh m626-parent MCP trajectory warm/random ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_cross_surface_controls import compare


PARENT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
DATASET = "obaydata/mcp-agent-trajectory-benchmark"
DATASET_URL = "https://huggingface.co/datasets/obaydata/mcp-agent-trajectory-benchmark"
REVISION = "f4f449d65271abc1e4ccd5157d121a59a1dd38c4"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _check_arm(report: dict[str, Any], init: str) -> None:
    if report.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("unexpected report kind")
    if report.get("parent", {}).get("sha256") != PARENT_SHA256:
        raise ValueError("report is not bound to m626")
    if report.get("rows") != {"train": 86, "eval": 21}:
        raise ValueError("unexpected MCP row counts")
    hyper = report.get("hyperparameters", {})
    if hyper.get("backbone_init") != init or hyper.get("steps") != 32 or hyper.get("batch_size") != 4:
        raise ValueError("warm/random protocol mismatch")
    if hyper.get("max_seq_len") != 512:
        raise ValueError("unexpected max sequence length")
    for source in [*report.get("train_sources", []), *report.get("eval_sources", [])]:
        if source.get("label") != "mcp" or source.get("revisions") != [REVISION]:
            raise ValueError("MCP source identity mismatch")


def assemble(warm_path: Path, random_path: Path) -> dict[str, Any]:
    warm = _load(warm_path)
    random = _load(random_path)
    _check_arm(warm, "parent")
    _check_arm(random, "random")
    comparison = compare(warm, random)
    payload: dict[str, Any] = {
        "kind": "localagent_m631_m626_mcp_warm_random_transfer",
        "schema_version": 1,
        "parent_checkpoint": warm["parent"],
        "source": {
            "dataset": DATASET,
            "url": DATASET_URL,
            "revision": REVISION,
            "license": "Apache-2.0",
        },
        "protocol": {
            "train_rows": 86,
            "eval_rows": 21,
            "steps": 32,
            "batch_size": 4,
            "learning_rate": 1.0e-5,
            "max_seq_len": 512,
            "split_policy": "public train trajectories with deterministic agent-disjoint internal holdout",
            "official_test_split": False,
        },
        "inputs": {
            "warm_report": _identity(warm_path),
            "random_report": _identity(random_path),
            "train": warm["train_sources"],
            "eval": warm["eval_sources"],
        },
        "warm": {
            "child": warm["child"],
            "before": warm["before"],
            "after": warm["after"],
            "weight_transfer": warm["weight_transfer"],
        },
        "random": {
            "child": random["child"],
            "before": random["before"],
            "after": random["after"],
            "weight_transfer": random["weight_transfer"],
        },
        "comparison": comparison,
        "weight_transfer_analysis": {
            "warm": warm["weight_transfer"],
            "random": random["weight_transfer"],
            "comparison": comparison,
        },
        "decision": {
            "warm_start_better_after": comparison["aggregate"]["warm_start_better_after"],
            "warm_minus_random_after_pp": comparison["aggregate"]["warm_minus_random_after_pp"],
            "current_checkpoint_weight_gate": True,
            "native_promotion": False,
        },
        "claim_boundary": (
            "Fresh m626-parent MCP trajectory continuation and matched random-body ablation. This is "
            "an internal structural holdout, not an official MCP benchmark score, live server run, "
            "or evidence of external email/Notion side effects."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    receipt = assemble(args.warm_report, args.random_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
