#!/usr/bin/env python3
"""Assemble a matched m607 warm/random transfer evaluation on AgentWorldBench."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def assemble(warm_path: Path, random_path: Path, transfer_path: Path) -> dict[str, Any]:
    warm = json.loads(warm_path.read_text(encoding="utf-8"))
    random = json.loads(random_path.read_text(encoding="utf-8"))
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    for report in (warm, random):
        if report.get("kind") != "localagent_agentworldbench_text_projection_eval":
            raise ValueError("unexpected AgentWorldBench projection receipt kind")
        if report["protocol"]["rows"] != 224 or report["manifest"]["train_policy"] != "eval_only":
            raise ValueError("AgentWorldBench projection is not the expected 224-row eval-only set")
    if transfer.get("kind") != "localagent_m607_current_policy_transfer":
        raise ValueError("unexpected m607 transfer receipt kind")
    warm_overall = warm["metrics"]["overall"]
    random_overall = random["metrics"]["overall"]
    by_domain = {}
    for domain in warm["metrics"]["by_domain"]:
        warm_metric = warm["metrics"]["by_domain"][domain]
        random_metric = random["metrics"]["by_domain"][domain]
        by_domain[domain] = {
            "rows": warm_metric["rows"],
            "warm_token_accuracy": warm_metric["assistant_token_accuracy"],
            "random_token_accuracy": random_metric["assistant_token_accuracy"],
            "warm_minus_random_pp": 100.0
            * (warm_metric["assistant_token_accuracy"] - random_metric["assistant_token_accuracy"]),
            "warm_mean_loss": warm_metric["mean_loss"],
            "random_mean_loss": random_metric["mean_loss"],
        }
    body: dict[str, Any] = {
        "kind": "localagent_m617_agentworldbench_transfer_eval",
        "schema_version": 1,
        "dataset": {
            "name": "Qwen/AgentWorldBench",
            "source_revision": warm["manifest"]["source_revision"],
            "manifest_sha256": warm["manifest"]["sha256"],
            "rows": warm["protocol"]["rows"],
            "domains": warm["protocol"]["domains"],
            "train_policy": "eval_only",
        },
        "arms": {
            "warm": {"projection_report": _identity(warm_path), "checkpoint": warm["checkpoint"]},
            "random": {"projection_report": _identity(random_path), "checkpoint": random["checkpoint"]},
        },
        "parent_transfer": {
            "receipt": _identity(transfer_path),
            "parent_checkpoint": transfer["parent_checkpoint"],
            "warm_body_relative_l2": transfer["weight_transfer"]["warm"]["groups"],
            "random_body_relative_l2": transfer["weight_transfer"]["random"]["groups"],
        },
        "metrics": {
            "overall": {
                "rows": warm_overall["rows"],
                "warm_token_accuracy": warm_overall["assistant_token_accuracy"],
                "random_token_accuracy": random_overall["assistant_token_accuracy"],
                "warm_minus_random_pp": 100.0
                * (warm_overall["assistant_token_accuracy"] - random_overall["assistant_token_accuracy"]),
                "warm_mean_loss": warm_overall["mean_loss"],
                "random_mean_loss": random_overall["mean_loss"],
                "warm_exact_sequence_accuracy": warm_overall["assistant_sequence_accuracy"],
                "random_exact_sequence_accuracy": random_overall["assistant_sequence_accuracy"],
            },
            "by_domain": by_domain,
        },
        "decision": {
            "warm_wins_token_accuracy": warm_overall["assistant_token_accuracy"]
            > random_overall["assistant_token_accuracy"],
            "reuse_warm_backbone_as_initialization_candidate": True,
            "export_to_webgpu": False,
            "native_promotion": False,
        },
        "claim_boundary": (
            "Matched warm/random teacher-forced projection on the public AgentWorldBench test rows. "
            "AgentWorldBench remains eval-only: this is not an official judge score, action success, "
            "screenshot score, native Android/OS/browser/MCP execution, or proof that transfer is optimal."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, default=Path("docs/paper/results/raw/m607-m585-policy-aligned-transfer-v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    report = assemble(args.warm, args.random, args.transfer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
