#!/usr/bin/env python3
"""Assemble a matched warm/random Agent-Diff test projection receipt."""

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


def assemble(warm_path: Path, random_path: Path, train_manifest: Path) -> dict[str, Any]:
    warm = json.loads(warm_path.read_text(encoding="utf-8"))
    random = json.loads(random_path.read_text(encoding="utf-8"))
    train = json.loads(train_manifest.read_text(encoding="utf-8"))
    if warm.get("kind") != "localagent_agentdiff_text_projection_eval":
        raise ValueError("unexpected warm Agent-Diff report kind")
    if random.get("kind") != "localagent_agentdiff_text_projection_eval":
        raise ValueError("unexpected random Agent-Diff report kind")
    if warm["protocol"]["rows"] != 45 or random["protocol"]["rows"] != 45:
        raise ValueError("expected the official 45-row Agent-Diff test split")
    if warm["manifest"]["train_policy"] != "eval_only" or random["manifest"]["train_policy"] != "eval_only":
        raise ValueError("Agent-Diff test rows must remain eval-only")
    if train.get("split") != "train" or train.get("train_policy") != "train":
        raise ValueError("training manifest is not the expected public train split")
    warm_overall = warm["metrics"]["overall"]
    random_overall = random["metrics"]["overall"]
    by_service = {}
    for service in warm["metrics"]["by_service"]:
        w = warm["metrics"]["by_service"][service]
        r = random["metrics"]["by_service"][service]
        by_service[service] = {
            "rows": w["rows"],
            "warm_token_accuracy": w["assistant_token_accuracy"],
            "random_token_accuracy": r["assistant_token_accuracy"],
            "warm_minus_random_pp": 100.0 * (w["assistant_token_accuracy"] - r["assistant_token_accuracy"]),
            "warm_mean_loss": w["mean_loss"],
            "random_mean_loss": r["mean_loss"],
        }
    body: dict[str, Any] = {
        "kind": "localagent_m618_agentdiff_transfer_eval",
        "schema_version": 1,
        "dataset": {
            "name": warm["manifest"]["dataset"],
            "source_revision": warm["manifest"]["source_revision"],
            "test_manifest_sha256": warm["manifest"]["sha256"],
            "train_manifest_sha256": train["manifest_sha256"],
            "test_rows": warm["protocol"]["rows"],
            "train_rows": train["records"]["selected"],
            "services": warm["protocol"]["services"],
            "license": "MIT",
            "train_policy": "train_split_only;test_split_eval_only",
        },
        "arms": {
            "warm": {"projection_report": _identity(warm_path), "checkpoint": warm["checkpoint"]},
            "random": {"projection_report": _identity(random_path), "checkpoint": random["checkpoint"]},
        },
        "metrics": {
            "overall": {
                "rows": warm_overall["rows"],
                "warm_token_accuracy": warm_overall["assistant_token_accuracy"],
                "random_token_accuracy": random_overall["assistant_token_accuracy"],
                "warm_minus_random_pp": 100.0 * (warm_overall["assistant_token_accuracy"] - random_overall["assistant_token_accuracy"]),
                "warm_mean_loss": warm_overall["mean_loss"],
                "random_mean_loss": random_overall["mean_loss"],
                "warm_exact_sequence_accuracy": warm_overall["assistant_sequence_accuracy"],
                "random_exact_sequence_accuracy": random_overall["assistant_sequence_accuracy"],
            },
            "by_service": by_service,
        },
        "decision": {
            "warm_wins_token_accuracy": warm_overall["assistant_token_accuracy"] > random_overall["assistant_token_accuracy"],
            "reuse_warm_backbone_as_initialization_candidate": True,
            "export_to_webgpu": False,
            "native_promotion": False,
        },
        "claim_boundary": (
            "Matched warm/random teacher-forced assertion-text projection on the public Agent-Diff "
            "test split. This is not deterministic state-diff success, sandbox execution, API "
            "interaction, an official leaderboard score, or evidence of external side effects."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    report = assemble(args.warm, args.random, args.train_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
