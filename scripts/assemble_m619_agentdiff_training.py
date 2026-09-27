#!/usr/bin/env python3
"""Assemble matched Agent-Diff train/test continuation and weight-movement evidence."""

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


def assemble(warm_report: Path, random_report: Path, warm_weight: Path, random_weight: Path) -> dict[str, Any]:
    warm = json.loads(warm_report.read_text(encoding="utf-8"))
    random = json.loads(random_report.read_text(encoding="utf-8"))
    warm_movement = json.loads(warm_weight.read_text(encoding="utf-8"))
    random_movement = json.loads(random_weight.read_text(encoding="utf-8"))
    for report in (warm, random):
        if report.get("kind") != "localagent_public_agent_continuation_report":
            raise ValueError("unexpected continuation report kind")
        if report["rows"] != {"train": 179, "eval": 45}:
            raise ValueError("unexpected Agent-Diff train/test row counts")
    if warm["source"]["dataset"] != "hubertmarek/agent-diff-bench":
        raise ValueError("unexpected Agent-Diff source")
    body: dict[str, Any] = {
        "kind": "localagent_m619_agentdiff_train_transfer",
        "schema_version": 1,
        "dataset": {
            "name": warm["source"]["dataset"],
            "revision": warm["source"]["revision"],
            "train_rows": 179,
            "test_rows": 45,
            "test_policy": "eval_only",
            "train_manifest": warm["source"]["manifest"],
        },
        "arms": {
            "warm": {
                "report": _identity(warm_report),
                "checkpoint": warm["child"],
                "before_test": warm["before"]["eval"],
                "after_test": warm["after"]["eval"],
                "after_train": warm["after"]["train"],
                "weight_movement": {"report": _identity(warm_weight), "groups": warm_movement["groups"]},
            },
            "random": {
                "report": _identity(random_report),
                "checkpoint": random["child"],
                "before_test": random["before"]["eval"],
                "after_test": random["after"]["eval"],
                "after_train": random["after"]["train"],
                "weight_movement": {"report": _identity(random_weight), "groups": random_movement["groups"]},
            },
        },
        "comparison": {
            "before_test_warm_minus_random_pp": 100.0
            * (warm["before"]["eval"]["assistant_token_accuracy"] - random["before"]["eval"]["assistant_token_accuracy"]),
            "after_test_warm_minus_random_pp": 100.0
            * (warm["after"]["eval"]["assistant_token_accuracy"] - random["after"]["eval"]["assistant_token_accuracy"]),
            "warm_test_gain_pp": 100.0
            * (warm["after"]["eval"]["assistant_token_accuracy"] - warm["before"]["eval"]["assistant_token_accuracy"]),
            "random_test_gain_pp": 100.0
            * (random["after"]["eval"]["assistant_token_accuracy"] - random["before"]["eval"]["assistant_token_accuracy"]),
            "warm_beats_random_after": warm["after"]["eval"]["assistant_token_accuracy"]
            > random["after"]["eval"]["assistant_token_accuracy"],
        },
        "decision": {
            "reuse_warm_initialization": True,
            "admit_test_rows_to_training": False,
            "export_to_webgpu": False,
            "native_promotion": False,
        },
        "claim_boundary": (
            "Matched 32-step SFT continuation on the public Agent-Diff train split with the 45-row "
            "test split held out. This is assertion-text teacher-forcing and weight movement, not "
            "sandbox state-diff success, native API execution, an official leaderboard score, or a "
            "WebGPU export recommendation."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--warm-weight", type=Path, required=True)
    parser.add_argument("--random-weight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    report = assemble(args.warm_report, args.random_report, args.warm_weight, args.random_weight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
