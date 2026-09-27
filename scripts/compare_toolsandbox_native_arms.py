#!/usr/bin/env python3
"""Compare matched native ToolSandbox scripted-user receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_KIND = "localagent_toolsandbox_native_smoke"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt must be an object: {path}")
    return value


def _scenario_signature(receipt: dict[str, Any]) -> list[str]:
    return [str(record["scenario"]) for record in receipt["scenarios"]]


def _assert_matched(warm: dict[str, Any], random: dict[str, Any]) -> None:
    if warm.get("kind") != EXPECTED_KIND or random.get("kind") != EXPECTED_KIND:
        raise ValueError("ToolSandbox receipt kind mismatch")
    if warm.get("benchmark_id") != "toolsandbox" or random.get("benchmark_id") != "toolsandbox":
        raise ValueError("ToolSandbox benchmark id mismatch")
    for key in ("source_revision", "task_count", "protocol", "max_agent_turns"):
        if warm.get(key) != random.get(key):
            raise ValueError(f"matched receipt mismatch for {key}")
    if warm.get("environment_executed") is not True or random.get("environment_executed") is not True:
        raise ValueError("native environment was not executed")
    if warm.get("official_split_verified") is not False or random.get("official_split_verified") is not False:
        raise ValueError("this bounded probe must not be labeled as the official split")
    if _scenario_signature(warm) != _scenario_signature(random):
        raise ValueError("scenario set mismatch")
    if warm.get("runner", {}).get("sha256") != random.get("runner", {}).get("sha256"):
        raise ValueError("runner hash mismatch")
    if warm.get("checkpoint", {}).get("sha256") == random.get("checkpoint", {}).get("sha256"):
        raise ValueError("warm and random checkpoints must differ")


def compare(warm: dict[str, Any], random: dict[str, Any]) -> dict[str, Any]:
    """Build a matched per-scenario native ToolSandbox comparison."""

    _assert_matched(warm, random)
    scenarios: dict[str, Any] = {}
    for warm_record, random_record in zip(warm["scenarios"], random["scenarios"], strict=True):
        name = str(warm_record["scenario"])
        scenarios[name] = {
            "warm_similarity": float(warm_record["similarity"]),
            "random_similarity": float(random_record["similarity"]),
            "warm_minus_random_similarity": float(warm_record["similarity"])
            - float(random_record["similarity"]),
            "warm_milestone_similarity": float(warm_record["milestone_similarity"]),
            "random_milestone_similarity": float(random_record["milestone_similarity"]),
            "warm_turn_count": int(warm_record["turn_count"]),
            "random_turn_count": int(random_record["turn_count"]),
        }
    warm_success = int(warm["success_count"])
    random_success = int(random["success_count"])
    return {
        "kind": "localagent_toolsandbox_native_head_ablation_report",
        "schema_version": 1,
        "benchmark_id": "toolsandbox",
        "source_revision": warm["source_revision"],
        "runner": warm["runner"],
        "scenario_names": _scenario_signature(warm),
        "task_count": int(warm["task_count"]),
        "warm_receipt": warm["checkpoint"],
        "random_receipt": random["checkpoint"],
        "warm_success_count": warm_success,
        "random_success_count": random_success,
        "warm_success_rate": float(warm["success_rate"]),
        "random_success_rate": float(random["success_rate"]),
        "warm_minus_random_success_rate_pp": 100.0
        * (float(warm["success_rate"]) - float(random["success_rate"])),
        "scenarios": scenarios,
        "decision": (
            "warm_head_native_advantage_on_bounded_toolsandbox_probe"
            if warm_success > random_success
            else "no_warm_native_advantage_on_bounded_toolsandbox_probe"
        ),
        "claim_boundary": (
            "Native pinned ToolSandbox simulator/verifier evidence over five selected scenarios with "
            "a bounded scripted user; official split, model-based user simulator, full scenario "
            "matrix, and optional RapidAPI tools were not executed. This is not an official "
            "ToolSandbox score or evidence of real email/Notion/MCP side effects."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(_load(args.warm_report), _load(args.random_report))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
