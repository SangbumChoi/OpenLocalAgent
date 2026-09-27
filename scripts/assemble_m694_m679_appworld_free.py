#!/usr/bin/env python3
"""Seal the matched multi-step native AppWorld free-running probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASKS = ["6bdbc26_1", "6bdbc26_2", "6bdbc26_3", "396c5a2_1", "396c5a2_2", "396c5a2_3"]
PARENT_SHA = "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
M692_WARM_SHA = "d39a92aaf5144274ce23e8992d9e78ad3f138b2af347ffc8dce8156539b7a1cb"
M692_RANDOM_SHA = "0873136c164ab2ea3ebe9c6137bcecfae329ac84176f083b3d83fae9d67f135f"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate(report: dict[str, Any], *, name: str, checkpoint_sha: str) -> None:
    if report.get("kind") != "localagent_appworld_checkpoint_free_running_trajectory_probe":
        raise ValueError(f"{name} report kind mismatch")
    if report.get("schema_version") != 1:
        raise ValueError(f"{name} schema mismatch")
    if report.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError(f"{name} checkpoint mismatch")
    configuration = report.get("configuration", {})
    if configuration.get("tasks") != TASKS:
        raise ValueError(f"{name} task list mismatch")
    if configuration.get("max_steps") != 4 or configuration.get("retrieve_k") != 100:
        raise ValueError(f"{name} protocol mismatch")
    if configuration.get("allow_completion") is not True:
        raise ValueError(f"{name} completion policy mismatch")
    if report.get("environment", {}).get("native_runtime_executed") is not True:
        raise ValueError(f"{name} native runtime flag missing")
    if [row.get("task_id") for row in report.get("tasks", [])] != TASKS:
        raise ValueError(f"{name} task rows mismatch")


def _arm(report: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "report": _identity(path),
        "checkpoint": report["checkpoint"],
        "summary": report["summary"],
        "task_results": [
            {
                "task_id": row["task_id"],
                "action_replayed": row["action_replayed"],
                "native_api_calls": row["native_api_calls"],
                "success": row["evaluation"]["success"],
                "passed": row["evaluation"]["passed"],
                "failed": row["evaluation"]["failed"],
                "apis": [step.get("api") for step in row["steps"] if step.get("api")],
                "stop_reasons": [step.get("stop_reason") for step in row["steps"] if step.get("stop_reason")],
            }
            for row in report["tasks"]
        ],
    }


def assemble(m679_warm: Path, m679_random: Path, m692_warm: Path, m692_random: Path, out: Path) -> dict[str, Any]:
    paths = {"m679_warm": m679_warm, "m679_random": m679_random, "m692_warm": m692_warm, "m692_random": m692_random}
    checkpoints = {"m679_warm": PARENT_SHA, "m679_random": "2722ea455de75fb1f99d29eb40ea88dedc59248e11ef1672c22582e7a79fa946", "m692_warm": M692_WARM_SHA, "m692_random": M692_RANDOM_SHA}
    reports = {name: _load(path) for name, path in paths.items()}
    for name, report in reports.items():
        _validate(report, name=name, checkpoint_sha=checkpoints[name])
    arms = {name: _arm(reports[name], paths[name]) for name in paths}
    sequences = {
        name: [tuple(item["apis"]) for item in arm["task_results"]]
        for name, arm in arms.items()
    }
    payload: dict[str, Any] = {
        "kind": "localagent_m694_m679_appworld_free_running",
        "schema_version": 1,
        "benchmark_id": "appworld_native_public_dev_free_running_probe",
        "source": {
            "dataset": "AppWorld",
            "url": "https://github.com/StonyBrookNLP/appworld",
            "data_version": "0.2.0",
            "tasks": TASKS,
            "split": "caller-selected public dev tasks",
        },
        "protocol": {
            "native_runtime": "resettable AppWorld environment with ground-truth verifier",
            "max_steps": 4,
            "retrieve_k": 100,
            "schema_adapter": "strict one literal API call per step",
            "observations": "redacted response type/keys summaries",
            "allow_completion": True,
            "environment_reset_per_task": True,
            "screenshots": False,
            "external_accounts": False,
            "state_side_effects": "isolated AppWorld task databases only",
        },
        "environment_contract": {
            "receipt": "docs/paper/results/raw/m693-m679-appworld-native-v1.json",
            "receipt_sha256": "4303b5cbbf50f22b869522c7aaa867d567a08ff078ba56a9ef0e9e1aa8039b40",
            "oracle_contract_passed": True,
        },
        "parent_checkpoint": {"sha256": PARENT_SHA},
        "arms": arms,
        "comparison": {
            "native_task_success": {name: arm["summary"]["native_successes"] for name, arm in arms.items()},
            "native_task_success_rate": {name: arm["summary"]["native_success_rate"] for name, arm in arms.items()},
            "action_replayed": {name: arm["summary"]["action_replayed"] for name, arm in arms.items()},
            "steps_attempted": {name: arm["summary"]["steps_attempted"] for name, arm in arms.items()},
            "m692_warm_random_action_sequences_identical": sequences["m692_warm"] == sequences["m692_random"],
            "m679_warm_random_action_sequences_identical": sequences["m679_warm"] == sequences["m679_random"],
            "all_arms_zero_task_completion": all(arm["summary"]["native_successes"] == 0 for arm in arms.values()),
        },
        "weight_adoption": {
            "m692_warm_vs_random_free_run_gain_pp": 100.0 * (arms["m692_warm"]["summary"]["native_success_rate"] - arms["m692_random"]["summary"]["native_success_rate"]),
            "decision": "do not promote m692 AppWorld child; retain m679 backbone only as an initialization candidate and require multi-step native success before deployment",
        },
        "claim_boundary": (
            "Native resettable AppWorld free-running trajectory probe over six caller-selected public dev tasks. "
            "The model receives only redacted tool-result summaries and may emit at most four translated API steps. "
            "All four arms score 0/6 task completion; this is not an official AppWorld/AppWorld-UL score, does not "
            "cover Gmail or Notion, and uses no external accounts, screenshots, or irreversible side effects."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m679-warm", type=Path, required=True)
    parser.add_argument("--m679-random", type=Path, required=True)
    parser.add_argument("--m692-warm", type=Path, required=True)
    parser.add_argument("--m692-random", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.m679_warm, args.m679_random, args.m692_warm, args.m692_random, args.out)["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
