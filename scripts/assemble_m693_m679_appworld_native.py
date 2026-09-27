#!/usr/bin/env python3
"""Seal the current native AppWorld execution probe and matched controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASKS = ["6bdbc26_1", "6bdbc26_2", "6bdbc26_3", "396c5a2_1", "396c5a2_2", "396c5a2_3"]
PARENT_SHA = "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"


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


def _validate(report: dict[str, Any], *, name: str) -> None:
    if report.get("kind") != "localagent_appworld_checkpoint_native_probe":
        raise ValueError(f"{name} report kind mismatch")
    if report.get("schema_version") != 1:
        raise ValueError(f"{name} schema mismatch")
    if report.get("configuration", {}).get("tasks") != TASKS:
        raise ValueError(f"{name} task list mismatch")
    if report.get("configuration", {}).get("replay_appworld_api_step") is not True:
        raise ValueError(f"{name} must use AppWorld API-step replay")
    if report.get("configuration", {}).get("schema_ground_appworld_api_step") is not True:
        raise ValueError(f"{name} must use schema grounding")
    if report.get("environment", {}).get("native_runtime_executed") is not True:
        raise ValueError(f"{name} native runtime flag missing")
    if report.get("runner", {}).get("contract_verification", {}).get("passed") != 1:
        raise ValueError(f"{name} verifier contract did not pass")
    if [row.get("task_id") for row in report.get("tasks", [])] != TASKS:
        raise ValueError(f"{name} task rows mismatch")


def _arm(report: dict[str, Any], path: Path) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "report": _identity(path),
        "checkpoint": report["checkpoint"],
        "summary": summary,
        "native_successes": summary["native_successes"],
        "native_success_rate": summary["native_success_rate"],
        "action_replayed": summary["action_replayed"],
        "native_api_calls": summary["native_api_calls"],
        "native_action_api_calls": summary["native_action_api_calls"],
        "native_bootstrap_api_calls": summary["native_bootstrap_api_calls"],
        "predicted_tools": [row.get("predicted_tool") for row in report["tasks"]],
        "task_results": [
            {
                "task_id": row["task_id"],
                "action_replayed": row["action_replayed"],
                "predicted_tool": row.get("predicted_tool"),
                "schema_translation_applied": row["schema_translation_applied"],
                "success": row["evaluation"]["success"],
                "passed": row["evaluation"]["passed"],
                "failed": row["evaluation"]["failed"],
            }
            for row in report["tasks"]
        ],
    }


def assemble(warm: Path, random: Path, adapted_warm: Path, adapted_random: Path, out: Path) -> dict[str, Any]:
    reports = {"m679_warm": _load(warm), "m679_random": _load(random), "m692_warm": _load(adapted_warm), "m692_random": _load(adapted_random)}
    for name, report in reports.items():
        _validate(report, name=name)
    arms = {name: _arm(report, path) for name, (report, path) in {
        "m679_warm": (reports["m679_warm"], warm),
        "m679_random": (reports["m679_random"], random),
        "m692_warm": (reports["m692_warm"], adapted_warm),
        "m692_random": (reports["m692_random"], adapted_random),
    }.items()}
    payload: dict[str, Any] = {
        "kind": "localagent_m693_m679_appworld_native",
        "schema_version": 1,
        "benchmark_id": "appworld_native_public_dev_probe",
        "source": {
            "dataset": "AppWorld",
            "url": "https://github.com/StonyBrookNLP/appworld",
            "data_version": "0.2.0",
            "package_version": reports["m679_warm"]["runner"]["version"],
            "dev_manifest": {
                "path": "/private/tmp/m663-grounding/dev.manifest.json",
                "bytes": 4875,
                "sha256": "86a7e2cc9ea093f0259003fa31d62868c40a5c845411af67c740dc673be05fdf",
                "output_sha256": "36fd92084d8aaac602e193b5b7db26a5fa783b9629a6f06a0087390d37527d07",
            },
            "tasks": TASKS,
            "split": "caller-selected public dev tasks",
        },
        "protocol": {
            "native_runtime": "AppWorld resettable environment with ground-truth verifier",
            "environment_reset_per_task": True,
            "max_interactions": 1,
            "action_translation": "strict AST appworld_api_step with checkpoint-ranked schema candidates",
            "selector_first": True,
            "retrieve_k": 100,
            "screenshots": False,
            "external_accounts": False,
            "state_side_effects": "isolated AppWorld task databases only",
        },
        "runner_contract": {
            "verified": True,
            "oracle_tasks": 1,
            "all_reports_contract_passed": True,
        },
        "parent_checkpoint": {"sha256": PARENT_SHA, "path": reports["m679_warm"]["checkpoint"]["path"], "bytes": reports["m679_warm"]["checkpoint"]["bytes"]},
        "arms": arms,
        "comparison": {
            "native_task_success": {name: arm["native_successes"] for name, arm in arms.items()},
            "native_task_success_rate": {name: arm["native_success_rate"] for name, arm in arms.items()},
            "action_replayed": {name: arm["action_replayed"] for name, arm in arms.items()},
            "m679_warm_vs_random_success_delta_pp": 100.0 * (arms["m679_warm"]["native_success_rate"] - arms["m679_random"]["native_success_rate"]),
            "m692_warm_vs_random_success_delta_pp": 100.0 * (arms["m692_warm"]["native_success_rate"] - arms["m692_random"]["native_success_rate"]),
            "m692_warm_minus_m679_warm_success_delta_pp": 100.0 * (arms["m692_warm"]["native_success_rate"] - arms["m679_warm"]["native_success_rate"]),
        },
        "claim_boundary": (
            "Native AppWorld reset/verifier execution on six caller-selected public dev tasks. "
            "Each model was allowed one translated API action; no task completed (0/6 in every arm). "
            "This is not an official AppWorld/AppWorld-UL score, does not cover Gmail or Notion, "
            "and uses no external accounts or screenshots. The successful oracle contract only proves "
            "that the pinned environment and verifier were live."
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
