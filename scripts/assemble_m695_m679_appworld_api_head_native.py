#!/usr/bin/env python3
"""Seal matched AppWorld API-head training and native free-run controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PARENT = "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
TASKS = ["6bdbc26_1", "6bdbc26_2", "6bdbc26_3", "396c5a2_1", "396c5a2_2", "396c5a2_3"]


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _check_head(report: dict[str, Any], name: str, parent: str) -> None:
    if report.get("kind") != "localagent_appworld_api_head_training_report":
        raise ValueError(f"{name} head kind mismatch")
    if report.get("parent", {}).get("sha256") != parent:
        raise ValueError(f"{name} head parent mismatch")
    if report.get("hyperparameters", {}).get("steps") != 1024:
        raise ValueError(f"{name} head step mismatch")
    if report.get("source", {}).get("train_rows") != 90 or report.get("source", {}).get("eval_rows") != 6:
        raise ValueError(f"{name} head split mismatch")


def _check_native(report: dict[str, Any], name: str) -> None:
    if report.get("kind") != "localagent_appworld_checkpoint_free_running_trajectory_probe":
        raise ValueError(f"{name} native kind mismatch")
    if report.get("configuration", {}).get("tasks") != TASKS:
        raise ValueError(f"{name} native task mismatch")
    if report.get("configuration", {}).get("appworld_api_head") in (None, ""):
        raise ValueError(f"{name} native head binding missing")
    if report.get("summary", {}).get("native_successes") != 0:
        raise ValueError(f"{name} unexpectedly completed a task")


def assemble(warm_head: Path, random_head: Path, warm_native: Path, random_native: Path, out: Path) -> dict[str, Any]:
    wh, rh, wn, rn = map(_load, (warm_head, random_head, warm_native, random_native))
    _check_head(wh, "warm", PARENT)
    _check_head(rh, "random", "2722ea455de75fb1f99d29eb40ea88dedc59248e11ef1672c22582e7a79fa946")
    _check_native(wn, "warm")
    _check_native(rn, "random")
    payload: dict[str, Any] = {
        "kind": "localagent_m695_m679_appworld_api_head_native",
        "schema_version": 1,
        "benchmark_id": "appworld_api_head_native_free_run",
        "source": {"dataset": "AppWorld", "url": "https://github.com/StonyBrookNLP/appworld", "data_version": "0.2.0", "train_tasks": 90, "dev_tasks": 6},
        "protocol": {"head_steps": 1024, "head_batch_size": 32, "head_learning_rate": 0.005, "native_max_steps": 4, "native_retrieve_k": 100, "backbone_frozen": True, "native_observations": "redacted response summaries"},
        "arms": {
            "warm": {"head_report": _identity(warm_head), "native_report": _identity(warm_native), "parent": wh["parent"], "head": wh["child"], "head_metrics": wh["metrics"], "native_summary": wn["summary"]},
            "random": {"head_report": _identity(random_head), "native_report": _identity(random_native), "parent": rh["parent"], "head": rh["child"], "head_metrics": rh["metrics"], "native_summary": rn["summary"]},
        },
        "comparison": {
            "head_eval_exact": {"warm": wh["metrics"]["eval"]["exact"], "random": rh["metrics"]["eval"]["exact"]},
            "head_eval_accuracy": {"warm": wh["metrics"]["eval"]["accuracy"], "random": rh["metrics"]["eval"]["accuracy"]},
            "native_successes": {"warm": wn["summary"]["native_successes"], "random": rn["summary"]["native_successes"]},
            "native_success_rate": {"warm": wn["summary"]["native_success_rate"], "random": rn["summary"]["native_success_rate"]},
            "native_action_replayed": {"warm": wn["summary"]["action_replayed"], "random": rn["summary"]["action_replayed"]},
        },
        "weight_adoption": {"backbone_frozen": True, "decision": "do not promote API head; native completion remains 0/6 despite warm held-out routing 2/6"},
        "claim_boundary": "Frozen-backbone public-train AppWorld API-schema head with matched random control and native resettable free-run. Not an official AppWorld score, complete policy, Gmail/Notion result, or external-account evaluation.",
    }
    payload["receipt_self_sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-head", type=Path, required=True)
    parser.add_argument("--random-head", type=Path, required=True)
    parser.add_argument("--warm-native", type=Path, required=True)
    parser.add_argument("--random-native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.warm_head, args.random_head, args.warm_native, args.random_native, args.out)["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
