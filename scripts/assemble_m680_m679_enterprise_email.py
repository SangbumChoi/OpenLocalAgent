#!/usr/bin/env python3
"""Seal matched m679 warm/random EnterpriseOps-Gym email retrieval evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET = "ServiceNow-AI/EnterpriseOps-Gym"
REVISION = "c8e538eae8a6205294f0a86675fefdc1fac408f6"
WARM_CHECKPOINT = "dbf45ea710e1b88cbc6631813b2abc7cfd6b454ee0052b0d0d4881c85d932533"
RANDOM_CHECKPOINT = "2722ea455de75fb1f99d29eb40ea88dedc59248e11ef1672c22582e7a79fa946"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected regular file: {path}")
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate(report: dict[str, Any], *, name: str, checkpoint_sha: str) -> None:
    if report.get("kind") != "localagent_enterpriseopsgym_tool_retrieval_receipt":
        raise ValueError(f"{name} report kind mismatch")
    if report.get("dataset") != DATASET or report.get("dataset_revision") != REVISION:
        raise ValueError(f"{name} dataset or revision mismatch")
    if report.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError(f"{name} checkpoint mismatch")
    if report.get("summary", {}).get("records") != 67:
        raise ValueError(f"{name} expected 67 email records")
    protocol = report.get("protocol", {})
    if protocol.get("candidate_mode") != "plus_15_tools":
        raise ValueError(f"{name} candidate protocol mismatch")
    if protocol.get("execution") != "frozen_localagent_dense_selector_no_tool_execution":
        raise ValueError(f"{name} execution protocol mismatch")
    if protocol.get("verifiers_dropped") is not True:
        raise ValueError(f"{name} must record dropped verifiers")


def _arm(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": report["checkpoint"],
        "summary": report["summary"],
        "protocol": report["protocol"],
        "source_files": report["source_files"],
    }


def assemble(warm_path: Path, random_path: Path, output: Path) -> dict[str, Any]:
    warm_report = _load(warm_path)
    random_report = _load(random_path)
    _validate(warm_report, name="warm", checkpoint_sha=WARM_CHECKPOINT)
    _validate(random_report, name="random", checkpoint_sha=RANDOM_CHECKPOINT)
    warm_summary = warm_report["summary"]
    random_summary = random_report["summary"]
    warm = _arm(warm_report)
    random = _arm(random_report)
    comparison: dict[str, Any] = {}
    for k in ("hit_at_1", "hit_at_3", "hit_at_5"):
        warm_value = float(warm_summary[k])
        random_value = float(random_summary[k])
        comparison[f"warm_{k}"] = warm_value
        comparison[f"random_{k}"] = random_value
        comparison[f"warm_minus_random_{k}_pp"] = (warm_value - random_value) * 100.0
    comparison["records"] = 67
    comparison["warm_start_better_at_1"] = comparison["warm_hit_at_1"] > comparison["random_hit_at_1"]
    comparison["warm_start_better_at_3"] = comparison["warm_hit_at_3"] > comparison["random_hit_at_3"]
    comparison["warm_start_better_at_5"] = comparison["warm_hit_at_5"] > comparison["random_hit_at_5"]
    payload: dict[str, Any] = {
        "kind": "localagent_m680_m679_enterpriseopsgym_email_control",
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "url": "https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym",
            "original_code": "https://github.com/ServiceNow/EnterpriseOps-Gym",
            "revision": REVISION,
            "slice": "oracle/email vs plus_15_tools/email",
        },
        "protocol": {
            "records": 67,
            "candidate_tools": 20,
            "oracle_tools_mean": warm_summary["mean_oracle_tool_count"],
            "execution": "frozen_localagent_dense_selector_no_tool_execution",
            "official_native_score": False,
            "warm_lineage": "m679 AndroidControl+MCPMark child",
            "random_lineage": "matched m679 random child control",
        },
        "arms": {"warm": warm, "random": random},
        "comparison": comparison,
        "weight_adoption": {
            "compatible_parameter_count": warm["checkpoint"]["parameters"] == random["checkpoint"]["parameters"],
            "compatible_tokenizer": warm["checkpoint"]["tokenizer_sha256"] == random["checkpoint"]["tokenizer_sha256"],
            "reuse_warm_backbone_for_email": False,
            "recommendation": "do not promote m679 warm transfer for this email retrieval slice; investigate task-family interference",
        },
        "raw_reports": {"warm": _identity(warm_path), "random": _identity(random_path)},
        "claim_boundary": (
            "Public EnterpriseOps-Gym email tool-retrieval diagnostic with generated name-only tool "
            "descriptions. Server configuration and verifiers are dropped and no tool executes. "
            "This is not official EnterpriseOps-Gym stateful task success, leaderboard performance, "
            "or evidence of real email side effects."
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
    payload = assemble(args.warm_report, args.random_report, args.out)
    print(json.dumps(payload["comparison"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
