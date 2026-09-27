#!/usr/bin/env python3
"""Assemble a warm-versus-random EnterpriseOps-Gym retrieval ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


WARM_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
RANDOM_SHA256 = "390f1414260e118cd621af735fe6e87b01e8641b1cff650d594585e39b212e45"
DATASET_REVISION = "c8e538eae8a6205294f0a86675fefdc1fac408f6"
DATASET_URL = "https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assemble(warm_path: Path, random_path: Path, output: Path) -> dict[str, Any]:
    warm = _load(warm_path)
    random = _load(random_path)
    for label, report, expected in (("warm", warm, WARM_SHA256), ("random", random, RANDOM_SHA256)):
        if report.get("kind") != "localagent_enterpriseopsgym_tool_retrieval_receipt":
            raise ValueError(f"unexpected {label} report kind")
        if report.get("dataset_revision") != DATASET_REVISION:
            raise ValueError(f"{label} dataset revision mismatch")
        if report.get("checkpoint", {}).get("sha256") != expected:
            raise ValueError(f"{label} report checkpoint mismatch")
        if report.get("summary", {}).get("records") != 67:
            raise ValueError(f"{label} report must contain 67 records")
    if warm.get("source_files") != random.get("source_files"):
        raise ValueError("warm and random reports do not use identical source files")
    warm_summary = warm["summary"]
    random_summary = random["summary"]
    delta = {
        metric: warm_summary[metric] - random_summary[metric]
        for metric in ("hit_at_1", "hit_at_3", "hit_at_5")
    }
    payload: dict[str, Any] = {
        "kind": "localagent_m638_enterpriseopsgym_warm_random_weight_transfer_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "ServiceNow-AI/EnterpriseOps-Gym",
            "dataset_url": DATASET_URL,
            "dataset_revision": DATASET_REVISION,
            "slice": "oracle/email vs plus_15_tools/email",
            "records": 67,
        },
        "warm_checkpoint": {"identity": warm["checkpoint"], "summary": warm_summary},
        "random_checkpoint": {"identity": random["checkpoint"], "summary": random_summary},
        "warm_minus_random_delta": delta,
        "protocol": {
            "adapter": warm["protocol"]["tool_description_policy"],
            "execution": warm["protocol"]["execution"],
            "verifiers_dropped": True,
            "server_configuration_dropped": True,
            "same_source_files": True,
            "training_on_eval_rows": False,
        },
        "source_files": warm["source_files"],
        "warm_raw_report": _identity(warm_path),
        "random_raw_report": _identity(random_path),
        "claim_boundary": (
            "Matched warm-versus-random backbone ablation on the public EnterpriseOps-Gym email "
            "retrieval projection. The warm advantage is evidence for retaining the pretrained "
            "backbone, but the adapter performs no tool execution and drops verifiers, servers, "
            "credentials, and side effects; this is not official stateful benchmark success."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.warm, args.random, args.out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
