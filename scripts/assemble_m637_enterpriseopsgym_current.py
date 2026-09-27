#!/usr/bin/env python3
"""Bind the current checkpoint to the public EnterpriseOps-Gym retrieval diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CURRENT_CHECKPOINT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
DATASET_REVISION = "c8e538eae8a6205294f0a86675fefdc1fac408f6"
DATASET_URL = "https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym"
CODE_URL = "https://github.com/ServiceNow/EnterpriseOps-Gym"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assemble(raw_path: Path, output: Path) -> dict[str, Any]:
    raw = _load(raw_path)
    if raw.get("kind") != "localagent_enterpriseopsgym_tool_retrieval_receipt":
        raise ValueError("unexpected EnterpriseOps-Gym receipt kind")
    if raw.get("dataset") != "ServiceNow-AI/EnterpriseOps-Gym":
        raise ValueError("unexpected EnterpriseOps-Gym dataset")
    if raw.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("dataset revision mismatch")
    if raw.get("checkpoint", {}).get("sha256") != CURRENT_CHECKPOINT_SHA256:
        raise ValueError("receipt is not bound to current m626 checkpoint")
    if raw.get("summary", {}).get("records") != 67:
        raise ValueError("expected pinned email slice with 67 records")
    payload: dict[str, Any] = {
        "kind": "localagent_m637_enterpriseopsgym_current_email_retrieval_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": raw["dataset"],
            "dataset_url": DATASET_URL,
            "dataset_revision": DATASET_REVISION,
            "code_url": CODE_URL,
            "slice": "oracle/email vs plus_15_tools/email",
        },
        "checkpoint": raw["checkpoint"],
        "protocol": raw["protocol"],
        "source_files": raw["source_files"],
        "summary": raw["summary"],
        "records": raw["records"],
        "raw_report": _file_identity(raw_path),
        "claim_boundary": (
            "Current m626 checkpoint on the public EnterpriseOps-Gym email tool-retrieval slice. "
            "The adapter drops server configuration and verifiers and uses generated name-only "
            "tool descriptions with no execution. This is not official EnterpriseOps-Gym "
            "stateful task success, leaderboard performance, or a training artifact."
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
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.raw, args.out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
