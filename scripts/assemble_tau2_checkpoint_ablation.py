#!/usr/bin/env python3
"""Assemble hash-only native tau2 checkpoint/mode ablation evidence.

The native runner writes one receipt per checkpoint/mode.  This command joins those receipts with
an independent tensor movement report without retaining task text, tool outputs, or predicted
arguments.  It is an evidence join, not another benchmark scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    if value.get("kind") != "localagent_tau2_mock_checkpoint_native_probe":
        raise ValueError(f"unexpected tau2 receipt kind: {path}")
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported tau2 receipt schema: {path}")
    if value.get("source", {}).get("dataset") != "tau2-bench":
        raise ValueError(f"unexpected tau2 source: {path}")
    return value


def _arm(path: Path, label: str, *, source_revision: str) -> dict[str, Any]:
    receipt = _load(path)
    if receipt["runner"]["source_revision"] != source_revision:
        raise ValueError(f"{label} has a different tau2 source revision")
    source = receipt["source"]
    return {
        "label": label,
        "receipt": _identity(path),
        "receipt_self_sha256": receipt["receipt_self_sha256"],
        "checkpoint": receipt["checkpoint"],
        "configuration": receipt["configuration"],
        "summary": receipt["summary"],
        "contract_verification": receipt["contract_verification"],
        "source": {
            "revision": source["revision"],
            "split": source["split"],
            "domain": source["domain"],
            "task_count": source["task_count"],
            "files": {
                name: item["sha256"] for name, item in sorted(source["files"].items())
            },
        },
    }


def _weight_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "localagent_weight_transfer_analysis":
        raise ValueError(f"unexpected weight report kind: {path}")
    compatibility = value["compatibility"]
    return {
        "report": _identity(path),
        "base": {"sha256": value["base"]["sha256"], "stage": value["base"].get("stage")},
        "target": {"sha256": value["target"]["sha256"], "stage": value["target"].get("stage")},
        "compatibility": {
            "config_mismatches": compatibility["config_mismatches"],
            "shared_tensor_count": compatibility["shared_tensor_count"],
            "shape_mismatches": compatibility["shape_mismatches"],
            "tokenizer_sha256_equal": compatibility["tokenizer_sha256_equal"],
        },
        "groups": value["groups"],
    }


def assemble(*, arms: list[tuple[str, Path]], weight_report: Path, source_revision: str) -> dict[str, Any]:
    if not arms:
        raise ValueError("at least one --arm is required")
    normalized = [_arm(path, label, source_revision=source_revision) for label, path in arms]
    task_counts = {item["source"]["task_count"] for item in normalized}
    source_revisions = {item["source"]["revision"] for item in normalized}
    if task_counts != {10} or source_revisions != {source_revision}:
        raise ValueError("arms must use the same ten-task tau2 mock source")
    result: dict[str, Any] = {
        "kind": "localagent_tau2_mock_checkpoint_ablation",
        "schema_version": 1,
        "source": {
            "dataset": "tau2-bench",
            "domain": "mock",
            "split": "base",
            "revision": source_revision,
            "task_count": 10,
            "task_text_retained": False,
            "tool_outputs_retained": False,
        },
        "arms": normalized,
        "weight_transfer": _weight_summary(weight_report),
        "claim_boundary": (
            "Native resettable tau2 mock diagnostic only. The checkpoint-selector arm, zero-training "
            "schema-retriever arm, and tensor movement report do not constitute a complete tau2 base "
            "split, user-simulator, retail/telecom, leaderboard, or WebGPU task-success result."
        ),
    }
    result["receipt_self_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=RECEIPT")
    parser.add_argument("--weight-report", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms: list[tuple[str, Path]] = []
    for value in args.arm:
        if "=" not in value:
            raise SystemExit("--arm must use LABEL=RECEIPT")
        label, path = value.split("=", 1)
        if not label or not path:
            raise SystemExit("--arm must use LABEL=RECEIPT")
        arms.append((label, Path(path)))
    result = assemble(arms=arms, weight_report=args.weight_report, source_revision=args.source_revision)
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"arms": len(arms), "receipt_self_sha256": result["receipt_self_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
