#!/usr/bin/env python3
"""Assemble a hash-bound current-parent MCPMark trajectory transfer receipt.

Only public provenance, split identities, aggregate metrics, and tensor movement are retained;
redacted trajectory prompts, tool schemas, and outputs remain outside the tracked receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET = "Jakumetsu/mcpmark-trajectory-log"
SOURCE_URL = "https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log"
SOURCE_REVISION = "e50578f0ab904d8e6a7c576c387c1e76ae482c89"


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _source_input(report: dict[str, Any], key: str) -> Path:
    sources = report[key]
    if len(sources) != 1:
        raise ValueError(f"expected one {key} source, found {len(sources)}")
    path = Path(sources[0]["input"]["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _split_audit(train_path: Path, eval_path: Path) -> dict[str, Any]:
    train, evaluation = _rows(train_path), _rows(eval_path)
    train_ids = {str(row.get("meta", {}).get("parent_record_id")) for row in train}
    eval_ids = {str(row.get("meta", {}).get("parent_record_id")) for row in evaluation}
    overlap = sorted((train_ids & eval_ids) - {"None"})
    if overlap:
        raise ValueError(f"train/eval parent-record overlap: {overlap[:5]}")
    for label, rows in (("train", train), ("eval", evaluation)):
        for row in rows:
            meta = row.get("meta", {})
            if meta.get("source_dataset") != DATASET:
                raise ValueError(f"{label} row has unexpected source dataset")
            if meta.get("source_revision") != SOURCE_REVISION:
                raise ValueError(f"{label} row has unexpected source revision")
            if meta.get("tool_outputs_redacted") is not True or meta.get("assistant_text_redacted") is not True:
                raise ValueError(f"{label} row is not redacted according to its provenance")
    return {
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "train_parent_records": len(train_ids - {"None"}),
        "eval_parent_records": len(eval_ids - {"None"}),
        "parent_record_disjoint": True,
        "train_parent_records_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
        "eval_parent_records_sha256": hashlib.sha256("\n".join(sorted(eval_ids)).encode()).hexdigest(),
        "tool_outputs_redacted": True,
        "assistant_text_redacted": True,
        "visual_input_omitted": True,
    }


def _arm(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report": _identity(report_path),
        "parent": report.get("parent"),
        "child": report.get("child"),
        "hyperparameters": report.get("hyperparameters"),
        "rows": report.get("rows"),
        "before_eval": report.get("before", {}).get("eval"),
        "after_eval": report.get("after", {}).get("eval"),
        "weight_transfer": report.get("weight_transfer"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")

    warm = _load(args.warm_report)
    random_arm = _load(args.random_report)
    train_path = _source_input(warm, "train_sources")
    eval_path = _source_input(warm, "eval_sources")
    split = _split_audit(train_path, eval_path)
    warm_after = warm["after"]["eval"]["assistant_token_accuracy"]
    random_after = random_arm["after"]["eval"]["assistant_token_accuracy"]
    receipt: dict[str, Any] = {
        "kind": "localagent_mcpmark_current_parent_transfer_receipt",
        "schema_version": 1,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "license": "MIT (public trajectory-log source; verify upstream terms before redistribution)",
        "inputs": {"train": _identity(train_path), "eval": _identity(eval_path)},
        "split_audit": split,
        "training": {
            "warm": _arm(args.warm_report, warm),
            "random": _arm(args.random_report, random_arm),
        },
        "comparison": {
            "warm_after_token_accuracy": warm_after,
            "random_after_token_accuracy": random_after,
            "warm_minus_random_after_pp": (warm_after - random_after) * 100.0,
            "warm_after_sequence_accuracy": warm["after"]["eval"]["assistant_sequence_accuracy"],
            "random_after_sequence_accuracy": random_arm["after"]["eval"]["assistant_sequence_accuracy"],
            "warm_start_better_after": warm_after > random_after,
        },
        "decision": "diagnostic_only",
        "claim_boundary": (
            "Public MIT-licensed MCPMark trajectory-log rows with tool outputs and assistant free "
            "text redacted, source-disjoint teacher-forced continuation on the current m142 parent. "
            "This is not an official MCPMark score, live MCP/server/verifier execution, native "
            "browser or desktop success, screenshot grounding, or a real Notion/email side effect."
        ),
    }
    receipt["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
