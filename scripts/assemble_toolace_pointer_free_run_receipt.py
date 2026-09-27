#!/usr/bin/env python3
"""Join a ToolACE pointer transfer report with a before/after free-run receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.conversation_artifact import canonical_json_bytes


DATASET = "Team-ACE/ToolACE"
REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
URL = "https://huggingface.co/datasets/Team-ACE/ToolACE"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _same_identity(left: Any, right: Any) -> bool:
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256")
    )


def assemble(
    *,
    pointer_report_path: Path,
    before_path: Path,
    after_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    pointer = _load(pointer_report_path)
    before = _load(before_path)
    after = _load(after_path)
    source = pointer.get("source", {})
    if pointer.get("kind") != "localagent_toolace_action_history_pointer_transfer_probe":
        raise ValueError("pointer report has an unexpected kind")
    if source.get("dataset") != DATASET or source.get("revision") != REVISION:
        raise ValueError("pointer report has unexpected ToolACE identity")
    for label, report in (("before", before), ("after", after)):
        report_source = report.get("source", {})
        if report.get("kind") != "localagent_toolace_action_history_free_run_probe":
            raise ValueError(f"{label} receipt is not a ToolACE free-run probe")
        if report_source.get("dataset") != DATASET or report_source.get("revision") != REVISION:
            raise ValueError(f"{label} receipt has unexpected ToolACE identity")
        if report_source.get("training_used") is not False:
            raise ValueError(f"{label} receipt must declare training_used=false")
        if not _same_identity(report_source.get("input"), source.get("eval")):
            raise ValueError(f"{label} receipt does not use the pointer eval input")
    if before.get("checkpoint", {}).get("sha256") != pointer.get("parent", {}).get("sha256"):
        raise ValueError("before receipt is not bound to the pointer parent")
    if after.get("checkpoint", {}).get("sha256") != pointer.get("child", {}).get("sha256"):
        raise ValueError("after receipt is not bound to the pointer child")

    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    transferred = pointer["arms"]["retrained_pretrained_backbone"]
    inherited = pointer["arms"]["inherited_pretrained_pointer"]
    argument_delta_pp = (after_metrics["argument_exact_rate"] - before_metrics["argument_exact_rate"]) * 100.0
    body: dict[str, Any] = {
        "kind": "localagent_toolace_action_history_pointer_free_run_transfer_receipt",
        "schema_version": 1,
        "dataset": {
            "dataset": DATASET,
            "url": URL,
            "revision": REVISION,
            "train": source["train"],
            "eval": source["eval"],
        },
        "parent": pointer["parent"],
        "child": pointer["child"],
        "pointer_training": {
            "report": _identity(pointer_report_path),
            "hyperparameters": pointer.get("hyperparameters"),
            "train_rows": source.get("train_rows"),
            "eval_rows": source.get("eval_rows"),
            "train_locatable_spans": source.get("train_locatable_spans"),
            "eval_locatable_spans": source.get("eval_locatable_spans"),
            "inherited_metrics": inherited["metrics"],
            "transferred_metrics": transferred["metrics"],
            "random_metrics": pointer["arms"]["retrained_matched_random_backbone"]["metrics"],
            "pointer_relative_movement": transferred["pointer_relative_movement"],
        },
        "free_run": {
            "before": {"receipt": _identity(before_path), "checkpoint": before["checkpoint"], "metrics": before_metrics},
            "after": {"receipt": _identity(after_path), "checkpoint": after["checkpoint"], "metrics": after_metrics},
        },
        "comparison": {
            "tool_exact_delta_pp": (after_metrics["tool_exact_rate"] - before_metrics["tool_exact_rate"]) * 100.0,
            "argument_exact_delta_pp": (after_metrics["argument_exact_rate"] - before_metrics["argument_exact_rate"]) * 100.0,
            "schema_valid_delta_pp": (after_metrics["schema_valid_rate"] - before_metrics["schema_valid_rate"]) * 100.0,
            "step_exact_delta_pp": (after_metrics["step_exact_rate"] - before_metrics["step_exact_rate"]) * 100.0,
            "episode_exact_delta_pp": (after_metrics["episode_exact_rate"] - before_metrics["episode_exact_rate"]) * 100.0,
        },
        "decision": {
            "pointer_span_improves": transferred["metrics"]["decoded_value_exact"]
            > inherited["metrics"]["decoded_value_exact"],
            "free_run_argument_exact_improves": after_metrics["argument_exact_rate"]
            > before_metrics["argument_exact_rate"],
            "adoption": "reject_pointer_promotion",
            "reason": (
                "Offline locatable-span training improves the pointer diagnostic, but the free-run "
                f"argument exactness delta is {argument_delta_pp:.2f} pp "
                "and episode exactness remains below a deployment threshold; retain the pointer "
                "adapter as a diagnostic only."
            ),
        },
        "claim_boundary": (
            "Source-disjoint public ToolACE pointer/copy transfer with matched random control and a "
            "bounded WebGPU-shaped free-run comparison. No official ToolACE/BFCL score, native "
            "browser/MCP execution, email or Notion side effect, screenshot grounding, or external "
            "account access is implied."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer-report", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = assemble(
        pointer_report_path=args.pointer_report,
        before_path=args.before,
        after_path=args.after,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
