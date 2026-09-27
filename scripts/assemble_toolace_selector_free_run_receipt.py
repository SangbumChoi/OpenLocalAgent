#!/usr/bin/env python3
"""Join a ToolACE selector transfer report with before/after free-run receipts.

The resulting receipt keeps the public train/eval identities, selector movement, and the
WebGPU-shaped free-run comparison together. It never executes a tool or claims an official
ToolACE/BFCL result.
"""

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _same_file_identity(left: Any, right: Any) -> bool:
    """Compare content identity while allowing equivalent temporary-path copies."""

    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return left.get("bytes") == right.get("bytes") and left.get("sha256") == right.get("sha256")


def assemble(
    *,
    selector_report_path: Path,
    before_path: Path,
    after_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    selector = _load(selector_report_path)
    before = _load(before_path)
    after = _load(after_path)
    source = selector.get("source", {})
    if source.get("dataset") != DATASET or source.get("revision") != REVISION:
        raise ValueError("selector report has unexpected ToolACE identity")
    for label, report in (("before", before), ("after", after)):
        report_source = report.get("source", {})
        if report.get("kind") != "localagent_toolace_action_history_free_run_probe":
            raise ValueError(f"{label} receipt is not a ToolACE free-run probe")
        if report_source.get("dataset") != DATASET or report_source.get("revision") != REVISION:
            raise ValueError(f"{label} receipt has unexpected ToolACE identity")
        if report_source.get("training_used") is not False:
            raise ValueError(f"{label} receipt must declare training_used=false")
        if not _same_file_identity(report_source.get("input"), source.get("eval")):
            raise ValueError(f"{label} receipt does not use the selector eval input")
    parent = selector.get("parent", {})
    child = selector.get("child", {})
    if before.get("checkpoint", {}).get("sha256") != parent.get("sha256"):
        raise ValueError("before receipt is not bound to the selector parent")
    if after.get("checkpoint", {}).get("sha256") != child.get("sha256"):
        raise ValueError("after receipt is not bound to the selector child")
    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    selector_arm = selector["arms"]["retrained_pretrained_backbone"]
    inherited_arm = selector["arms"]["inherited_pretrained_selector"]
    after_argument = float(after_metrics["argument_exact_rate"])
    after_episode = float(after_metrics["episode_exact_rate"])
    body: dict[str, Any] = {
        "kind": "localagent_toolace_action_history_selector_free_run_transfer_receipt",
        "schema_version": 1,
        "dataset": {
            "dataset": DATASET,
            "url": URL,
            "revision": REVISION,
            "train": source["train"],
            "eval": source["eval"],
        },
        "parent": parent,
        "child": child,
        "selector_training": {
            "report": _identity(selector_report_path),
            "hyperparameters": selector.get("hyperparameters"),
            "train_rows": source.get("train_rows"),
            "eval_rows": source.get("eval_rows"),
            "train_actions": source.get("train_actions"),
            "eval_actions": source.get("eval_actions"),
            "inherited_metrics": inherited_arm["metrics"],
            "transferred_metrics": selector_arm["metrics"],
            "random_metrics": selector["arms"]["retrained_matched_random_backbone"]["metrics"],
            "selector_relative_movement": selector_arm["selector_relative_movement"],
        },
        "free_run": {
            "before": {
                "receipt": _identity(before_path),
                "checkpoint": before["checkpoint"],
                "metrics": before_metrics,
            },
            "after": {
                "receipt": _identity(after_path),
                "checkpoint": after["checkpoint"],
                "metrics": after_metrics,
            },
        },
        "comparison": {
            "tool_exact_delta_pp": (after_metrics["tool_exact_rate"] - before_metrics["tool_exact_rate"])
            * 100.0,
            "argument_exact_delta_pp": (after_metrics["argument_exact_rate"] - before_metrics["argument_exact_rate"])
            * 100.0,
            "schema_valid_delta_pp": (after_metrics["schema_valid_rate"] - before_metrics["schema_valid_rate"])
            * 100.0,
            "step_exact_delta_pp": (after_metrics["step_exact_rate"] - before_metrics["step_exact_rate"])
            * 100.0,
            "episode_exact_delta_pp": (after_metrics["episode_exact_rate"] - before_metrics["episode_exact_rate"])
            * 100.0,
        },
        "decision": {
            "selector_top1_improves": selector_arm["metrics"]["selector_top1"]
            > inherited_arm["metrics"]["selector_top1"],
            "free_run_tool_exact_improves": after_metrics["tool_exact_rate"]
            > before_metrics["tool_exact_rate"],
            "adoption": "reject_full_policy_promotion",
            "reason": (
                "Selector ranking improves, but free-run argument exactness is "
                f"{after_argument * 100.0:.2f}% and episode exactness is "
                f"{after_episode * 100.0:.2f}%; keep the selector as a candidate adapter and "
                "require larger source-disjoint and native verifier-backed controls before "
                "deployment adoption."
            ),
        },
        "claim_boundary": (
            "Source-disjoint public ToolACE action-history selector transfer with a matched random "
            "selector control and a bounded WebGPU-shaped free-run comparison. No ToolACE/BFCL "
            "official score, native browser/MCP execution, email or Notion side effect, screenshot "
            "grounding, or external account access is implied."
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
    parser.add_argument("--selector-report", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = assemble(
        selector_report_path=args.selector_report,
        before_path=args.before,
        after_path=args.after,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
