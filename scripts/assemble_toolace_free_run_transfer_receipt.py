#!/usr/bin/env python3
"""Join parent/child ToolACE free-run probes into a transfer decision receipt.

The probes use the same public action-history rows and catalog-constrained decoder.  This
assembler does not execute tools, contact an external service, or claim an official ToolACE/BFCL
score; it only binds the checkpoint identities and computes the deployment-shaped deltas.
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _same_identity(left: Any, right: Any) -> bool:
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256")
    )


def assemble(*, before_path: Path, after_path: Path, output_path: Path) -> dict[str, Any]:
    before = _load(before_path)
    after = _load(after_path)
    for label, report in (("before", before), ("after", after)):
        if report.get("kind") != "localagent_toolace_action_history_free_run_probe":
            raise ValueError(f"{label} report is not a ToolACE free-run probe")
        source = report.get("source", {})
        if source.get("dataset") != DATASET or source.get("revision") != REVISION:
            raise ValueError(f"{label} report has unexpected ToolACE identity")
        if source.get("training_used") is not False:
            raise ValueError(f"{label} report must declare training_used=false")
    before_source = before["source"]
    after_source = after["source"]
    if not _same_identity(before_source.get("input"), after_source.get("input")):
        raise ValueError("parent and child probes must use the same input rows")
    if before.get("rows_evaluated") != after.get("rows_evaluated"):
        raise ValueError("parent and child probes must evaluate the same row count")

    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    delta_keys = (
        "tool_exact_rate",
        "argument_exact_rate",
        "schema_valid_rate",
        "step_exact_rate",
        "episode_exact_rate",
    )
    deltas = {
        f"{key}_delta_pp": (float(after_metrics[key]) - float(before_metrics[key])) * 100.0
        for key in delta_keys
    }
    body: dict[str, Any] = {
        "kind": "localagent_toolace_free_run_parent_child_transfer_receipt",
        "schema_version": 1,
        "dataset": {
            "dataset": DATASET,
            "url": URL,
            "revision": REVISION,
            "input": before_source["input"],
            "rows_evaluated": before["rows_evaluated"],
        },
        "parent": {
            "checkpoint": before["checkpoint"],
            "receipt": _identity(before_path),
            "metrics": before_metrics,
        },
        "child": {
            "checkpoint": after["checkpoint"],
            "receipt": _identity(after_path),
            "metrics": after_metrics,
        },
        "comparison": deltas,
        "decision": {
            "free_run_tool_exact_improves": deltas["tool_exact_rate_delta_pp"] > 0.0,
            "free_run_argument_exact_improves": deltas["argument_exact_rate_delta_pp"] > 0.0,
            "free_run_step_exact_improves": deltas["step_exact_rate_delta_pp"] > 0.0,
            "free_run_episode_exact_improves": deltas["episode_exact_rate_delta_pp"] > 0.0,
            "adoption": "reject_full_policy_promotion",
            "reason": (
                "The continuation child is not promoted from this free-run control: tool, argument, "
                "step, and episode exactness remain deployment-critical, and the probe is not an "
                "official benchmark or native side-effect evaluation."
            ),
        },
        "claim_boundary": (
            "Matched parent/child public ToolACE action-history free-run probe with catalog-constrained "
            "generation. No ToolACE/BFCL official score, tool dispatch, native browser/MCP execution, "
            "email or Notion side effect, screenshot grounding, or external account access is implied."
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
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            assemble(before_path=args.before, after_path=args.after, output_path=args.output),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
