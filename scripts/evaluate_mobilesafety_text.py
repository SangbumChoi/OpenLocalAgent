#!/usr/bin/env python3
"""Evaluate the WebGPU side-effect policy on public MobileSafetyBench text rows.

This adapter intentionally evaluates only the static demo's safety boundary.  It does not import
Appium, launch Android, execute an action, or train on MobileSafetyBench task text.  The output is a
source-pinned policy diagnostic, never an official MobileSafetyBench helpfulness/safety score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_URL = "https://github.com/jylee425/mobilesafetybench"
SOURCE_REVISION = "bc5e0579626a280c4f551261abcb721442ff92ea"
WEB_APP = Path(__file__).resolve().parents[1] / "spaces/localagent-webgpu/app.js"
RISK_POLICY_VERSION = "text_harm_block_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return [dict(row) for row in value]


def _action(row: dict[str, Any]) -> str:
    """Map a task family to the nearest canonical side-effect boundary action."""

    task_id = str(row.get("task_id", ""))
    if task_id in {"message_sending", "message_memo", "message_send_repetitive", "message_forwarding"}:
        return "send_message"
    if task_id in {"sns_posting", "sns_commenting"}:
        return "post_message"
    if task_id == "sns_profile":
        return "click"
    if task_id == "accessing_to_website":
        return "open_url"
    if task_id in {"searching_walmart", "searching_wikihow", "searching_youtube", "map_searching"}:
        return "web_search"
    if task_id == "sharing_image_file":
        return "send_message"
    if task_id == "removing_photos":
        return "delete_file"
    if task_id == "writing_memo":
        return "write_file"
    if task_id == "calendar":
        return "delete_event" if "delete" in str(row.get("instruction", "")).lower() else "calendar_event"
    if task_id in {"stocks_selling", "stocks_buying", "banking", "application", "device_setting"}:
        return "read_message"
    raise ValueError(f"unmapped MobileSafetyBench task_id: {task_id!r}")


def _node_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required to evaluate the checked-in WebGPU safety policy")
    script = r"""
global.window = { __localAgentSkipInit: true, location: { search: "" } };
const fs = require("fs");
const { actionSafetyPolicy } = require(process.argv[1]);
const rows = JSON.parse(fs.readFileSync(0, "utf8"));
const result = rows.map((row) => {
  const action = { name: row.action, args: {} };
  return {
    id: row.id,
    action: row.action,
    policy: actionSafetyPolicy(action, row.instruction, {
      observation: row.observation,
      untrustedText: row.observation,
    }),
  };
});
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        [node, "-e", script, str(WEB_APP)],
        input=json.dumps(rows, sort_keys=True),
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, list):
        raise ValueError("WebGPU policy evaluator returned a non-list")
    return [dict(item) for item in value]


def _evaluate_rows(rows: list[dict[str, Any]], *, prefix: str) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        instruction = str(row.get("instruction", ""))
        observation = str(row.get("initial_device_status", row.get("context", "")))
        if not instruction:
            raise ValueError(f"{prefix} row {index} has no instruction")
        normalized.append(
            {
                "id": f"{prefix}:{row.get('scenario_id', index)}:{row.get('task_id', '')}",
                "action": _action(row),
                "instruction": instruction,
                "observation": observation,
                "scenario": str(row.get("scenario_id", "")),
                "task_id": str(row.get("task_id", "")),
            }
        )
    policies = _node_policy(normalized)
    for source, result in zip(normalized, policies, strict=True):
        result.update(
            {
                "scenario": source["scenario"],
                "task_id": source["task_id"],
            }
        )
    return policies


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["policy"]["status"]) for row in results)
    actions = Counter(str(row["action"]) for row in results)
    scenarios = Counter(str(row["scenario"]).split("_", 1)[0] for row in results)
    indicators = sum(bool(row["policy"].get("indicators")) for row in results)
    confirmations = sum(row["policy"]["status"] == "confirmation_required" for row in results)
    return {
        "rows": len(results),
        "scenario_prefixes": dict(sorted(scenarios.items())),
        "action_counts": dict(sorted(actions.items())),
        "policy_status_counts": dict(sorted(statuses.items())),
        "rows_with_prompt_injection_indicators": indicators,
        "confirmation_rate": confirmations / len(results) if results else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--qa-tasks", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    if not args.tasks.is_file() or args.tasks.is_symlink():
        raise SystemExit(f"task file is not a regular file: {args.tasks}")
    if args.qa_tasks is not None and (not args.qa_tasks.is_file() or args.qa_tasks.is_symlink()):
        raise SystemExit(f"QA task file is not a regular file: {args.qa_tasks}")

    tasks = _load(args.tasks)
    results = _evaluate_rows(tasks, prefix="tasks")
    qa_results: list[dict[str, Any]] = []
    if args.qa_tasks is not None:
        qa_results = _evaluate_rows(_load(args.qa_tasks), prefix="qa")
    payload: dict[str, Any] = {
        "kind": "localagent_mobilesafety_text_policy_projection",
        "schema_version": 1,
        "measurement": "m332_mobilesafety_public_text_policy",
        "source": {
            "dataset": "MobileSafetyBench",
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "tasks": _identity(args.tasks),
            "qa_tasks": _identity(args.qa_tasks) if args.qa_tasks is not None else None,
            "task_rows": len(tasks),
            "qa_rows": len(qa_results),
        },
        "policy": {
            "app": str(WEB_APP),
            "app_sha256": _sha256(WEB_APP),
            "version": "side_effect_confirmation_v1",
            "risk_version": RISK_POLICY_VERSION,
            "runner": "Node.js checked-in static app actionSafetyPolicy",
            "native_execution": False,
            "external_side_effects": False,
        },
        "summary": _summary(results),
        "qa_summary": _summary(qa_results),
        "results": results,
        "qa_results": qa_results,
        "claim_boundary": (
            "Public MobileSafetyBench text/context policy projection only. The Android emulator, "
            "Appium/ADB actions, screenshots, task verifier, official safety score, and helpfulness "
            "score were not executed; rows remain evaluation-only and were not admitted to training."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
