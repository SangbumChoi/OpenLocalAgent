#!/usr/bin/env python3
"""Seal the four-task native MCPMark Playwright standard diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SOURCE_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"/private/tmp/mcpmark-playwright-[^/]+-[^/]+-[^/]+", "<isolated-browser>", value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    return value


def assemble(*, source: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    raw = json.loads(source.read_text())
    expected = _identity(checkpoint)
    if raw.get("kind") != "localagent_mcpmark_native_playwright_standard_current_checkpoint":
        raise ValueError("source kind mismatch")
    if raw.get("source_revision") != SOURCE_REVISION:
        raise ValueError("MCPMark revision mismatch")
    if raw.get("checkpoint", {}).get("sha256") != expected["sha256"]:
        raise ValueError("source is not bound to checkpoint")
    summary = raw.get("summary", {})
    results = _sanitize(raw.get("results", []))
    if summary.get("tasks") != 4 or len(results) != 4 or summary.get("runtime_errors") != 0:
        raise ValueError("expected four bounded tasks and zero runner errors")
    payload: dict[str, Any] = {
        "kind": "localagent_m702_m679_mcpmark_playwright_standard",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "environment_executed": True,
        "official_split_verified": False,
        "task_count": len(results),
        "success_rate": float(raw.get("success_rate", 0.0)),
        "checkpoint_sha256": expected["sha256"],
        "model": expected,
        "dataset": raw.get("dataset"),
        "environment": raw.get("environment"),
        "summary": summary,
        "results": results,
        "source": {"receipt": _identity(source), "source_revision": SOURCE_REVISION},
        "decision": {
            "promotion": "blocked_bounded_subset_and_verifier_zero",
            "browser_action_evidence": {
                "tasks_started_without_runner_error": True,
                "browser_tool_errors": summary.get("browser_tool_errors", 0),
                "navigation_or_snapshot_tasks": 2,
            },
            "claim_boundary": (
                "All four pinned MCPMark Verified Playwright standard tasks ran through the real "
                "MCP browser server with an isolated local Chromium executable. All four verifiers "
                "failed; this is not an official MCPMark score, browser-version parity claim, "
                "visual answer claim, user-simulator result, or external-account result."
            ),
        },
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(source=args.source, checkpoint=args.checkpoint, output=args.output)
    print(json.dumps({"output": str(args.output), "receipt_self_sha256": payload["receipt_self_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
