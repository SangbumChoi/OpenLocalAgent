#!/usr/bin/env python3
"""Seal the current-checkpoint MCPMark filesystem/standard native receipt."""

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
        return re.sub(r"/private/tmp/mcpmark-standard-[^/]+-[^/]+-[^/]+", "<isolated-workspace>", value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    return value


def assemble(*, source: Path, checkpoint: Path) -> dict[str, Any]:
    raw = json.loads(source.read_text())
    if raw.get("kind") != "localagent_mcpmark_native_filesystem_standard_current_checkpoint":
        raise ValueError("source receipt kind mismatch")
    if raw.get("source_revision") != SOURCE_REVISION:
        raise ValueError("MCPMark source revision mismatch")
    expected = _identity(checkpoint)
    if raw.get("checkpoint", {}).get("sha256") != expected["sha256"]:
        raise ValueError("source receipt is not bound to supplied checkpoint")
    results = _sanitize(raw.get("results", []))
    summary = raw.get("summary")
    if not isinstance(summary, dict) or len(results) != int(summary.get("tasks", 0)):
        raise ValueError("invalid source summary/results")
    payload: dict[str, Any] = {
        "kind": "localagent_mcpmark_current_filesystem_standard_receipt",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "environment_executed": True,
        "official_split_verified": True,
        "task_count": len(results),
        "success_rate": float(raw.get("success_rate", 0.0)),
        "checkpoint_sha256": expected["sha256"],
        "source": {
            "dataset": raw.get("dataset"),
            "receipt": _identity(source),
            "source_revision": SOURCE_REVISION,
        },
        "model": expected,
        "environment": raw.get("environment"),
        "results": results,
        "summary": summary,
        "decision": {
            "native_task_passes": int(summary.get("verifier_passes", 0)),
            "native_task_count": len(results),
            "official_split_verified": True,
            "promotion": "blocked_pending_cross_service_mcpmark_and_stateful_success",
            "claim_boundary": (
                "All 30 pinned MCPMark filesystem/standard tasks ran in isolated temporary "
                "workspaces through a real stdio MCP filesystem server and the version-pinned "
                "verifiers. This is an official service subset, not a complete cross-service "
                "MCPMark score; no Notion, GitHub, Postgres, Playwright, user simulator, or "
                "external accounts were executed."
            ),
        },
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")
    payload = assemble(source=args.source, checkpoint=args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "receipt_self_sha256": payload["receipt_self_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
