#!/usr/bin/env python3
"""Seal a checkpoint-matched MCPMark Playwright verifier replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assemble(*, warm: Path, parent: Path, output: Path) -> dict[str, Any]:
    warm_payload = _load(warm)
    parent_payload = _load(parent)
    for label, payload in (("warm", warm_payload), ("parent", parent_payload)):
        if payload.get("kind") != "localagent_mcpmark_native_playwright_current_checkpoint":
            raise ValueError(f"{label} receipt kind mismatch")
        if payload.get("benchmark_id") != "mcpmark":
            raise ValueError(f"{label} benchmark mismatch")
        if payload.get("dataset", {}).get("revision") != "cd45b7f57923b9b3985467f5139927575f83141c":
            raise ValueError(f"{label} source revision mismatch")
    warm_result = warm_payload["results"][0]
    parent_result = parent_payload["results"][0]
    if warm_result["task"] != parent_result["task"]:
        raise ValueError("warm and parent task mismatch")
    payload: dict[str, Any] = {
        "kind": "localagent_mcpmark_native_playwright_replay_receipt",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "source": {
            "dataset": "MCPMark",
            "url": "https://github.com/eval-sys/mcpmark",
            "revision": warm_payload["dataset"]["revision"],
            "suite": warm_payload["dataset"]["suite"],
            "service": warm_payload["dataset"]["service"],
            "task": warm_result["task"],
            "official_split_verified": False,
        },
        "runtime": {
            "server": warm_payload["environment"].get("server"),
            "mcp_server_executed": warm_payload["environment"].get("mcp_server_executed"),
            "verifiers_executed": warm_payload["task_source"].get("verifiers_executed"),
            "external_api_called": warm_payload["environment"].get("external_api_called"),
            "headless": warm_payload["environment"].get("headless"),
        },
        "parent": {
            "checkpoint": parent_payload["model"],
            "receipt": _identity(parent),
            "summary": parent_payload["summary"],
            "task_result": parent_result,
        },
        "warm_child": {
            "checkpoint": warm_payload["model"],
            "receipt": _identity(warm),
            "summary": warm_payload["summary"],
            "task_result": warm_result,
        },
        "comparison": {
            "warm_tool_calls": len(warm_result.get("turns", [])),
            "parent_tool_calls": len(parent_result.get("turns", [])),
            "warm_verifier_pass": warm_result.get("verifier_exit_code") == 0,
            "parent_verifier_pass": parent_result.get("verifier_exit_code") == 0,
            "warm_runtime_error": warm_result.get("server_error") is not None,
            "parent_runtime_error": parent_result.get("server_error") is not None,
        },
        "claim_boundary": (
            "One-task native MCPMark Playwright replay with a real stdio MCP server and independent "
            "task verifier. The official split is not verified, the warm child did not pass the CSV "
            "verifier, the parent timed out before tool discovery, and no external account side effect "
            "is implied."
        ),
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = assemble(warm=args.warm, parent=args.parent, output=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
