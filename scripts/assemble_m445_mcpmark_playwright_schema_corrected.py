#!/usr/bin/env python3
"""Assemble a fail-closed MCPMark Playwright receipt after the MCP schema bridge repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CURRENT_CHILD_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
MCPMARK_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"


def digest(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def native_summary(report: dict, *, label: str) -> dict:
    env = report["environment"]
    results = report["results"]
    return {
        "label": label,
        "model_sha256": report["model"]["sha256"],
        "mcp_server_executed": env["mcp_server_executed"],
        "server": env["server"],
        "headless": env["headless"],
        "official_split_verified": env["official_split_verified"],
        "user_simulator_executed": env["user_simulator_executed"],
        "external_api_called": env["external_api_called"],
        "task_count": len(results),
        "verifier_passes": sum(r["verifier_exit_code"] == 0 for r in results),
        "verifier_failures": sum(r["verifier_exit_code"] != 0 for r in results),
        "runtime_errors": sum(bool(r["server_error"]) for r in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load(args.base)
    selector = load(args.selector)
    transfer = load(args.transfer)
    catalog = load(args.catalog)
    payload = {
        "kind": "localagent_mcpmark_playwright_schema_corrected_abi_guard",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "dataset": {
            "name": "MCPMark",
            "url": "https://github.com/eval-sys/mcpmark",
            "revision": MCPMARK_REVISION,
            "suite": "standard",
            "service": "playwright",
            "tasks_requested": 4,
        },
        "checkpoint_sha256": CURRENT_CHILD_SHA256,
        "environment_executed": True,
        "official_split_verified": False,
        "task_count": 4,
        "success_rate": 0.0,
        "checkpoint": {
            "path": "runs/sft-mind2web-public-continuation-20260805/latest.pt",
            "sha256": CURRENT_CHILD_SHA256,
            "parameters": 10524544,
        },
        "schema_bridge": {
            "mcp_python_sdk_key": "input_schema",
            "accepted_fallback_key": "inputSchema",
            "tool_catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
            "tool_count": len(catalog),
            "m440_superseded": True,
            "m440_failure_reason": "runner read inputSchema only; MCP Python SDK exposed input_schema",
        },
        "native_playwright": native_summary(base, label="current_child_base"),
        "native_selector_warm": native_summary(selector, label="selector_only_warm_child"),
        "abi_guard": {
            "implementation": "src/localagent/agent/constrained.py",
            "behavior": [
                "explicit URL task step -> browser_navigate with extracted URL",
                "after navigation -> browser_snapshot with empty arguments",
                "required ref/function without live grounding -> no candidate (abstain)",
            ],
            "not_learned_performance": True,
            "canary": digest(args.canary) if args.canary else None,
        },
        "trajectory_transfer": transfer["trajectory_transfer"],
        "claim_boundary": (
            "The corrected reports execute four pinned MCPMark Playwright standard fixtures "
            "against the current child and a selector-only warm child with a real @playwright/mcp "
            "stdio server and independent verifiers. Both are native service diagnostics, not "
            "official MCPMark scores: the user simulator, official/verified aggregation, and "
            "other services were not executed. The selector intervention is not promoted because "
            "native success remains 0/4. m440 is superseded and excluded from capability claims "
            "because its schema bridge omitted input_schema."
        ),
        "source_artifacts": {
            "base_report": digest(args.base),
            "selector_report": digest(args.selector),
            "transfer_receipt": digest(args.transfer),
            "tool_catalog": digest(args.catalog),
        },
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
