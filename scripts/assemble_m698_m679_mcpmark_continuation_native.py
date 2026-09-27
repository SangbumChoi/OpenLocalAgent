#!/usr/bin/env python3
"""Seal MCPMark public-trajectory training plus native filesystem replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

MCP_REVISION = "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
MCPMARK_REVISION = "cd45b7f57923b9b3985467f5139927575f83141c"


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


def assemble(*, report: Path, native: Path, parent: Path, child: Path, output: Path) -> dict[str, Any]:
    train = json.loads(report.read_text())
    native_run = json.loads(native.read_text())
    parent_id = _identity(parent)
    child_id = _identity(child)
    if train.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("continuation report kind mismatch")
    if train.get("parent", {}).get("sha256") != parent_id["sha256"]:
        raise ValueError("continuation parent mismatch")
    if train.get("child", {}).get("sha256") != child_id["sha256"]:
        raise ValueError("continuation child mismatch")
    if train.get("hyperparameters", {}).get("steps") != 128:
        raise ValueError("expected 128 optimizer updates")
    if train.get("train_sources", [{}])[0].get("revisions") != [MCP_REVISION]:
        raise ValueError("trajectory source revision mismatch")
    if native_run.get("kind") != "localagent_mcpmark_native_filesystem_standard_current_checkpoint":
        raise ValueError("native receipt kind mismatch")
    if native_run.get("checkpoint", {}).get("sha256") != child_id["sha256"]:
        raise ValueError("native run is not bound to child checkpoint")
    native_summary = native_run.get("summary", {})
    if native_summary.get("tasks") != 30 or native_summary.get("runtime_errors") != 0:
        raise ValueError("native task/runtime summary mismatch")
    payload: dict[str, Any] = {
        "kind": "localagent_m698_m679_mcpmark_continuation_native",
        "schema_version": 1,
        "benchmark_id": "mcpmark",
        "parent": parent_id,
        "child": child_id,
        "training": {
            "dataset": "Jakumetsu/mcpmark-trajectory-log",
            "url": "https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log",
            "revision": MCP_REVISION,
            "original_benchmark": "https://github.com/eval-sys/mcpmark",
            "train_rows": train.get("rows", {}).get("train"),
            "eval_rows": train.get("rows", {}).get("eval"),
            "steps": train["hyperparameters"]["steps"],
            "batch_size": train["hyperparameters"]["batch_size"],
            "learning_rate": train["hyperparameters"]["learning_rate"],
            "max_seq_len": train["hyperparameters"]["max_seq_len"],
            "before_eval": train["before"]["eval"],
            "after_eval": train["after"]["eval"],
            "weight_transfer": train["weight_transfer"],
            "report": _identity(report),
        },
        "native": {
            "task_revision": MCPMARK_REVISION,
            "suite": "filesystem/standard",
            "task_count": native_summary["tasks"],
            "verifier_passes": native_summary["verifier_passes"],
            "verifier_failures": native_summary["verifier_failures"],
            "runtime_errors": native_summary["runtime_errors"],
            "receipt": _identity(native),
            "results": _sanitize(native_run.get("results", [])),
        },
        "comparison": {
            "held_out_token_accuracy_gain_pp": (
                train["after"]["eval"]["assistant_token_accuracy"]
                - train["before"]["eval"]["assistant_token_accuracy"]
            )
            * 100.0,
            "held_out_sequence_accuracy_after": train["after"]["eval"]["assistant_sequence_accuracy"],
            "native_parent_verifier_passes": 0,
            "native_child_verifier_passes": native_summary["verifier_passes"],
            "text_gain_transferred_to_native_state": False,
        },
        "decision": {
            "promotion": "blocked_native_stateful_execution_zero",
            "reuse_warm_backbone": True,
            "claim_boundary": (
                "Public trajectory SFT improves held-out token imitation, but the exact child "
                "still passes zero of 30 independent MCPMark filesystem/standard verifiers. "
                "This is not a complete cross-service MCPMark score and makes no Notion, email, "
                "browser, user-simulator, or external-account claim."
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        report=args.report,
        native=args.native,
        parent=args.parent,
        child=args.child,
        output=args.output,
    )
    print(json.dumps({"output": str(args.output), "receipt_self_sha256": payload["receipt_self_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
