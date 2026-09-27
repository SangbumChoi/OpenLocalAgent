#!/usr/bin/env python3
"""Assemble a hash-bound public mobile dispatch transfer receipt.

The trainer and native runner deliberately write separate payloads.  This assembler joins them
without copying prompts, screenshots, task text, or model arguments into the tracked result.
"""

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
    return json.loads(path.read_text(encoding="utf-8"))


def _native_summary(payload: dict[str, Any]) -> dict[str, Any]:
    run = payload.get("run") or {}
    claim_boundary = payload.get("claim_boundary")
    if run.get("full_official_test_split") is False:
        claim_boundary = (
            "Bounded native MobileGym simulator canary over selected official test IDs using a "
            "DOM/text observation projection; this is not the full official test split, a visual "
            "mobile-agent score, or an Android emulator result."
        )
    return {
        "receipt": {
            "path": payload.get("_path"),
            "sha256": payload.get("_sha256"),
        },
        "environment_executed": payload.get("environment_executed"),
        "official_split_verified": payload.get("official_split_verified"),
        "official_test_task_count": payload.get("official_test_task_count"),
        "task_count": payload.get("task_count"),
        "passed_tasks": payload.get("passed_tasks"),
        "failed_tasks": payload.get("failed_tasks"),
        "success_rate": payload.get("success_rate"),
        "errors": payload.get("errors"),
        "run": run,
        "observation_mode": payload.get("observation_mode"),
        "vision_used": payload.get("vision_used"),
        "tool_counts": payload.get("tool_counts"),
        "claim_boundary": claim_boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--parent-native", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.output}")

    report = _load(args.report)
    weight = _load(args.weight)
    native = _load(args.native)
    parent_native = _load(args.parent_native)
    native["_path"], native["_sha256"] = str(args.native), _identity(args.native)["sha256"]
    parent_native["_path"], parent_native["_sha256"] = (
        str(args.parent_native),
        _identity(args.parent_native)["sha256"],
    )
    parent_canary = dict(parent_native.get("native_canary", {}))
    parent_canary["_path"], parent_canary["_sha256"] = (
        str(args.parent_native),
        parent_native["_sha256"],
    )
    training = report["mobile_dispatch_training"]
    held_out = training["held_out"]
    receipt: dict[str, Any] = {
        "kind": "localagent_realistic_mobile_dispatch_transfer_receipt",
        "schema_version": 1,
        "source": {
            "dataset": "AndroidControl / Android-in-the-Wild (AITW) public projection",
            "mirror": "OfficerChul/Android-Control-84k",
            "mirror_url": "https://huggingface.co/datasets/OfficerChul/Android-Control-84k",
            "original_androidcontrol_url": (
                "https://github.com/google-research/google-research/tree/master/android_control"
            ),
            "original_aitw_url": (
                "https://github.com/google-research/google-research/tree/master/android_in_the_wild"
            ),
            "license": "Apache-2.0 (reformatted public mirror; verify upstream terms before redistribution)",
            "train_manifest": _identity(args.train_manifest),
            "eval_manifest": _identity(args.eval_manifest),
            "train_jsonl": _identity(args.train),
            "eval_jsonl": _identity(args.eval),
            "visual_input_omitted": True,
            "split_boundary": "public train projection only; the 904-row eval projection stayed outside optimization",
        },
        "training": {
            "report": _identity(args.report),
            "weight_transfer": {
                "report": _identity(args.weight),
                "compatibility": weight.get("compatibility"),
                "groups": weight.get("groups"),
                "recommendation": weight.get("recommendation"),
            },
            "parent": report.get("parent"),
            "child": report.get("child"),
            "steps": training.get("steps"),
            "probe_initialization": training.get("probe_initialization"),
            "focus_tools": training.get("focus_tools"),
            "focus_repeat": training.get("focus_repeat"),
            "train_rows": training.get("rows"),
            "held_out_rows": training.get("external_eval_rows"),
            "train": training.get("train"),
            "held_out": held_out,
            "pointer_training": training.get("pointer_training"),
            "productivity_held_out": training.get("productivity_held_out"),
        },
        "native_canary": _native_summary(native),
        "parent_native_canary": _native_summary(parent_canary),
        "comparison": {
            "selector_top1_after": held_out.get("selector_top1"),
            "route_accuracy_after": held_out.get("route_accuracy"),
            "native_success_rate": native.get("success_rate"),
            "parent_native_same_range_success_rate": (
                parent_native.get("native_canary", {}).get("parent_same_range_success_rate")
            ),
            "native_success_delta_vs_parent_same_range": (
                native.get("success_rate", 0.0)
                - parent_native.get("native_canary", {}).get("parent_same_range_success_rate", 0.0)
            ),
        },
        "decision": "diagnostic_only",
        "claim_boundary": (
            "Public AndroidControl/AITW text-and-accessibility action projection with a disjoint "
            "904-row held-out file plus a 20-task native MobileGym simulator canary. Screenshots "
            "were omitted; this is not Android emulator/AndroidWorld success, the official full "
            "MobileGym score, BrowserGym/OSWorld/MCP success, real email or Notion access, or a "
            "WebGPU hardware throughput claim."
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
