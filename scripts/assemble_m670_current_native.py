#!/usr/bin/env python3
"""Seal current-checkpoint MobileGym and BrowserGym native receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MOBILE_REVISION = "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
BROWSERGYM_REVISION = "9e779f087de9a65668b6974d11f9ce9816026e96"
MINIWOB_REVISION = "7fd85d71a4b60325c6585396ec4f48377d049838"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _write(payload: dict[str, Any], output: Path) -> dict[str, Any]:
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def assemble_mobile(
    raw_path: Path, checkpoint: Path, output: Path, *, receipt_prefix: str = "m670_m666"
) -> dict[str, Any]:
    raw = _load(raw_path)
    current = _identity(checkpoint)
    if raw.get("kind") != "localagent_mobilegym_native_text_eval":
        raise ValueError("unexpected MobileGym raw kind")
    if raw.get("checkpoint_sha256") != current["sha256"]:
        raise ValueError("MobileGym raw run is not bound to current checkpoint")
    if raw.get("source", {}).get("revision") != MOBILE_REVISION:
        raise ValueError("MobileGym revision mismatch")
    if raw.get("official_split_verified") is not True or raw.get("native_receipt_eligible") is not True:
        raise ValueError("MobileGym official split is not eligible")
    if raw.get("task_count") != 256 or raw.get("official_test_task_count") != 256:
        raise ValueError("MobileGym task count mismatch")
    if raw.get("errors"):
        raise ValueError("MobileGym raw run contains errors")
    return _write(
        {
            "kind": f"localagent_{receipt_prefix}_mobilegym_native_receipt",
            "schema_version": 1,
            "benchmark_id": "mobilegym",
            "checkpoint": current,
            "environment_executed": raw["environment_executed"],
            "official_split_verified": raw["official_split_verified"],
            "task_count": raw["task_count"],
            "success_rate": raw["success_rate"],
            "source": {**raw["source"], "raw_report": _identity(raw_path)},
            "protocol": {
                "official_split": raw["official_split"],
                "observation_mode": raw["observation_mode"],
                "vision_used": raw["vision_used"],
                "run": raw["run"],
            },
            "result": {
                "passed_tasks": raw["passed_tasks"],
                "failed_tasks": raw["failed_tasks"],
                "success_rate": raw["success_rate"],
                "suite_summary": raw["suite_summary"],
                "tool_counts": raw["tool_counts"],
            },
            "decision": {
                "native_mobile_promotion": False,
                "retain_as_current_negative_control": True,
                "next_training_need": "accessibility/screenshot-grounded multi-step mobile state actions",
            },
            "claim_boundary": (
                "Complete pinned MobileGym native simulator/state-diff evaluation over the official "
                "256-task test split using a bounded DOM/text projection. This is not visual Android "
                "control, AndroidWorld success, or real-account side effects."
            ),
        },
        output,
    )


def assemble_browser(
    raw_path: Path, checkpoint: Path, output: Path, *, receipt_prefix: str = "m670_m666"
) -> dict[str, Any]:
    raw = _load(raw_path)
    current = _identity(checkpoint)
    if raw.get("kind") != "localagent_browsergym_native_eval":
        raise ValueError("unexpected BrowserGym raw kind")
    if raw.get("checkpoint", {}).get("sha256") != current["sha256"]:
        raise ValueError("BrowserGym raw run is not bound to current checkpoint")
    if raw.get("browsergym", {}).get("revision") != BROWSERGYM_REVISION:
        raise ValueError("BrowserGym revision mismatch")
    if raw.get("runtime", {}).get("miniwob_revision") != MINIWOB_REVISION:
        raise ValueError("MiniWoB revision mismatch")
    if raw.get("environment_executed") is not True or raw.get("official_split_verified") is not True:
        raise ValueError("BrowserGym official split is not eligible")
    if raw.get("task_count") != 240 or not isinstance(raw.get("cases"), list) or len(raw["cases"]) != 240:
        raise ValueError("BrowserGym task count mismatch")
    payload: dict[str, Any] = {
        "kind": f"localagent_{receipt_prefix}_browsergym_native_receipt",
        "schema_version": 1,
        "benchmark_id": "browsergym_miniwob",
        "checkpoint": current,
        "environment_executed": raw["environment_executed"],
        "official_split_verified": raw["official_split_verified"],
        "task_count": raw["task_count"],
        "success_rate": raw["success_rate"],
        "browsergym": raw["browsergym"],
        "runtime": raw["runtime"],
        "task_plan": raw["task_plan"],
        "result": {
            "passed_tasks": sum(int(case.get("success") is True) for case in raw["cases"]),
            "failed_tasks": sum(int(case.get("success") is not True) for case in raw["cases"]),
            "success_rate": raw["success_rate"],
            "action_errors": sum(
                int(bool(step.get("info_action_error")))
                for case in raw["cases"]
                for step in case.get("steps", [])
                if isinstance(step, dict)
            ),
            "observation_mode": "accessibility_tree_text",
            "vision_used": False,
            "coordinate_fallback": raw.get("coordinate_fallback"),
            "semantic_fallback": raw.get("semantic_fallback"),
        },
        "raw_report": _identity(raw_path),
        "decision": {
            "native_browser_promotion": False,
            "retain_as_current_negative_control": True,
            "next_training_need": "visual/DOM state grounding and multi-step form actions",
        },
        "claim_boundary": (
            "Complete pinned BrowserGym/MiniWoB native evaluation over the official 240-episode "
            "plan using accessibility-tree text. This is not visual computer use, WebArena, or "
            "real-account email/Notion execution."
        ),
    }
    return _write(payload, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobile-raw", type=Path, required=True)
    parser.add_argument("--browser-raw", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mobile-out", type=Path, required=True)
    parser.add_argument("--browser-out", type=Path, required=True)
    parser.add_argument(
        "--receipt-prefix",
        default="m670_m666",
        help="version prefix for the sealed receipt kind (default: m670_m666)",
    )
    args = parser.parse_args()
    mobile = assemble_mobile(
        args.mobile_raw, args.checkpoint, args.mobile_out, receipt_prefix=args.receipt_prefix
    )
    browser = assemble_browser(
        args.browser_raw, args.checkpoint, args.browser_out, receipt_prefix=args.receipt_prefix
    )
    print(json.dumps({"mobile": mobile["result"], "browser": browser["result"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
