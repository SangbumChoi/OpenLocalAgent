#!/usr/bin/env python3
"""Seal the full native MobileGym result for the AndroidControl-adapted child."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MOBILEGYM_REVISION = "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
ADOPTION_KIND = "localagent_m628_androidcontrol_webgpu_adoption"
NATIVE_KIND = "localagent_mobilegym_native_text_eval"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def assemble(native_path: Path, adoption_path: Path) -> dict[str, Any]:
    native = _load(native_path)
    adoption = _load(adoption_path)
    if native.get("kind") != NATIVE_KIND:
        raise ValueError("unexpected MobileGym receipt kind")
    if native.get("source", {}).get("revision") != MOBILEGYM_REVISION:
        raise ValueError("MobileGym revision mismatch")
    if native.get("official_split_verified") is not True:
        raise ValueError("MobileGym official test split is not verified")
    if native.get("native_receipt_eligible") is not True:
        raise ValueError("native receipt is not eligible")
    if native.get("task_count") != 256 or native.get("official_test_task_count") != 256:
        raise ValueError("expected the complete 256-task test split")
    if native.get("errors"):
        raise ValueError("native MobileGym run contains errors")
    if native.get("vision_used") is not False or native.get("observation_mode") != "text_projection":
        raise ValueError("MobileGym modality boundary changed")
    if adoption.get("kind") != ADOPTION_KIND:
        raise ValueError("unexpected m628 adoption receipt kind")
    checkpoint_sha256 = native.get("checkpoint_sha256")
    if checkpoint_sha256 != adoption.get("checkpoint", {}).get("sha256"):
        raise ValueError("MobileGym and WebGPU adoption checkpoint mismatch")
    payload: dict[str, Any] = {
        "kind": "localagent_m629_androidcontrol_child_mobilegym_native",
        "schema_version": 1,
        "benchmark_id": "mobilegym",
        "checkpoint": adoption["checkpoint"],
        "environment_executed": native["environment_executed"],
        "official_split_verified": native["official_split_verified"],
        "task_count": native["task_count"],
        "success_rate": native["success_rate"],
        "source": {
            **native["source"],
            "native_receipt": _identity(native_path),
            "webgpu_adoption_receipt": _identity(adoption_path),
        },
        "protocol": {
            "official_split": native["official_split"],
            "official_split_verified": native["official_split_verified"],
            "task_count": native["task_count"],
            "max_steps": native["run"]["max_steps"],
            "observation_mode": native["observation_mode"],
            "vision_used": native["vision_used"],
            "environment_executed": native["environment_executed"],
            "errors": len(native["errors"]),
        },
        "result": {
            "passed_tasks": native["passed_tasks"],
            "failed_tasks": native["failed_tasks"],
            "success_rate": native["success_rate"],
            "suite_summary": native["suite_summary"],
            "tool_counts": native["tool_counts"],
            "elapsed_seconds": native["run"]["elapsed_seconds"],
        },
        "diagnosis": {
            "dominant_tool": max(native["tool_counts"], key=native["tool_counts"].get),
            "dominant_tool_count": max(native["tool_counts"].values()),
            "state_grounding_transfer": False,
            "native_mobile_promotion": False,
            "next_training_need": "screenshot_or_accessibility-grounded multi-step mobile state actions",
        },
        "claim_boundary": (
            "Complete pinned MobileGym native simulator/state-diff evaluation over the official 256-task "
            "test split using the text projection. The 1/256 result is an honest native diagnostic, not "
            "visual AndroidControl success, AndroidWorld success, or evidence of real-account side effects."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    receipt = assemble(args.native, args.adoption)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt["result"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
