"""Seal trajectory API-head adaptation and paired native replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assemble(head_report_path: Path, native_head_path: Path, native_baseline_path: Path, output: Path) -> dict[str, Any]:
    head = _load(head_report_path)
    native_head = _load(native_head_path)
    native_baseline = _load(native_baseline_path)
    if head.get("kind") != "localagent_appworld_trajectory_api_head_training_report":
        raise ValueError("trajectory API-head kind mismatch")
    for label, report in (("head", native_head), ("baseline", native_baseline)):
        if report.get("kind") != "localagent_appworld_checkpoint_free_running_trajectory_probe":
            raise ValueError(f"{label} native kind mismatch")
        if report.get("summary", {}).get("tasks") != 6:
            raise ValueError(f"{label} native task count mismatch")
    if native_head["configuration"]["tasks"] != native_baseline["configuration"]["tasks"]:
        raise ValueError("native task order differs")
    payload: dict[str, Any] = {
        "kind": "localagent_m661_appworld_api_head_native_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "AppWorld", "data_version": "0.2.0",
            "source_url": "https://github.com/StonyBrookNLP/appworld",
            "api_head_train_examples": head["trajectory_examples"]["train"],
            "api_head_eval_examples": head["trajectory_examples"]["eval"],
            "native_tasks": 6, "protected_test_used": False,
        },
        "api_head": {
            "report": _identity(head_report_path),
            "classes": head["classes"],
            "metrics": head["metrics"],
        },
        "native_replay": {
            "baseline": {"report": _identity(native_baseline_path), "summary": native_baseline["summary"]},
            "api_head": {"report": _identity(native_head_path), "summary": native_head["summary"]},
            "paired_tasks": native_head["configuration"]["tasks"],
        },
        "decision": {
            "retain_head_as_diagnostic": True,
            "promote_to_native_success": False,
            "reason": (
                "The frozen trajectory API head reaches 42.86% on seen-label public dev prefixes and "
                "raises replayed actions from 5 to 8, but native verifier success remains 0/6; wrong "
                "first-API choices and repeated candidates still dominate."
            ),
        },
        "claim_boundary": (
            "AppWorld compact-trajectory API-head adaptation and paired resettable native replay. This "
            "is not an official leaderboard score, complete AppWorld task success, external-account "
            "side effect, or WebGPU email/Notion promotion."
        ),
        "inputs": {
            "head_report": _identity(head_report_path),
            "native_head": _identity(native_head_path),
            "native_baseline": _identity(native_baseline_path),
        },
    }
    payload["receipt_self_sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--native-head", type=Path, required=True)
    parser.add_argument("--native-baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.head_report, args.native_head, args.native_baseline, args.out)["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
