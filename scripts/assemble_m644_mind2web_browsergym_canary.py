#!/usr/bin/env python3
"""Compare m626 and Mind2Web-trained native BrowserGym canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BASELINE_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
MIND2WEB_CHILD_SHA256 = "3b1737e93fbfbdc6c412d8b9385a885098b280494fb07c0e2bdb8839749f0076"
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


def assemble(baseline_path: Path, child_path: Path, output: Path) -> dict[str, Any]:
    baseline = _load(baseline_path)
    child = _load(child_path)
    for label, report, checkpoint in (
        ("baseline", baseline, BASELINE_SHA256),
        ("mind2web_child", child, MIND2WEB_CHILD_SHA256),
    ):
        if report.get("benchmark_id") != "browsergym_miniwob":
            raise ValueError(f"{label} benchmark mismatch")
        if report.get("checkpoint", {}).get("sha256") != checkpoint:
            raise ValueError(f"{label} checkpoint mismatch")
        if report.get("tool_pool") != "realistic_browser":
            raise ValueError("canary tool pool mismatch")
        if report.get("task_count") != 16 or report.get("official_split_verified") is not False:
            raise ValueError("expected bounded non-official 16-episode canary")
        if report.get("browsergym", {}).get("revision") != BROWSERGYM_REVISION:
            raise ValueError("BrowserGym revision mismatch")
        if report.get("runtime", {}).get("miniwob_revision") != MINIWOB_REVISION:
            raise ValueError("MiniWoB revision mismatch")
    baseline_cases = baseline["cases"]
    child_cases = child["cases"]
    if [(row["task"], row["seed"]) for row in baseline_cases] != [(row["task"], row["seed"]) for row in child_cases]:
        raise ValueError("baseline and child episode order mismatch")
    def _stable_steps(case: dict[str, Any]) -> list[dict[str, Any]]:
        return [{key: value for key, value in step.items() if key != "wall_ms"} for step in case["steps"]]

    identical_actions = sum(
        _stable_steps(row) == _stable_steps(child_row)
        for row, child_row in zip(baseline_cases, child_cases, strict=True)
    )
    payload: dict[str, Any] = {
        "kind": "localagent_m644_mind2web_browsergym_native_canary_receipt",
        "schema_version": 1,
        "benchmark": {
            "id": "browsergym_miniwob",
            "browsergym_revision": BROWSERGYM_REVISION,
            "miniwob_revision": MINIWOB_REVISION,
            "episodes": 16,
            "fixed_seed_plan": True,
            "tool_pool": "realistic_browser",
            "official_split_verified": False,
        },
        "baseline": {
            "checkpoint": baseline["checkpoint"],
            "success_count": sum(bool(case["success"]) for case in baseline_cases),
            "success_rate": baseline["success_rate"],
            "report": _identity(baseline_path),
        },
        "mind2web_child": {
            "checkpoint": child["checkpoint"],
            "success_count": sum(bool(case["success"]) for case in child_cases),
            "success_rate": child["success_rate"],
            "report": _identity(child_path),
        },
        "paired_comparison": {
            "same_episode_order": True,
            "identical_step_traces": identical_actions,
            "success_delta": child["success_rate"] - baseline["success_rate"],
            "successful_tasks_baseline": sorted(case["task"] for case in baseline_cases if case["success"]),
            "successful_tasks_child": sorted(case["task"] for case in child_cases if case["success"]),
        },
        "runtime": baseline["runtime"],
        "claim_boundary": (
            "Native pinned BrowserGym/MiniWoB checkpoint-in-the-loop canary using the realistic_browser "
            "vocabulary. It is a bounded non-official diagnostic: Mind2Web backend IDs are not live "
            "MiniWoB IDs, screenshots were not used, and no real website/account side effect ran."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.baseline, args.child, args.out)
    print(json.dumps(report["paired_comparison"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
