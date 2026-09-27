#!/usr/bin/env python3
"""Assemble paired BrowserGym grounding diagnostics for the m626 and m643 checkpoints."""

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


def _stable_steps(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [{key: value for key, value in step.items() if key != "wall_ms"} for step in case["steps"]]


def _validate_report(
    label: str,
    report: dict[str, Any],
    checkpoint_sha: str,
    *,
    semantic_fallback: bool,
    coordinate_fallback: bool,
) -> None:
    if report.get("benchmark_id") != "browsergym_miniwob":
        raise ValueError(f"{label} benchmark mismatch")
    if report.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError(f"{label} checkpoint mismatch")
    if report.get("tool_pool") != "realistic_browser":
        raise ValueError(f"{label} tool pool mismatch")
    if report.get("task_count") != 16 or report.get("official_split_verified") is not False:
        raise ValueError(f"{label} expected bounded non-official 16-episode canary")
    if report.get("browsergym", {}).get("revision") != BROWSERGYM_REVISION:
        raise ValueError(f"{label} BrowserGym revision mismatch")
    if report.get("runtime", {}).get("miniwob_revision") != MINIWOB_REVISION:
        raise ValueError(f"{label} MiniWoB revision mismatch")
    if report.get("semantic_fallback") is not semantic_fallback:
        raise ValueError(f"{label} semantic fallback flag mismatch")
    if report.get("coordinate_fallback") is not coordinate_fallback:
        raise ValueError(f"{label} coordinate fallback flag mismatch")


def _surface(
    baseline: dict[str, Any],
    child: dict[str, Any],
    baseline_path: Path,
    child_path: Path,
) -> dict[str, Any]:
    baseline_cases = baseline["cases"]
    child_cases = child["cases"]
    if [(row["task"], row["seed"]) for row in baseline_cases] != [
        (row["task"], row["seed"]) for row in child_cases
    ]:
        raise ValueError("baseline and child episode order mismatch")
    return {
        "baseline": {
            "checkpoint": baseline["checkpoint"],
            "success_count": sum(bool(case["success"]) for case in baseline_cases),
            "success_rate": baseline["success_rate"],
            "grounded_step_count": sum(
                bool(step["grounded"]) for case in baseline_cases for step in case["steps"]
            ),
            "step_count": sum(len(case["steps"]) for case in baseline_cases),
            "successful_tasks": sorted(case["task"] for case in baseline_cases if case["success"]),
            "report": _identity(baseline_path),
        },
        "mind2web_child": {
            "checkpoint": child["checkpoint"],
            "success_count": sum(bool(case["success"]) for case in child_cases),
            "success_rate": child["success_rate"],
            "grounded_step_count": sum(
                bool(step["grounded"]) for case in child_cases for step in case["steps"]
            ),
            "step_count": sum(len(case["steps"]) for case in child_cases),
            "successful_tasks": sorted(case["task"] for case in child_cases if case["success"]),
            "report": _identity(child_path),
        },
        "paired_comparison": {
            "same_episode_order": True,
            "identical_step_traces_excluding_wall_ms": sum(
                _stable_steps(row) == _stable_steps(child_row)
                for row, child_row in zip(baseline_cases, child_cases, strict=True)
            ),
            "success_delta": child["success_rate"] - baseline["success_rate"],
            "grounded_step_delta": sum(
                bool(step["grounded"]) for case in child_cases for step in case["steps"]
            )
            - sum(bool(step["grounded"]) for case in baseline_cases for step in case["steps"]),
        },
    }


def assemble(
    semantic_baseline_path: Path,
    semantic_child_path: Path,
    coordinate_baseline_path: Path,
    coordinate_child_path: Path,
    output: Path,
) -> dict[str, Any]:
    semantic_baseline = _load(semantic_baseline_path)
    semantic_child = _load(semantic_child_path)
    coordinate_baseline = _load(coordinate_baseline_path)
    coordinate_child = _load(coordinate_child_path)
    _validate_report(
        "semantic baseline",
        semantic_baseline,
        BASELINE_SHA256,
        semantic_fallback=True,
        coordinate_fallback=False,
    )
    _validate_report(
        "semantic child",
        semantic_child,
        MIND2WEB_CHILD_SHA256,
        semantic_fallback=True,
        coordinate_fallback=False,
    )
    _validate_report(
        "coordinate baseline",
        coordinate_baseline,
        BASELINE_SHA256,
        semantic_fallback=False,
        coordinate_fallback=True,
    )
    _validate_report(
        "coordinate child",
        coordinate_child,
        MIND2WEB_CHILD_SHA256,
        semantic_fallback=False,
        coordinate_fallback=True,
    )
    semantic = _surface(semantic_baseline, semantic_child, semantic_baseline_path, semantic_child_path)
    coordinate = _surface(
        coordinate_baseline,
        coordinate_child,
        coordinate_baseline_path,
        coordinate_child_path,
    )
    payload: dict[str, Any] = {
        "kind": "localagent_m645_mind2web_browsergym_grounding_canary_receipt",
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
        "semantic_fallback": semantic,
        "coordinate_fallback": coordinate,
        "paired_checkpoint_identity": {
            "baseline_sha256": BASELINE_SHA256,
            "mind2web_child_sha256": MIND2WEB_CHILD_SHA256,
        },
        "runtime": semantic_baseline["runtime"],
        "claim_boundary": (
            "Bounded, non-official native BrowserGym/MiniWoB checkpoint-in-the-loop diagnostics over "
            "the same 16 fixed-seed episodes. Semantic fallback reads only the accessibility tree and "
            "did not improve success (both checkpoints 4/16); coordinate fallback reads live DOM "
            "clickable geometry and raises both checkpoints to 8/16, showing an environment-grounding "
            "bridge rather than a Mind2Web policy-transfer gain. The realistic_browser vocabulary is "
            "not an official Mind2Web or BrowserGym score, no screenshots or real accounts were used, "
            "and no email/Notion side effect ran."
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
    parser.add_argument("--semantic-baseline", type=Path, required=True)
    parser.add_argument("--semantic-child", type=Path, required=True)
    parser.add_argument("--coordinate-baseline", type=Path, required=True)
    parser.add_argument("--coordinate-child", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        args.semantic_baseline,
        args.semantic_child,
        args.coordinate_baseline,
        args.coordinate_child,
        args.out,
    )
    print(json.dumps({"semantic": payload["semantic_fallback"], "coordinate": payload["coordinate_fallback"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
