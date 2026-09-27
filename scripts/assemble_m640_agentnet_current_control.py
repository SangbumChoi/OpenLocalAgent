#!/usr/bin/env python3
"""Assemble a bounded current-checkpoint AgentNet text-projection control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


WARM_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
RANDOM_SHA256 = "390f1414260e118cd621af735fe6e87b01e8641b1cff650d594585e39b212e45"
SOURCE_REVISION = "d76ee50"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assemble(warm_full_path: Path, warm_subset_path: Path, random_subset_path: Path, output: Path) -> dict[str, Any]:
    warm_full = _load(warm_full_path)
    warm_subset = _load(warm_subset_path)
    random_subset = _load(random_subset_path)
    for label, report, expected in (
        ("warm_full", warm_full, WARM_SHA256),
        ("warm_subset", warm_subset, WARM_SHA256),
        ("random_subset", random_subset, RANDOM_SHA256),
    ):
        if report.get("kind") != "localagent_agentnet_text_projection_eval":
            raise ValueError(f"unexpected {label} report kind")
        if report.get("checkpoint", {}).get("sha256") != expected:
            raise ValueError(f"{label} checkpoint mismatch")
        if report.get("projection", {}).get("sha256") != warm_full.get("projection", {}).get("sha256"):
            raise ValueError(f"{label} projection hash mismatch")
        if report.get("source_revision") not in (None, SOURCE_REVISION):
            raise ValueError(f"{label} AgentNet source revision mismatch")
    if warm_subset.get("rows", {}).get("parents") != 4 or random_subset.get("rows", {}).get("parents") != 4:
        raise ValueError("expected four-parent matched subset")
    warm_overall = warm_subset["overall"]
    random_overall = random_subset["overall"]
    payload: dict[str, Any] = {
        "kind": "localagent_m640_agentnet_current_text_projection_control_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "xlangai/AgentNet",
            "dataset_url": "https://huggingface.co/datasets/xlangai/AgentNet",
            "source_url": "https://github.com/xlang-ai/OpenCUA",
            "source_revision": SOURCE_REVISION,
            "projection_sha256": warm_full["projection"]["sha256"],
        },
        "warm_full": {"checkpoint": warm_full["checkpoint"], "rows": warm_full["rows"], "overall": warm_full["overall"]},
        "matched_subset": {
            "parents": 4,
            "warm": {"checkpoint": warm_subset["checkpoint"], "rows": warm_subset["rows"], "overall": warm_overall},
            "random": {"checkpoint": random_subset["checkpoint"], "rows": random_subset["rows"], "overall": random_overall},
            "warm_minus_random": {
                "first_action_type_rate": warm_overall["first_action_type_rate"] - random_overall["first_action_type_rate"],
                "mean_total": warm_overall["mean_total"] - random_overall["mean_total"],
                "exact_trajectory_rate": warm_overall["exact_trajectory_rate"] - random_overall["exact_trajectory_rate"],
            },
        },
        "source_reports": {
            "warm_full": _identity(warm_full_path),
            "warm_subset": _identity(warm_subset_path),
            "random_subset": _identity(random_subset_path),
        },
        "claim_boundary": (
            "Current m626 AgentNet text-observation/action projection. The four-parent random arm "
            "is bounded because random decoding produced unbounded malformed action text on the "
            "full 16-parent run; no full-random score is claimed. Screenshots, desktop state, and "
            "the upstream AgentNetBench runtime were not used, so this is not native computer-use "
            "success or an official leaderboard result."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-full", type=Path, required=True)
    parser.add_argument("--warm-subset", type=Path, required=True)
    parser.add_argument("--random-subset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.warm_full, args.warm_subset, args.random_subset, args.out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
