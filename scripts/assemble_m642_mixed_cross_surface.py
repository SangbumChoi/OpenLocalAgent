#!/usr/bin/env python3
"""Seal the source-disjoint mixed mobile/browser/tool continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_cross_surface_controls import compare


CURRENT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assemble(warm_path: Path, random_path: Path, output: Path) -> dict[str, Any]:
    warm = _load(warm_path)
    random = _load(random_path)
    if warm["parent"]["sha256"] != CURRENT_SHA256 or random["parent"]["sha256"] != CURRENT_SHA256:
        raise ValueError("mixed continuation must use current m626 parent")
    if warm["hyperparameters"]["backbone_init"] != "parent" or random["hyperparameters"]["backbone_init"] != "random":
        raise ValueError("warm/random arm contract mismatch")
    comparison = compare(warm, random)
    if not comparison["decision"].startswith("warm_start_dominates"):
        raise ValueError("warm arm did not dominate on every measured surface")
    selected_train_rows = sum(source["rows"] for source in warm["train_sources"])
    selected_eval_rows = sum(source["rows"] for source in warm["eval_sources"])
    payload: dict[str, Any] = {
        "kind": "localagent_m642_mixed_cross_surface_warm_random_receipt",
        "schema_version": 1,
        "parent_checkpoint": warm["parent"],
        "children": {"warm": warm["child"], "random": random["child"]},
        "public_sources": {
            source["label"]: {
                "train_rows_selected": next(item["rows"] for item in warm["train_sources"] if item["label"] == source["label"]),
                "eval_rows_selected": source["rows"],
                "dataset": source["public_reference"]["dataset"],
                "url": source["public_reference"]["url"],
                "train_input": next(item["input"] for item in warm["train_sources"] if item["label"] == source["label"]),
                "eval_input": source["input"],
            }
            for source in warm["eval_sources"]
        },
        "rows": {"train_selected": selected_train_rows, "eval_selected": selected_eval_rows},
        "split_contract": {
            "mode": "source_local_parent_and_slot_disjoint",
            "validated_by_training_runner": True,
            "cross_source_slot_reuse_allowed": True,
            "visual_input_omitted_rows": {
                source["label"]: source["visual_input_omitted_rows"]
                for source in warm["train_sources"] + warm["eval_sources"]
            },
        },
        "training": {
            "hyperparameters": warm["hyperparameters"],
            "warm_before": warm["before"],
            "warm_after": warm["after"],
            "random_before": random["before"],
            "random_after": random["after"],
            "warm_weight_groups": warm["weight_transfer"]["groups"],
            "random_weight_groups": random["weight_transfer"]["groups"],
        },
        "comparison": comparison,
        "decision": {
            "retain_current_warm_initialization": True,
            "surface_specific_adapters_still_required": True,
            "export_to_webgpu_as_native_agent": False,
            "reason": (
                "Warm initialization dominates the matched random arm on AndroidControl, AgentNet, "
                "and redacted MCP trajectories, but exact sequence accuracy remains zero and no "
                "native emulator/desktop/MCP environment was executed."
            ),
        },
        "input_reports": {"warm": _identity(warm_path), "random": _identity(random_path)},
        "claim_boundary": (
            "Public-train-only mixed continuation over AndroidControl, AgentNet, and redacted MCPMark "
            "text projections. Rows are source-local bounded selections with parent/slot checks. This "
            "is not an official benchmark score, native mobile/browser/desktop/MCP execution, visual "
            "grounding, or real email/Notion side effects."
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
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.warm_report, args.random_report, args.out)
    print(json.dumps(report["comparison"]["aggregate"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
