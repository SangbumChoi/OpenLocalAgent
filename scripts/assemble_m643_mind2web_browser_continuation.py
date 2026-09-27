#!/usr/bin/env python3
"""Seal the source-disjoint Mind2Web browser continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_cross_surface_controls import compare


CURRENT_SHA256 = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
DATASET_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"


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
    for label, report in (("warm", warm), ("random", random)):
        if report["parent"]["sha256"] != CURRENT_SHA256:
            raise ValueError(f"{label} report is not based on current m626")
        if report["train_sources"][0]["public_reference"]["dataset"] != "osunlp/Mind2Web":
            raise ValueError(f"{label} report is not Mind2Web")
        if report["hyperparameters"]["max_train_rows_per_source"] != 0:
            raise ValueError("Mind2Web continuation must use all 96 train rows")
    comparison = compare(warm, random)
    if not comparison["decision"].startswith("warm_start_dominates"):
        raise ValueError("warm Mind2Web arm did not dominate random")
    train_source = warm["train_sources"][0]
    eval_source = warm["eval_sources"][0]
    payload: dict[str, Any] = {
        "kind": "localagent_m643_mind2web_current_browser_continuation_receipt",
        "schema_version": 1,
        "benchmark": {
            "dataset": "osunlp/Mind2Web",
            "dataset_url": "https://huggingface.co/datasets/osunlp/Mind2Web",
            "dataset_revision": DATASET_REVISION,
            "projection": "normalized DOM/action Conversation projection",
        },
        "parent_checkpoint": warm["parent"],
        "children": {"warm": warm["child"], "random": random["child"]},
        "rows": {"train": train_source["rows"], "eval": eval_source["rows"]},
        "split_contract": {
            "mode": "source_local_parent_and_slot_disjoint",
            "validated_by_training_runner": True,
            "train_parent_records": train_source["unique_parent_records"],
            "eval_parent_records": eval_source["unique_parent_records"],
            "visual_input_omitted_rows": train_source["visual_input_omitted_rows"] + eval_source["visual_input_omitted_rows"],
        },
        "training": {
            "hyperparameters": warm["hyperparameters"],
            "warm_before": warm["before"]["eval_by_source"]["mind2web"],
            "warm_after": warm["after"]["eval_by_source"]["mind2web"],
            "random_before": random["before"]["eval_by_source"]["mind2web"],
            "random_after": random["after"]["eval_by_source"]["mind2web"],
            "warm_weight_groups": warm["weight_transfer"]["groups"],
            "random_weight_groups": random["weight_transfer"]["groups"],
        },
        "comparison": comparison,
        "source_inputs": {"train": train_source["input"], "eval": eval_source["input"]},
        "decision": {
            "retain_warm_browser_initialization": True,
            "keep_native_browsergym_gate_separate": True,
            "export_to_webgpu_as_native_browser_agent": False,
            "reason": (
                "Warm initialization improves the held-out Mind2Web text projection, but exact "
                "sequence accuracy remains zero and no live BrowserGym site or side effect ran."
            ),
        },
        "input_reports": {"warm": _identity(warm_path), "random": _identity(random_path)},
        "claim_boundary": (
            "Public Mind2Web normalized DOM/action continuation only. This is not the official "
            "Mind2Web test score, BrowserGym score, screenshot grounding, live website control, "
            "or an external side effect."
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
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.warm_report, args.random_report, args.out)
    print(json.dumps(report["comparison"]["aggregate"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
