#!/usr/bin/env python3
"""Assemble the Android-Control whole-episode warm/random transfer receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_cross_surface_controls import compare


PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
SOURCE_REVISION = "hf:OfficerChul/Android-Control-84k@train4096"
SOURCE_DATASET = "google/androidcontrol"
SOURCE_URL = "https://github.com/google-research/google-research/tree/master/android_control"


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _check_arm(report: dict[str, Any], *, init: str) -> None:
    if report.get("parent", {}).get("sha256") != PARENT_SHA256:
        raise ValueError("arm does not bind the current parent checkpoint")
    if report["hyperparameters"].get("backbone_init") != init:
        raise ValueError(f"expected {init} initialization")
    if report.get("rows") != {"train": 32, "eval": 8}:
        raise ValueError("bounded row caps must be 32 train / 8 eval")
    if report["hyperparameters"].get("steps") != 8:
        raise ValueError("expected eight continuation steps")
    for source in report["train_sources"] + report["eval_sources"]:
        if source["public_reference"] != {"dataset": SOURCE_DATASET, "url": SOURCE_URL}:
            raise ValueError("Android-Control public reference mismatch")
        if source["revisions"] != [SOURCE_REVISION]:
            raise ValueError("Android-Control revision mismatch")
        if source["source_families"] != ["androidcontrol_json_mirror"]:
            raise ValueError("Android-Control source family mismatch")


def assemble(warm: dict[str, Any], random: dict[str, Any], *, warm_path: Path, random_path: Path) -> dict[str, Any]:
    _check_arm(warm, init="parent")
    _check_arm(random, init="random")
    comparison = compare(warm, random)
    body: dict[str, Any] = {
        "kind": "localagent_androidcontrol_current_warm_random_transfer_receipt",
        "schema_version": 1,
        "benchmark_id": "androidcontrol_text_projection",
        "parent_checkpoint": warm["parent"],
        "source": {
            "dataset": SOURCE_DATASET,
            "url": SOURCE_URL,
            "mirror": "OfficerChul/Android-Control-84k",
            "revision": SOURCE_REVISION,
            "source_split": "train4096",
            "source_rows": 4096,
            "source_episodes": 3483,
            "whole_episode_split": "sha256_bucket_v1 (80% train / 20% eval by episode)",
            "images_dropped": True,
            "grounding_evaluable": False,
            "official_split_verified": False,
        },
        "protocol": {
            "train_rows": 32,
            "eval_rows": 8,
            "steps": 8,
            "batch_size": 1,
            "learning_rate": 1.0e-5,
            "max_seq_len": 256,
            "seed": 2027,
            "random_backbone_seed": 2028,
            "split_contract": {
                "mode": "whole_episode_sha256_bucket",
                "no_episode_overlap": True,
                "source_rows_train": 3269,
                "source_rows_eval": 827,
                "source_episodes_train": 2767,
                "source_episodes_eval": 716,
            },
        },
        "inputs": {
            "warm_report": _identity(warm_path),
            "random_report": _identity(random_path),
            "train": warm["train_sources"],
            "eval": warm["eval_sources"],
        },
        "warm": {
            "child": warm["child"],
            "before": warm["before"],
            "after": warm["after"],
            "weight_transfer": warm["weight_transfer"],
        },
        "random": {
            "child": random["child"],
            "before": random["before"],
            "after": random["after"],
            "weight_transfer": random["weight_transfer"],
        },
        "weight_transfer_analysis": {
            "warm": warm["weight_transfer"],
            "random": random["weight_transfer"],
        },
        "comparison": comparison,
        "decision": {
            "adopt_parent_as_initialization_candidate": comparison["aggregate"]["warm_start_better_after"],
            "export_child_to_webgpu": False,
            "native_promotion": False,
            "reason": (
                "The matched warm current-parent arm improves held-out Android-Control text-"
                "projection token accuracy while the random backbone remains at zero token "
                "accuracy. Warm backbone movement stays below 0.10% relative L2. Exact sequence "
                "accuracy is 0%, screenshots were omitted, and no Android emulator executed; "
                "retain as initialization evidence only and require native visual replay."
            ),
        },
        "claim_boundary": (
            "Bounded text-only Android-Control continuation over a deterministic whole-episode "
            "split. This is not an official AndroidControl score, screenshot grounding result, "
            "native mobile success, or evidence of email/Notion side effects."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assemble(
        _load(args.warm_report),
        _load(args.random_report),
        warm_path=args.warm_report,
        random_path=args.random_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["comparison"]["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
