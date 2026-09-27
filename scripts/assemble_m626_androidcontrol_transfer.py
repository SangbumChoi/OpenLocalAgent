#!/usr/bin/env python3
"""Seal the matched AndroidControl continuation and weight-movement receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_cross_surface_controls import compare


DATASET = "OfficerChul/Android-Control-84k"
DATASET_URL = "https://huggingface.co/datasets/OfficerChul/Android-Control-84k"
ORIGINAL_URL = "https://github.com/google-research/google-research/tree/master/android_control"
REVISION = "hf:OfficerChul/Android-Control-84k@0248027f747c9d57bd09c14e8f044f9a8103dddd"
PARENT_SHA256 = "984152a802357e18387a6a28c93c9d30f43c5b0c9e9fede48caa24157716b43b"


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


def _check_arm(report: dict[str, Any], *, backbone_init: str) -> None:
    if report.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("unexpected continuation report kind")
    if report.get("parent", {}).get("sha256") != PARENT_SHA256:
        raise ValueError("report is not bound to the m624 warm child")
    if report.get("rows") != {"train": 512, "eval": 256}:
        raise ValueError("expected 512 train and 256 evaluation rows")
    hyperparameters = report.get("hyperparameters", {})
    if hyperparameters.get("backbone_init") != backbone_init:
        raise ValueError(f"expected {backbone_init} backbone initialization")
    if hyperparameters.get("steps") != 64 or hyperparameters.get("batch_size") != 4:
        raise ValueError("matched continuation hyperparameters changed")
    if hyperparameters.get("max_seq_len") != 512:
        raise ValueError("expected max_seq_len=512")
    for source in [*report.get("train_sources", []), *report.get("eval_sources", [])]:
        if source.get("label") != "android":
            raise ValueError("unexpected source label")
        if source.get("revisions") != [REVISION]:
            raise ValueError("AndroidControl revision mismatch")
        if source.get("source_families") != ["androidcontrol_json_mirror"]:
            raise ValueError("AndroidControl source-family mismatch")
        if source.get("visual_input_omitted_rows") != source.get("rows"):
            raise ValueError("receipt must account for every omitted screenshot")


def assemble(
    warm_path: Path,
    random_path: Path,
    train_manifest: Path,
    eval_manifest: Path,
) -> dict[str, Any]:
    warm = _load(warm_path)
    random = _load(random_path)
    _check_arm(warm, backbone_init="parent")
    _check_arm(random, backbone_init="random")
    if warm["parent"] != random["parent"]:
        raise ValueError("warm/random parents differ")
    comparison = compare(warm, random)
    payload: dict[str, Any] = {
        "kind": "localagent_m626_androidcontrol_warm_random_transfer",
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "url": DATASET_URL,
            "original_url": ORIGINAL_URL,
            "revision": REVISION,
            "license": "Apache-2.0 (reformatted mirror; verify upstream terms before redistribution)",
            "train_manifest": _identity(train_manifest),
            "eval_manifest": _identity(eval_manifest),
            "train_input_sha256": _load(train_manifest)["source"]["sha256"],
            "eval_input_sha256": _load(eval_manifest)["source"]["sha256"],
        },
        "parent_checkpoint": warm["parent"],
        "protocol": {
            "train_rows": 512,
            "eval_rows": 256,
            "steps": 64,
            "batch_size": 4,
            "learning_rate": 1.0e-5,
            "max_seq_len": 512,
            "random_backbone_seed": random["hyperparameters"]["random_backbone_seed"],
            "split_contract": "public mirror train/test preserved; source-local structural check",
            "screenshots_loaded": False,
            "visual_input_omitted": True,
            "grounding_evaluable": False,
            "official_leaderboard_score": False,
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
        "comparison": comparison,
        "weight_transfer_analysis": {
            "warm": warm["weight_transfer"],
            "random": random["weight_transfer"],
            "comparison": comparison,
        },
        "decision": {
            "warm_start_better_on_held_out_projection": comparison["aggregate"][
                "warm_start_better_after"
            ],
            "reuse_m624_as_mobile_initialization_candidate": True,
            "export_child_to_webgpu": False,
            "native_mobile_promotion": False,
            "reason": (
                "The warm arm improves the disjoint AndroidControl text projection and retains "
                "small shared-body movement, but all screenshots were omitted and no emulator or "
                "official AndroidControl judge ran. Keep the result as transfer evidence only."
            ),
        },
        "claim_boundary": (
            "Public-train-only AndroidControl continuation over a screenshot-omitted JSON mirror. "
            "This is not official AndroidControl accuracy, screenshot grounding, Android emulator "
            "success, AndroidWorld/MobileGym success, or real email/Notion/WebGPU side effects."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    receipt = assemble(args.warm_report, args.random_report, args.train_manifest, args.eval_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt["comparison"]["aggregate"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
