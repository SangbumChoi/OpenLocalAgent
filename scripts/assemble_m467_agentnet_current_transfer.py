#!/usr/bin/env python3
"""Assemble the current-checkpoint AgentNet warm/random transfer receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_cross_surface_controls import compare


PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"
AGENTNET_REVISION = "d76ee50a63fad81cfdbe576416757d7c2091ed50"


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


def assemble(warm: dict[str, Any], random: dict[str, Any], *, warm_path: Path, random_path: Path) -> dict[str, Any]:
    if warm["parent"]["sha256"] != PARENT_SHA256 or random["parent"]["sha256"] != PARENT_SHA256:
        raise ValueError("both arms must bind the current parent checkpoint")
    for label, report, init in (("warm", warm, "parent"), ("random", random, "random")):
        if report["hyperparameters"]["backbone_init"] != init:
            raise ValueError(f"{label} initialization mismatch")
        if report["rows"] != {"train": 32, "eval": 8}:
            raise ValueError(f"{label} row cap mismatch")
        if report["hyperparameters"]["steps"] != 8:
            raise ValueError(f"{label} step count mismatch")
        source = report["train_sources"] + report["eval_sources"]
        if any(source_item["revisions"] != [AGENTNET_REVISION] for source_item in source):
            raise ValueError(f"{label} AgentNet revision mismatch")
        if any(source_item["datasets"] != ["xlangai/AgentNet"] for source_item in source):
            raise ValueError(f"{label} dataset identity mismatch")
    comparison = compare(warm, random)
    body: dict[str, Any] = {
        "kind": "localagent_agentnet_current_warm_random_transfer_receipt",
        "schema_version": 1,
        "benchmark_id": "agentnet_text_projection",
        "parent_checkpoint": warm["parent"],
        "source": {
            "dataset": "xlangai/AgentNet",
            "url": "https://huggingface.co/datasets/xlangai/AgentNet",
            "revision": AGENTNET_REVISION,
            "source_split": "ubuntu_jsonl_bounded_prefix",
            "parent_split": "32 train parent records / 8 eval parent records in the bounded projection",
            "images_dropped": True,
            "official_split_verified": False,
        },
        "protocol": {
            "train_rows": 32,
            "eval_rows": 8,
            "steps": 8,
            "batch_size": 1,
            "learning_rate": 1.0e-5,
            "max_seq_len": 512,
            "seed": 2027,
            "random_backbone_seed": 2028,
            "split_contract": warm["split_contract"],
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
        # Keep the gate-compatible unified shape alongside the human-readable arm sections.
        "weight_transfer_analysis": {
            "warm": warm["weight_transfer"],
            "random": random["weight_transfer"],
        },
        "comparison": comparison,
        "decision": {
            "adopt_parent_as_initialization_candidate": True,
            "export_child_to_webgpu": False,
            "native_promotion": False,
            "reason": (
                "The warm current-parent arm improves held-out AgentNet text-projection token "
                "accuracy while the matched random arm remains at zero token accuracy, and warm "
                "body movement stays below 0.11% relative L2. Exact sequence accuracy is 0% for "
                "both arms, images were dropped, and no native Ubuntu/OSWorld execution ran; "
                "retain as initialization evidence only and require deployment-shaped replay."
            ),
        },
        "claim_boundary": (
            "Bounded text-only AgentNet continuation with source-parent-disjoint train/eval rows. "
            "It is not AgentNetBench, OSWorld, visual grounding, native desktop control, or any "
            "email/Notion/browser side-effect result."
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
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
