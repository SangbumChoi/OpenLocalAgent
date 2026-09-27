#!/usr/bin/env python3
"""Assemble the current m585 policy-aligned public transfer receipt.

The experiment trains on four normalized public projections and evaluates a held-out AgentNet
desktop projection without training on it.  ToolACE is retained as an explicit matrix candidate,
but the executable catalog still requires a terms/split review; the receipt therefore cannot
promote this child to the production training plan or WebGPU export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_cross_surface_controls import compare


KIND = "localagent_m607_current_policy_transfer"
PARENT_SHA256 = "6553dc2b161c03a916379fb77f174866143da6ef87173be07a12b57c4417b1ff"
TRAIN_LABELS = ("androidcontrol", "mind2web", "xlam", "toolace")
EVAL_LABELS = (*TRAIN_LABELS, "agentnet")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"receipt must be an object: {path}")
    return payload


def assemble(warm: dict[str, Any], random: dict[str, Any], *, warm_path: Path, random_path: Path):
    if warm.get("parent", {}).get("sha256") != PARENT_SHA256:
        raise ValueError("warm report is not bound to the m585 parent")
    if random.get("parent", {}).get("sha256") != PARENT_SHA256:
        raise ValueError("random report is not bound to the m585 parent")
    if tuple(source["label"] for source in warm["train_sources"]) != TRAIN_LABELS:
        raise ValueError("warm report has unexpected train labels")
    if tuple(source["label"] for source in warm["eval_sources"]) != EVAL_LABELS:
        raise ValueError("warm report has unexpected eval labels")
    if warm["rows"] != {"train": 32, "eval": 40} or random["rows"] != warm["rows"]:
        raise ValueError("unexpected m607 row counts")
    comparison = compare(warm, random)
    body: dict[str, Any] = {
        "kind": KIND,
        "schema_version": 1,
        "parent_checkpoint": warm["parent"],
        "input_reports": {"warm": _identity(warm_path), "random": _identity(random_path)},
        "protocol": {
            "train_labels": list(TRAIN_LABELS),
            "held_out_labels": ["agentnet"],
            "rows": warm["rows"],
            "steps": warm["hyperparameters"]["steps"],
            "batch_size": warm["hyperparameters"]["batch_size"],
            "learning_rate": warm["hyperparameters"]["learning_rate"],
            "max_seq_len": warm["hyperparameters"]["max_seq_len"],
            "max_train_rows_per_source": warm["hyperparameters"]["max_train_rows_per_source"],
            "max_eval_rows_per_source": warm["hyperparameters"]["max_eval_rows_per_source"],
            "seed": warm["hyperparameters"]["seed"],
            "random_backbone_seed": random["hyperparameters"]["random_backbone_seed"],
            "split_contract": warm["split_contract"],
        },
        "source_policy": {
            "androidcontrol": "executable_catalog_train",
            "mind2web": "executable_catalog_train_projection",
            "xlam": "executable_catalog_train_projection_from_public_derivative",
            "toolace": "public_matrix_train_candidate; executable_catalog_terms_review_required",
            "agentnet": "held_out_evaluation_only; no AgentNet training rows used",
        },
        "train_sources": warm["train_sources"],
        "eval_sources": warm["eval_sources"],
        "comparison": comparison,
        "weight_transfer": {
            "warm": warm["weight_transfer"],
            "random": random["weight_transfer"],
        },
        "weight_transfer_analysis": {
            "warm": warm["weight_transfer"],
            "random": random["weight_transfer"],
            "comparison": comparison,
        },
        "decision": {
            "warm_beats_random_all_surfaces": comparison["decision"]
            == "warm_start_dominates_matched_random_on_all_surfaces",
            "warm_minus_random_after_pp": comparison["aggregate"]["warm_minus_random_after_pp"],
            "reuse_parent_as_initialization_candidate": True,
            "export_child_to_webgpu": False,
            "promote_toolace_to_executable_training_plan": False,
            "native_promotion": False,
        },
        "claim_boundary": (
            "Matched current-checkpoint public projection transfer only. The warm arm trains four "
            "normalized projections and evaluates a source-disjoint AgentNet text projection; all "
            "screenshots are omitted, and no Android emulator, browser, desktop VM, MCP service, "
            "email/Notion account, or external side effect ran. This is not an official score for "
            "AndroidControl, Mind2Web, xLAM, ToolACE, or AgentNet. ToolACE remains a matrix train "
            "candidate pending the executable-catalog terms/split review."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite receipt: {args.out}")
    receipt = assemble(
        _load(args.warm_report),
        _load(args.random_report),
        warm_path=args.warm_report,
        random_path=args.random_report,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
