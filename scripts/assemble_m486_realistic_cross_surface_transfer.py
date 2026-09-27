#!/usr/bin/env python3
"""Seal the longer matched warm/random realistic-agent continuation canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assemble(*, warm: Path, random: Path, comparison: Path) -> dict[str, Any]:
    warm_report = _load(warm)
    random_report = _load(random)
    comparison_report = _load(comparison)
    if warm_report.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("warm report kind mismatch")
    if random_report.get("kind") != warm_report.get("kind"):
        raise ValueError("random report kind mismatch")
    if comparison_report.get("kind") != "localagent_cross_surface_transfer_ablation_report":
        raise ValueError("comparison kind mismatch")
    if warm_report["parent"]["sha256"] != random_report["parent"]["sha256"]:
        raise ValueError("matched arms use different parents")
    if comparison_report["parent"]["sha256"] != warm_report["parent"]["sha256"]:
        raise ValueError("comparison parent mismatch")
    if warm_report["rows"] != {"train": 59, "eval": 18}:
        raise ValueError(f"unexpected row counts: {warm_report['rows']}")
    warm_hyper = dict(warm_report["hyperparameters"])
    random_hyper = dict(random_report["hyperparameters"])
    warm_hyper.pop("backbone_init", None)
    random_hyper.pop("backbone_init", None)
    warm_hyper.pop("random_backbone_seed", None)
    random_hyper.pop("random_backbone_seed", None)
    if warm_hyper != random_hyper or warm_report["hyperparameters"]["steps"] != 16:
        raise ValueError("matched hyperparameter contract mismatch")
    labels = {source["label"] for source in warm_report["train_sources"] + warm_report["eval_sources"]}
    required = {
        "androidcontrol",
        "agentnet",
        "mind2web",
        "mcpmark_productivity",
        "mcpmark_filesystem",
        "mcpmark_playwright_extraction",
        "mcpmark_playwright_travel",
    }
    if labels != required:
        raise ValueError(f"unexpected source labels: {sorted(labels)}")
    payload: dict[str, Any] = {
        "kind": "localagent_realistic_cross_surface_transfer_receipt",
        "schema_version": 1,
        "benchmark_id": "realistic_agent_cross_surface_16step_canary",
        "sources": {
            "androidcontrol": {"dataset": "OfficerChul/Android-Control-84k", "url": "https://huggingface.co/datasets/OfficerChul/Android-Control-84k", "revision": "hf:OfficerChul/Android-Control-84k@train4096"},
            "agentnet": {"dataset": "xlangai/AgentNet", "url": "https://huggingface.co/datasets/xlangai/AgentNet", "revision": "d76ee50a63fad81cfdbe576416757d7c2091ed50"},
            "mind2web": {"dataset": "osunlp/Mind2Web", "url": "https://huggingface.co/datasets/osunlp/Mind2Web", "revision": "17ece8eb89862368edc0cc806acee6fca5163474"},
            "mcpmark_trajectory_log": {"dataset": "Jakumetsu/mcpmark-trajectory-log", "url": "https://huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log", "revision": "e50578f0ab904d8e6a7c576c387c1e76ae482c89"},
            "mcpmark": {"dataset": "MCPMark", "url": "https://github.com/eval-sys/mcpmark", "revision": "cd45b7f57923b9b3985467f5139927575f83141c", "official_split_verified": False},
        },
        "parent": warm_report["parent"],
        "children": {"warm": warm_report["child"], "random": random_report["child"]},
        "training": {
            "warm_report": _identity(warm),
            "random_report": _identity(random),
            "comparison": _identity(comparison),
            "train_sources": warm_report["train_sources"],
            "eval_sources": warm_report["eval_sources"],
            "rows": warm_report["rows"],
            "hyperparameters": warm_report["hyperparameters"],
            "split_contract": warm_report["split_contract"],
            "cap_contract": {"max_train_rows_per_source": 16, "max_eval_rows_per_source": 4, "max_seq_len": 128, "steps": 16, "purpose": "matched_canary_not_final_training_run"},
        },
        "aggregate": comparison_report["aggregate"],
        "surfaces": comparison_report["surfaces"],
        "transfer_decision": comparison_report["decision"],
        "weight_groups": {"warm": comparison_report["warm_weight_groups"], "random": comparison_report["random_weight_groups"]},
        "claim_boundary": "Matched public-train-only text/accessibility continuation across AndroidControl, AgentNet, Mind2Web, and MCPMark projections. This is a 16-step canary, not an official benchmark score, native environment result, screenshot-grounding result, or evidence of real email/Notion/MCP side effects.",
    }
    payload["receipt_self_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = assemble(warm=args.warm, random=args.random, comparison=args.comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": payload["receipt_self_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
