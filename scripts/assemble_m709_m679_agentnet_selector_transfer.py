#!/usr/bin/env python3
"""Assemble the current m679 AgentNet selector transfer and text-projection receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--warm-eval", type=Path, required=True)
    parser.add_argument("--random-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.parent,
        args.train,
        args.eval,
        args.warm,
        args.random,
        args.training_report,
        args.warm_eval,
        args.random_eval,
    ]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise SystemExit(f"missing receipt input(s): {', '.join(missing)}")
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    warm_eval = json.loads(args.warm_eval.read_text(encoding="utf-8"))
    random_eval = json.loads(args.random_eval.read_text(encoding="utf-8"))
    if training.get("parent", {}).get("sha256") != identity(args.parent)["sha256"]:
        raise ValueError("training report parent hash does not match --parent")
    if warm_eval.get("checkpoint", {}).get("sha256") != identity(args.warm)["sha256"]:
        raise ValueError("warm evaluation checkpoint hash does not match --warm")
    if random_eval.get("checkpoint", {}).get("sha256") != identity(args.random)["sha256"]:
        raise ValueError("random evaluation checkpoint hash does not match --random")
    if warm_eval.get("projection", {}).get("sha256") != random_eval.get("projection", {}).get("sha256"):
        raise ValueError("warm/random projections differ")
    payload = {
        "kind": "localagent_m709_m679_agentnet_selector_transfer",
        "schema_version": 1,
        "parent": identity(args.parent),
        "source": {
            "dataset": "xlangai/AgentNet",
            "url": "https://huggingface.co/datasets/xlangai/AgentNet",
            "original_repository": "https://github.com/xlang-ai/OpenCUA",
            "revision": "d76ee50a63fad81cfdbe576416757d7c2091ed50",
            "train_projection": identity(args.train),
            "eval_projection": identity(args.eval),
            "official_split_verified": False,
            "screenshots_consumed": False,
            "desktop_runtime_executed": False,
        },
        "training": {
            "trainer": "scripts/train_agentnet_selector_repair.py",
            "trainer_sha256": identity(Path("scripts/train_agentnet_selector_repair.py"))["sha256"],
            "train_rows": training["rows"]["train"],
            "eval_rows": training["rows"]["eval"],
            "tool_surface": training["rows"]["tool_surface"],
            "steps": 400,
            "batch_size": 128,
            "learning_rate": 0.005,
            "seed": 2042,
            "backbone_frozen": True,
            "warm": training["warm"],
            "random": training["random"],
            "warm_checkpoint": identity(args.warm),
            "random_checkpoint": identity(args.random),
            "training_report": identity(args.training_report),
        },
        "evaluation": {
            "evaluator": "scripts/evaluate_agentnet_text.py",
            "evaluator_sha256": identity(Path("scripts/evaluate_agentnet_text.py"))["sha256"],
            "warm": {
                "report": identity(args.warm_eval),
                "metrics": warm_eval["overall"],
                "completeness_verified": warm_eval["completeness"]["verified"],
            },
            "random": {
                "report": identity(args.random_eval),
                "metrics": random_eval["overall"],
                "completeness_verified": random_eval["completeness"]["verified"],
            },
            "max_parents": warm_eval["bounds"]["max_parents"],
            "projected_actions": warm_eval["rows"]["projected_actions"],
        },
        "weight_analysis": {
            "warm_selector_relative_l2": training["warm"]["relative_l2"],
            "random_selector_relative_l2": training["random"]["relative_l2"],
            "warm_minus_random_eval_mean_total": (
                warm_eval["overall"]["mean_total"] - random_eval["overall"]["mean_total"]
            ),
            "adoption_decision": "reject_agentnet_selector_for_webgpu",
            "reason": (
                "Warm and random controls have identical first-action and zero success/exact-"
                "trajectory rates; warm moves the selector farther while its bounded text score is "
                "worse than random. No visual input was consumed."
            ),
        },
        "claim_boundary": (
            "This is a matched offline AgentNet text-observation/action projection and selector "
            "weight-reuse diagnostic. It is not AgentNetBench, native desktop, screenshot-grounded, "
            "OSWorld, or WebGPU visual evidence; the adapted selector is not promoted."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["weight_analysis"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
