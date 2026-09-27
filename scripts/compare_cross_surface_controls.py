#!/usr/bin/env python3
"""Compare matched warm-start and random-backbone cross-surface receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt must be an object: {path}")
    return value


def _source_signature(receipt: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [
        {
            "label": source["label"],
            "rows": source["rows"],
            "input_sha256": source["input"]["sha256"],
            "public_reference": source["public_reference"],
        }
        for source in receipt[key]
    ]


def _assert_matched(warm: dict[str, Any], random: dict[str, Any]) -> None:
    if warm.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("warm receipt has the wrong kind")
    if random.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("random receipt has the wrong kind")
    if warm.get("parent", {}).get("sha256") != random.get("parent", {}).get("sha256"):
        raise ValueError("parent checkpoint mismatch")
    for key in ("rows", "train_sources", "eval_sources"):
        if key == "rows":
            if warm[key] != random[key]:
                raise ValueError("row-count mismatch")
        elif _source_signature(warm, key) != _source_signature(random, key):
            raise ValueError(f"{key} mismatch")
    warm_hyper = dict(warm["hyperparameters"])
    random_hyper = dict(random["hyperparameters"])
    warm_hyper.pop("backbone_init", None)
    random_hyper.pop("backbone_init", None)
    warm_hyper.pop("random_backbone_seed", None)
    random_hyper.pop("random_backbone_seed", None)
    if warm_hyper != random_hyper:
        raise ValueError("training hyperparameter mismatch")
    if warm["hyperparameters"].get("backbone_init", "parent") != "parent":
        raise ValueError("warm receipt is not parent-initialized")
    if random["hyperparameters"].get("backbone_init") != "random":
        raise ValueError("random receipt is not random-initialized")


def compare(warm: dict[str, Any], random: dict[str, Any]) -> dict[str, Any]:
    """Build a source-matched, per-surface warm-vs-random comparison."""

    _assert_matched(warm, random)
    surfaces: dict[str, Any] = {}
    for source in warm["eval_sources"]:
        label = source["label"]
        warm_before = warm["before"]["eval_by_source"][label]
        warm_after = warm["after"]["eval_by_source"][label]
        random_before = random["before"]["eval_by_source"][label]
        random_after = random["after"]["eval_by_source"][label]
        surfaces[label] = {
            "rows": warm_after["rows"],
            "warm_start": {
                "before_token_accuracy": warm_before["assistant_token_accuracy"],
                "after_token_accuracy": warm_after["assistant_token_accuracy"],
                "delta_token_accuracy": (
                    warm_after["assistant_token_accuracy"]
                    - warm_before["assistant_token_accuracy"]
                ),
            },
            "random_backbone": {
                "before_token_accuracy": random_before["assistant_token_accuracy"],
                "after_token_accuracy": random_after["assistant_token_accuracy"],
                "delta_token_accuracy": (
                    random_after["assistant_token_accuracy"]
                    - random_before["assistant_token_accuracy"]
                ),
            },
            "warm_minus_random_after_pp": 100.0
            * (warm_after["assistant_token_accuracy"] - random_after["assistant_token_accuracy"]),
            "warm_start_better_after": warm_after["assistant_token_accuracy"]
            > random_after["assistant_token_accuracy"],
        }
    aggregate_warm = warm["after"]["eval"]["assistant_token_accuracy"]
    aggregate_random = random["after"]["eval"]["assistant_token_accuracy"]
    aggregate_delta_pp = 100.0 * (aggregate_warm - aggregate_random)
    return {
        "kind": "localagent_cross_surface_transfer_ablation_report",
        "schema_version": 1,
        "warm_start_receipt": warm["child"],
        "random_backbone_receipt": random["child"],
        "parent": warm["parent"],
        "rows": warm["rows"],
        "train_sources": warm["train_sources"],
        "eval_sources": warm["eval_sources"],
        "arm_contract": {
            "warm_backbone_init": warm["hyperparameters"].get("backbone_init", "parent"),
            "random_backbone_init": random["hyperparameters"]["backbone_init"],
            "random_backbone_seed": random["hyperparameters"]["random_backbone_seed"],
        },
        "hyperparameters": warm["hyperparameters"],
        "aggregate": {
            "warm_after_token_accuracy": aggregate_warm,
            "random_after_token_accuracy": aggregate_random,
            "warm_minus_random_after_pp": aggregate_delta_pp,
            "warm_start_better_after": aggregate_warm > aggregate_random,
        },
        "surfaces": surfaces,
        "warm_weight_groups": warm["weight_transfer"]["groups"],
        "random_weight_groups": random["weight_transfer"]["groups"],
        "decision": (
            "retain_parent_as_compatible_initialization_but_keep_surface_specific_adapters"
            if aggregate_delta_pp <= 0.0 or not all(
                item["warm_start_better_after"] for item in surfaces.values()
            )
            else "warm_start_dominates_matched_random_on_all_surfaces"
        ),
        "claim_boundary": (
            "Matched public-train-only text/accessibility continuation comparison; this is not an "
            "official benchmark score, native environment success, screenshot-grounding result, "
            "or evidence of real email/Notion/MCP side effects."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(_load(args.warm_report), _load(args.random_report))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
