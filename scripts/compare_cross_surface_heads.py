#!/usr/bin/env python3
"""Compare matched frozen-backbone route/selector-head adaptation receipts."""

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
    expected_kind = "localagent_cross_surface_dispatch_head_report"
    if warm.get("kind") != expected_kind or random.get("kind") != expected_kind:
        raise ValueError("head receipts have the wrong kind")
    for key in ("rows", "train_sources", "eval_sources"):
        if key == "rows":
            if warm[key] != random[key]:
                raise ValueError("row-count mismatch")
        elif _source_signature(warm, key) != _source_signature(random, key):
            raise ValueError(f"{key} mismatch")
    warm_hyper = dict(warm["hyperparameters"])
    random_hyper = dict(random["hyperparameters"])
    if warm_hyper != random_hyper:
        raise ValueError("head hyperparameter mismatch")
    if not warm_hyper.get("backbone_frozen"):
        raise ValueError("warm backbone is not frozen")
    if not random_hyper.get("backbone_frozen"):
        raise ValueError("random backbone is not frozen")
    if warm["parent"]["sha256"] == random["parent"]["sha256"]:
        raise ValueError("warm and random head probes must have distinct continuation parents")


def _aggregate(metrics: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    rows = sum(int(item["rows"]) for item in metrics.values())
    tool_rows = sum(int(item["tool_rows"]) for item in metrics.values())
    route_correct = sum(float(item["route_accuracy"]) * int(item["rows"]) for item in metrics.values())
    selector_correct = sum(
        float(item["selector_top1_accuracy"]) * int(item["tool_rows"])
        for item in metrics.values()
    )
    return {
        "rows": rows,
        "tool_rows": tool_rows,
        "route_accuracy": route_correct / rows if rows else 0.0,
        "selector_top1_accuracy": selector_correct / tool_rows if tool_rows else 0.0,
    }


def compare(warm: dict[str, Any], random: dict[str, Any]) -> dict[str, Any]:
    """Build a source-matched, per-surface warm-vs-random head comparison."""

    _assert_matched(warm, random)
    surfaces: dict[str, Any] = {}
    for source in warm["eval_sources"]:
        label = source["label"]
        warm_before = warm["before_eval_by_source"][label]
        warm_after = warm["after_eval_by_source"][label]
        random_before = random["before_eval_by_source"][label]
        random_after = random["after_eval_by_source"][label]
        if (
            warm_after["rows"] != random_after["rows"]
            or warm_after["tool_rows"] != random_after["tool_rows"]
        ):
            raise ValueError(f"metric row mismatch for source {label!r}")
        surfaces[label] = {
            "rows": warm_after["rows"],
            "tool_rows": warm_after["tool_rows"],
            "warm_start": {"before": warm_before, "after": warm_after},
            "random_backbone": {"before": random_before, "after": random_after},
            "warm_minus_random_after_route_pp": 100.0
            * (warm_after["route_accuracy"] - random_after["route_accuracy"]),
            "warm_minus_random_after_selector_pp": 100.0
            * (warm_after["selector_top1_accuracy"] - random_after["selector_top1_accuracy"]),
            "warm_selector_better_after": warm_after["selector_top1_accuracy"]
            > random_after["selector_top1_accuracy"],
        }
    warm_after = _aggregate(
        {label: surface["warm_start"]["after"] for label, surface in surfaces.items()}
    )
    random_after = _aggregate(
        {label: surface["random_backbone"]["after"] for label, surface in surfaces.items()}
    )
    selector_deltas = [value["warm_minus_random_after_selector_pp"] for value in surfaces.values()]
    return {
        "kind": "localagent_cross_surface_dispatch_head_ablation_report",
        "schema_version": 1,
        "warm_start_receipt": warm["child"],
        "random_backbone_receipt": random["child"],
        "warm_parent": warm["parent"],
        "random_parent": random["parent"],
        "rows": warm["rows"],
        "train_sources": warm["train_sources"],
        "eval_sources": warm["eval_sources"],
        "hyperparameters": warm["hyperparameters"],
        "aggregate": {
            "warm_after": warm_after,
            "random_after": random_after,
            "warm_minus_random_route_pp": 100.0
            * (warm_after["route_accuracy"] - random_after["route_accuracy"]),
            "warm_minus_random_selector_pp": 100.0
            * (warm_after["selector_top1_accuracy"] - random_after["selector_top1_accuracy"]),
        },
        "surfaces": surfaces,
        "decision": (
            "head_adaptation_is_surface_specific_with_warm_selector_advantage"
            if max(selector_deltas) > 0.0 and min(selector_deltas) < 0.0
            else "head_adaptation_has_no_uniform_warm_start_selector_advantage"
        ),
        "claim_boundary": (
            "Matched frozen-backbone route/selector adaptation on public-train-only text/accessibility "
            "rows; this is not an official benchmark score, native environment success, "
            "screenshot-grounding result, or evidence of real email/Notion/MCP side effects."
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
