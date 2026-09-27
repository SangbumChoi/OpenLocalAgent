#!/usr/bin/env python3
"""Assemble a self-hashed public cross-surface warm/random transfer receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from localagent.data.conversation_artifact import canonical_json_bytes


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _source_signature(receipt: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [
        {
            "label": source["label"],
            "rows": source["rows"],
            "input": source["input"],
            "public_reference": source["public_reference"],
            "revisions": source.get("revisions", []),
            "splits": source.get("splits", []),
            "visual_input_omitted_rows": source.get("visual_input_omitted_rows", 0),
        }
        for source in receipt[key]
    ]


def _assert_matched(warm: dict[str, Any], random: dict[str, Any]) -> None:
    if warm.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("warm report has the wrong kind")
    if random.get("kind") != "localagent_cross_surface_public_continuation_report":
        raise ValueError("random report has the wrong kind")
    if warm.get("parent") != random.get("parent"):
        raise ValueError("parent checkpoint mismatch")
    if warm.get("rows") != random.get("rows"):
        raise ValueError("row-count mismatch")
    for key in ("train_sources", "eval_sources"):
        if _source_signature(warm, key) != _source_signature(random, key):
            raise ValueError(f"{key} mismatch")
    warm_hyper = dict(warm["hyperparameters"])
    random_hyper = dict(random["hyperparameters"])
    for hyper in (warm_hyper, random_hyper):
        hyper.pop("backbone_init", None)
        hyper.pop("random_backbone_seed", None)
    if warm_hyper != random_hyper:
        raise ValueError("training hyperparameter mismatch")
    if warm["hyperparameters"].get("backbone_init", "parent") != "parent":
        raise ValueError("warm report is not parent-initialized")
    if random["hyperparameters"].get("backbone_init") != "random":
        raise ValueError("random report is not random-initialized")


def assemble(*, warm_path: Path, random_path: Path, comparison_path: Path, output_path: Path) -> dict[str, Any]:
    warm = _load(warm_path)
    random = _load(random_path)
    comparison = _load(comparison_path)
    _assert_matched(warm, random)
    if comparison.get("kind") != "localagent_cross_surface_transfer_ablation_report":
        raise ValueError("comparison report has the wrong kind")
    if comparison.get("parent") != warm.get("parent"):
        raise ValueError("comparison is not bound to the parent checkpoint")
    if comparison.get("warm_start_receipt") != warm.get("child"):
        raise ValueError("comparison is not bound to the warm child")
    if comparison.get("random_backbone_receipt") != random.get("child"):
        raise ValueError("comparison is not bound to the random child")

    source_labels = [source["label"] for source in warm["train_sources"]]
    aggregate = comparison["aggregate"]
    warm_after = float(aggregate["warm_after_token_accuracy"])
    random_after = float(aggregate["random_after_token_accuracy"])
    warm_groups = warm["weight_transfer"]["groups"]
    body: dict[str, Any] = {
        "kind": "localagent_cross_surface_public_transfer_receipt",
        "schema_version": 1,
        "experiment": {
            "id": "cross_surface_public_continuation_v1",
            "surfaces": source_labels,
            "protocol": (
                f"Matched {warm['hyperparameters']['steps']}-step continuation from the deployed "
                "BPE parent and a shape-matched "
                "random-backbone control. Source-local parent/slot disjointness is enforced; "
                f"evaluation is teacher-forced assistant-token accuracy at "
                f"max_seq_len={warm['hyperparameters']['max_seq_len']}."
            ),
            "environment": {"device": warm["hyperparameters"]["device"]},
        },
        "parent": warm["parent"],
        # Keep an explicit alias for the fail-closed publication gate, which expects the
        # checkpoint identity under ``parent_checkpoint`` for unified transfer receipts.
        "parent_checkpoint": warm["parent"],
        "children": {"warm_start": warm["child"], "random_backbone": random["child"]},
        "reports": {
            "warm_start": _identity(warm_path),
            "random_backbone": _identity(random_path),
            "comparison": _identity(comparison_path),
        },
        "rows": warm["rows"],
        "sources": {
            "train": _source_signature(warm, "train_sources"),
            "eval": _source_signature(warm, "eval_sources"),
        },
        "split_contract": warm["split_contract"],
        "hyperparameters": warm["hyperparameters"],
        "comparison": {
            "arm_contract": comparison["arm_contract"],
            "aggregate": comparison["aggregate"],
            "surfaces": comparison["surfaces"],
            "decision": comparison["decision"],
            "warm_weight_groups": comparison["warm_weight_groups"],
            "random_weight_groups": comparison["random_weight_groups"],
        },
        "weight_transfer_analysis": {
            "warm": {
                "compatibility": warm["weight_transfer"]["compatibility"],
                "groups": warm["weight_transfer"]["groups"],
            },
            "random": {
                "compatibility": random["weight_transfer"]["compatibility"],
                "groups": random["weight_transfer"]["groups"],
            },
        },
        "interpretation": {
            "result": (
                f"The parent-initialized arm reaches {warm_after:.4%} versus "
                f"{random_after:.4%} for the matched random-backbone control after "
                f"{warm['hyperparameters']['steps']} updates "
                f"({float(aggregate['warm_minus_random_after_pp']):+.2f} percentage points)."
            ),
            "weight_transfer": (
                "The warm child moves shared groups by "
                f"embedding {float(warm_groups['embedding']['relative_delta_l2']):.3%}, "
                f"attention/mixer {float(warm_groups['attention_or_mixer']['relative_delta_l2']):.3%}, "
                f"FFN {float(warm_groups['ffn']['relative_delta_l2']):.3%}, and "
                f"normalization {float(warm_groups['normalization']['relative_delta_l2']):.3%}; "
                "action heads remain unchanged. This supports compatibility and a lower-rate "
                "backbone/high-rate-head recipe, not optimality."
            ),
        },
        "claim_boundary": (
            "Diagnostic public-train-only text/accessibility continuation across "
            f"{', '.join(source_labels)} projections. No official benchmark score, native Android, "
            "BrowserGym, desktop VM, screenshot grounding, MCP server, real email/Notion side effect, "
            "or external-account claim is made."
        ),
    }
    body["receipt_self_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = assemble(
        warm_path=args.warm_report,
        random_path=args.random_report,
        comparison_path=args.comparison,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
