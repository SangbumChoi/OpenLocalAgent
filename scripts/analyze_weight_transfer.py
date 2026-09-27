#!/usr/bin/env python
"""Measure pretrained-weight reuse between two LocalAgent checkpoints.

This is an audit tool, not a training recipe.  It checks model/tokenizer compatibility, quantifies
how much each shared tensor moved, identifies newly introduced action heads, and emits a
parameter-group recommendation for a subsequent controlled ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_MODEL_FIELDS = (
    "vocab_size",
    "d_model",
    "embed_dim",
    "n_layers",
    "n_loops",
    "n_heads",
    "n_kv_heads",
    "ffn_hidden",
    "max_seq_len",
    "rope_theta",
    "norm_eps",
    "tie_embeddings",
    "dropout",
    "qk_norm",
    "conv_kernel",
    "layer_types",
)
_HEAD_CONTAINERS = frozenset(
    {"tool_head", "ptr_head", "route_head", "dense_selector", "selector_proj", "value_head"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    if not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError(f"checkpoint has no state_dict mapping: {path}")
    return checkpoint


def _flatten_aux(value: object, prefix: str) -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu()}
    if not isinstance(value, Mapping):
        return {}
    flattened: dict[str, torch.Tensor] = {}
    for key, item in value.items():
        flattened.update(_flatten_aux(item, f"{prefix}.{key}"))
    return flattened


def _tensors(checkpoint: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model = {
        str(key): value.detach().cpu()
        for key, value in checkpoint["state_dict"].items()
        if isinstance(value, torch.Tensor)
    }
    auxiliary: dict[str, torch.Tensor] = {}
    for key in _HEAD_CONTAINERS:
        if key in checkpoint:
            auxiliary.update(_flatten_aux(checkpoint[key], key))
    return model, auxiliary


def _group(name: str) -> str:
    if name.startswith("embed"):
        return "embedding"
    if ".attn" in name or ".mixer" in name:
        return "attention_or_mixer"
    if ".ffn" in name:
        return "ffn"
    if "norm" in name:
        return "normalization"
    if name.startswith("lm_head") or name.startswith("output"):
        return "output"
    if "head" in name or "selector" in name:
        return "action_heads"
    return "other"


def _cosine(base: torch.Tensor, target: torch.Tensor) -> float:
    base_flat = base.to(dtype=torch.float64).reshape(-1)
    target_flat = target.to(dtype=torch.float64).reshape(-1)
    denominator = torch.linalg.vector_norm(base_flat) * torch.linalg.vector_norm(target_flat)
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(base_flat, target_flat) else 0.0
    return float(torch.dot(base_flat, target_flat) / denominator)


def _stats(base: torch.Tensor, target: torch.Tensor) -> dict[str, float | int | list[int]]:
    if tuple(base.shape) != tuple(target.shape):
        raise ValueError("shape mismatch")
    base_f = base.to(dtype=torch.float64)
    target_f = target.to(dtype=torch.float64)
    delta = target_f - base_f
    base_l2 = float(torch.linalg.vector_norm(base_f))
    delta_l2 = float(torch.linalg.vector_norm(delta))
    target_l2 = float(torch.linalg.vector_norm(target_f))
    return {
        "shape": list(base.shape),
        "parameters": base.numel(),
        "base_l2": base_l2,
        "target_l2": target_l2,
        "delta_l2": delta_l2,
        "relative_delta_l2": delta_l2 / max(base_l2, 1e-12),
        "cosine": _cosine(base, target),
    }


def _aggregate(per_tensor: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, dict[str, float | int]] = {}
    for name, stats in per_tensor.items():
        group = _group(name)
        total = groups.setdefault(
            group,
            {
                "parameters": 0,
                "base_l2_squared": 0.0,
                "target_l2_squared": 0.0,
                "delta_l2_squared": 0.0,
            },
        )
        total["parameters"] += int(stats["parameters"])
        total["base_l2_squared"] += float(stats["base_l2"]) ** 2
        total["target_l2_squared"] += float(stats["target_l2"]) ** 2
        total["delta_l2_squared"] += float(stats["delta_l2"]) ** 2
    for total in groups.values():
        base_l2 = math.sqrt(float(total.pop("base_l2_squared")))
        target_l2 = math.sqrt(float(total.pop("target_l2_squared")))
        delta_l2 = math.sqrt(float(total.pop("delta_l2_squared")))
        total.update(
            {
                "base_l2": base_l2,
                "target_l2": target_l2,
                "delta_l2": delta_l2,
                "relative_delta_l2": delta_l2 / max(base_l2, 1e-12),
            }
        )
    return groups


def analyze(base_path: str | Path, target_path: str | Path) -> dict[str, Any]:
    """Return a JSON-compatible transfer report for two checkpoints."""

    base_path = Path(base_path).resolve()
    target_path = Path(target_path).resolve()
    base = _load(base_path)
    target = _load(target_path)
    base_model, base_aux = _tensors(base)
    target_model, target_aux = _tensors(target)
    base_all = {**base_model, **base_aux}
    target_all = {**target_model, **target_aux}
    common = sorted(set(base_all) & set(target_all))
    shape_mismatches = {
        name: {"base": list(base_all[name].shape), "target": list(target_all[name].shape)}
        for name in common
        if tuple(base_all[name].shape) != tuple(target_all[name].shape)
    }
    per_tensor = {
        name: _stats(base_all[name], target_all[name])
        for name in common
        if name not in shape_mismatches
    }
    added = sorted(set(target_all) - set(base_all))
    removed = sorted(set(base_all) - set(target_all))
    base_cfg = base.get("cfg") if isinstance(base.get("cfg"), Mapping) else {}
    target_cfg = target.get("cfg") if isinstance(target.get("cfg"), Mapping) else {}
    config_mismatches = {
        field: {"base": base_cfg.get(field), "target": target_cfg.get(field)}
        for field in _MODEL_FIELDS
        if base_cfg.get(field) != target_cfg.get(field)
    }
    report = {
        "kind": "localagent_weight_transfer_analysis",
        "schema_version": 1,
        "base": {
            "path": str(base_path),
            "sha256": _sha256(base_path),
            "stage": base.get("stage"),
            "step": base.get("step"),
            "model_tensors": len(base_model),
            "auxiliary_tensors": len(base_aux),
            "lineage": base.get("lineage"),
        },
        "target": {
            "path": str(target_path),
            "sha256": _sha256(target_path),
            "stage": target.get("stage"),
            "step": target.get("step"),
            "model_tensors": len(target_model),
            "auxiliary_tensors": len(target_aux),
            "lineage": target.get("lineage"),
        },
        "compatibility": {
            "config_mismatches": config_mismatches,
            "shared_tensor_count": len(per_tensor),
            "shape_mismatches": shape_mismatches,
            "added_tensors": added,
            "removed_tensors": removed,
            "tokenizer_sha256_equal": (
                base.get("tokenizer", {}).get("sha256")
                == target.get("tokenizer", {}).get("sha256")
                if isinstance(base.get("tokenizer"), Mapping)
                and isinstance(target.get("tokenizer"), Mapping)
                else None
            ),
        },
        "groups": _aggregate(per_tensor),
        "per_tensor": per_tensor,
        "recommendation": {
            "backbone": "reuse shared same-shape tensors only after config and tokenizer checks pass",
            "action_heads": "initialize newly added tool/pointer/route/selector heads from a controlled seed",
            "optimization": "use a smaller learning rate for transferred backbone tensors and a larger head rate; validate with a no-transfer ablation",
            "claim_boundary": "movement statistics establish lineage and compatibility, not that transfer is optimal or improves task accuracy",
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="pretrained or parent checkpoint")
    parser.add_argument("--target", required=True, help="child/fine-tuned checkpoint")
    parser.add_argument("--out", required=True, help="JSON report output path")
    args = parser.parse_args()
    report = analyze(args.base, args.target)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
