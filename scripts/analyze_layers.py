#!/usr/bin/env python
"""Where, by depth, does agent fine-tuning move each open model?

The per-role summary already showed the update concentrating unevenly across module roles; this
resolves the other axis. For every adapter, per (layer, role): ‖(α/r)·B@A‖ / ‖W_base‖, plus a
per-layer mean and the same curve on normalised depth so models of different depths overlay.
If the curve's shape agrees across architectures (GQA transformers, LFM2 conv hybrids, Granite
mamba hybrid), the concentration is a property of the task, not of any one architecture — which is
the paper's data-attribution hypothesis stated at the weight level.

  python scripts/analyze_layers.py --out results/runs/analysis/layer_profiles.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

LAYER_OF = re.compile(r"\.(?:layers|blocks|h)\.(\d+)\.")
ADAPTERS = Path("results/runs/lora")
BASELINES = Path("data/baselines")
# adapter tag -> baseline directory, mirrored from the fine-tuning campaign.
BASE_DIR = {
    "smollm2-135m": "SmolLM2-135M-Instruct", "lfm25-230m": "LFM2.5-230M",
    "lfm2-350m": "LFM2-350M", "granite-h-350m": "granite-4.0-h-350m",
    "granite-350m": "granite-4.0-350m", "smollm2-360m": "SmolLM2-360M-Instruct",
    "danube3-500m": "h2o-danube3-500m-chat", "qwen25-05b": "Qwen2.5-0.5B-Instruct",
    "qwen25-coder-05b": "Qwen2.5-Coder-0.5B-Instruct", "lfm2-700m": "LFM2-700M",
    "qwen3-06b": "Qwen3-0.6B", "qwen3-17b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B",
}


def base_weights(directory: Path) -> dict[str, tuple[Path, str]]:
    """Key -> (shard, key): an index so only the needed tensors are materialised."""
    from safetensors import safe_open

    index: dict[str, tuple[Path, str]] = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                index[key] = (shard, key)
    return index


def read_tensor(index: dict[str, tuple[Path, str]], key: str) -> torch.Tensor | None:
    from safetensors import safe_open

    hit = index.get(key)
    if hit is None:
        return None
    shard, name = hit
    with safe_open(str(shard), framework="pt") as handle:
        return handle.get_tensor(name)


def profile(tag: str) -> dict | None:
    adapter_dir = ADAPTERS / tag
    weights_file = adapter_dir / "adapter_model.safetensors"
    config_file = adapter_dir / "adapter_config.json"
    base = BASE_DIR.get(tag)
    if not weights_file.exists() or base is None:
        return None
    config = json.loads(config_file.read_text())
    scale = config["lora_alpha"] / config["r"]

    from safetensors.torch import load_file

    adapter = load_file(str(weights_file))
    index = base_weights(BASELINES / base)

    cells: list[dict] = []
    for key, a_tensor in adapter.items():
        if ".lora_A." not in key:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        b_tensor = adapter.get(b_key)
        found = LAYER_OF.search(key)
        if b_tensor is None or found is None:
            continue
        layer = int(found.group(1))
        # base key: strip peft's wrapper prefix and the lora suffix.
        base_key = key.split(".lora_A.")[0].removeprefix("base_model.model.") + ".weight"
        w_base = read_tensor(index, base_key)
        if w_base is None:
            continue
        delta = (b_tensor.float() @ a_tensor.float()) * scale
        ratio = float(delta.norm() / w_base.float().norm().clamp_min(1e-12))
        role = base_key.rsplit(".", 2)[-2]
        cells.append({"layer": layer, "role": role, "ratio": ratio})

    if not cells:
        return None
    n_layers = max(cell["layer"] for cell in cells) + 1
    per_layer = []
    for layer in range(n_layers):
        ratios = [cell["ratio"] for cell in cells if cell["layer"] == layer]
        per_layer.append(sum(ratios) / len(ratios) if ratios else 0.0)
    return {"tag": tag, "n_layers": n_layers, "cells": cells, "per_layer_mean": per_layer,
            "normalized_depth": [(layer + 0.5) / n_layers for layer in range(n_layers)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/runs/analysis/layer_profiles.json")
    ap.add_argument("--tags", default="", help="comma list; empty profiles every known adapter")
    args = ap.parse_args()

    tags = [t for t in args.tags.split(",") if t] or list(BASE_DIR)
    results = {}
    for tag in tags:
        got = profile(tag)
        if got is None:
            print(f"{tag:18s} skipped", flush=True)
            continue
        results[tag] = got
        curve = got["per_layer_mean"]
        top = sorted(range(len(curve)), key=lambda i: -curve[i])[:max(1, len(curve) // 3)]
        print(f"{tag:18s} layers={got['n_layers']:2d} "
              f"peak_third={sorted(top)} "
              f"first/mid/last-third means="
              f"{sum(curve[:len(curve)//3])/max(1,len(curve)//3)*1e3:.1f}/"
              f"{sum(curve[len(curve)//3:2*len(curve)//3])/max(1,len(curve)//3)*1e3:.1f}/"
              f"{sum(curve[2*len(curve)//3:])/max(1,len(curve)-2*(len(curve)//3))*1e3:.1f} x1e-3",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    print("LAYERS_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
