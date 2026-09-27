#!/usr/bin/env python
"""Where does fine-tuning actually change an open model?

Weights themselves do not survive a change of architecture, but the *shape of the update* might:
if every open family, fine-tuned on the same corpus with the same recipe, concentrates its change
in the same kinds of modules at the same relative depths, that profile is a property of the task
rather than of any one checkpoint — and it can be handed to a differently-shaped student.

For every LoRA adapter this computes the effective update dW = (alpha/r) * B @ A per module and
reports its Frobenius norm relative to the base weight it modifies, bucketed by projection type
and relative depth.

  python scripts/analyze_lora.py --out runs/analysis/lora_profile.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

ADAPTERS = {
    "smollm2-135m": "data/baselines/SmolLM2-135M-Instruct",
    "lfm2-350m": "data/baselines/LFM2-350M",
    "smollm2-360m": "data/baselines/SmolLM2-360M-Instruct",
    "danube3-500m": "data/baselines/h2o-danube3-500m-chat",
    "qwen25-05b": "data/baselines/Qwen2.5-0.5B-Instruct",
    "qwen25-coder-05b": "data/baselines/Qwen2.5-Coder-0.5B-Instruct",
    "lfm2-700m": "data/baselines/LFM2-700M",
    "qwen3-06b": "data/baselines/Qwen3-0.6B",
}
# Student-side roles, so the profile can be read straight onto our architecture.
ROLE = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "attn_out",
        "in_proj": "attn_in", "out_proj": "attn_out",
        "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down",
        "w1": "ffn_gate", "w2": "ffn_down", "w3": "ffn_up"}


def base_weights(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    weights: dict[str, torch.Tensor] = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(load_file(str(shard)))
    return weights


def adapter_pairs(adapter_dir: Path):
    """Yield (module_path, A, B) for every LoRA-adapted module."""
    from safetensors.torch import load_file

    files = list(adapter_dir.glob("adapter_model.safetensors"))
    if not files:
        return
    weights = load_file(str(files[0]))
    stems = {key.rsplit(".lora_", 1)[0] for key in weights if ".lora_" in key}
    for stem in sorted(stems):
        a = weights.get(f"{stem}.lora_A.weight")
        b = weights.get(f"{stem}.lora_B.weight")
        if a is not None and b is not None:
            yield stem, a.float(), b.float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-root", default="runs/lora")
    ap.add_argument("--out", default="runs/analysis/lora_profile.json")
    args = ap.parse_args()

    profile: dict[str, dict] = {}
    for tag, base_path in ADAPTERS.items():
        adapter_dir = Path(args.lora_root) / tag
        if not (adapter_dir / "adapter_model.safetensors").exists():
            continue
        config = json.loads((adapter_dir / "adapter_config.json").read_text())
        scale = config.get("lora_alpha", 32) / max(config.get("r", 16), 1)
        base = base_weights(Path(base_path))
        layers = [int(m.group(1)) for key in base
                  for m in [re.search(r"layers\.(\d+)\.", key)] if m]
        depth = max(layers) + 1 if layers else 1

        rows = []
        for stem, a, b in adapter_pairs(adapter_dir):
            update = (b @ a) * scale
            name = stem.split(".")[-1]
            role = ROLE.get(name)
            layer = re.search(r"layers\.(\d+)\.", stem)
            index = int(layer.group(1)) if layer else None
            # peft stores the module under base_model.model.<hf path>; recover the base tensor.
            key = stem.split("base_model.model.")[-1] + ".weight"
            reference = base.get(key)
            relative = (update.norm() / reference.float().norm()).item() if reference is not None \
                else None
            rows.append({"module": stem, "role": role, "layer": index,
                         "relative_depth": (index + 0.5) / depth if index is not None else None,
                         "update_norm": update.norm().item(), "relative_update": relative})
        profile[tag] = {"base": base_path, "depth": depth, "scale": scale, "modules": rows}
        touched = [r for r in rows if r["relative_update"] is not None]
        print(f"{tag:20s} depth={depth:3d} modules={len(rows):4d} matched={len(touched):4d} "
              f"mean rel update={sum(r['relative_update'] for r in touched)/max(len(touched),1):.4f}",
              flush=True)

    # Aggregate: mean relative update per role, and per role x depth-third.
    by_role: dict[str, list[float]] = defaultdict(list)
    by_role_depth: dict[str, list[float]] = defaultdict(list)
    for tag, entry in profile.items():
        for row in entry["modules"]:
            if row["relative_update"] is None or row["role"] is None:
                continue
            by_role[row["role"]].append(row["relative_update"])
            if row["relative_depth"] is not None:
                third = min(int(row["relative_depth"] * 3), 2)
                by_role_depth[f"{row['role']}|{['early','middle','late'][third]}"].append(
                    row["relative_update"])

    summary = {
        "per_role": {role: sum(v) / len(v) for role, v in sorted(by_role.items())},
        "per_role_depth": {key: sum(v) / len(v) for key, v in sorted(by_role_depth.items())},
        "models": len(profile),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "profile": profile}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("LORA_ANALYSIS_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
