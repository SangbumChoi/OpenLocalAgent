#!/usr/bin/env python3
"""Per-layer mean ||dW||/||W|| from a LoRA adapter, as a plain json list for prune_depth."""
import json, re, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

adapter_dir, base_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
weights = load_file(str(next(Path(adapter_dir).glob("adapter_model.safetensors"))))
base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
named = dict(base.named_parameters())
cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]
per_layer = {}
for key in [k for k in weights if k.endswith("lora_A.weight")]:
    bkey = key.replace("lora_A.weight", "lora_B.weight")
    match = re.search(r"layers\.(\d+)\.", key)
    if not match: continue
    layer = int(match.group(1))
    delta = scale * (weights[bkey].float() @ weights[key].float())
    wname = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
    w = named.get(wname)
    if w is None: continue
    ratio = (delta.norm() / w.norm()).item()
    per_layer.setdefault(layer, []).append(ratio)
n = max(per_layer) + 1
profile = [sum(per_layer.get(i, [0])) / max(1, len(per_layer.get(i, [1]))) for i in range(n)]
Path(out).write_text(json.dumps(profile))
print("layers:", n, "min/max:", min(profile), max(profile))
