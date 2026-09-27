#!/usr/bin/env python3
"""Depth-prune a released checkpoint at identical width: keep a chosen subset of layers.

The campaign's transfer results endorse exactly one import mode — identical-shape adoption — so
the shrink keeps every retained layer byte-identical and only drops whole layers. Selection
policies test whether the fine-tuning update map has prescriptive value for pruning, which it
did not have for adaptation placement (§5).
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def chosen_layers(policy: str, total: int, keep: int, profile: list[float] | None) -> list[int]:
    if policy == "uniform":
        step = (total - 1) / (keep - 1)
        return sorted({round(i * step) for i in range(keep)})
    if policy == "first":
        return list(range(keep))
    if policy in ("least_written", "most_written"):
        if profile is None:
            raise SystemExit("policy needs --profile json (per-layer update magnitudes)")
        order = sorted(range(total), key=lambda i: profile[i], reverse=(policy == "most_written"))
        return sorted(order[:keep])
    raise SystemExit(f"unknown policy {policy}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--policy", default="uniform",
                    choices=["uniform", "first", "least_written", "most_written"])
    ap.add_argument("--profile", default="", help="json list of per-layer update magnitudes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    total = model.config.num_hidden_layers
    profile = json.loads(Path(args.profile).read_text()) if args.profile else None
    kept = chosen_layers(args.policy, total, args.keep, profile)
    print(f"keeping layers {kept} of {total} ({args.policy})")

    layers = model.model.layers
    new_layers = torch.nn.ModuleList([layers[i] for i in kept])
    model.model.layers = new_layers
    # Any per-layer list in the config (layer_types on Qwen3, sliding-window patterns on
    # others) must be sliced to the kept layers or strict config validation rejects the save.
    for key, value in list(vars(model.config).items()):
        if isinstance(value, list) and len(value) == total:
            setattr(model.config, key, [value[i] for i in kept])
    model.config.num_hidden_layers = len(kept)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "PRUNE_MANIFEST.json").write_text(json.dumps(
        {"base": args.base, "policy": args.policy, "kept_layers": kept, "of": total}, indent=1))
    parameters = sum(p.numel() for p in model.parameters())
    print(f"saved {out} params={parameters/1e6:.0f}M")


if __name__ == "__main__":
    main()
