#!/usr/bin/env python
"""Write a pretrain checkpoint that keeps only one region of the donor's weights.

The staged trainer takes its parent as a checkpoint file, so a region ablation is expressed by
handing midtrain a checkpoint whose non-copied tensors were re-initialised. Everything else about
the chain — the corpus, the contract, the schedule — is held fixed, so the downstream score
difference is attributable to which region survived.

  python scripts/region_catalog_init.py --region attn --donor runs/mix-10m-hybrid-seed2026/latest.pt \
      --out runs/region-init/attn/latest.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localagent.model import LocalAgentLM, ModelConfig


def region_keys(region: str, donor: dict[str, torch.Tensor], n_layers: int) -> list[str]:
    """Donor keys to copy for a named region. `blocks.N.` prefixes carry the layer index."""

    def layer_of(key: str) -> int | None:
        parts = key.split(".")
        return int(parts[1]) if parts[0] == "blocks" else None

    half = n_layers // 2
    rules = {
        "scratch": lambda key: False,
        "embed": lambda key: key.startswith("embed"),
        "norms": lambda key: "norm" in key,
        "attn": lambda key: ".attn." in key,
        "ffn": lambda key: ".ffn." in key,
        "early": lambda key: (index := layer_of(key)) is not None and index < half,
        "late": lambda key: (index := layer_of(key)) is not None and index >= half,
        "attn_embed": lambda key: ".attn." in key or key.startswith("embed"),
        "no_embed": lambda key: not key.startswith("embed"),
        "full": lambda key: True,
    }
    if region not in rules:
        raise ValueError(f"unknown region {region!r}")
    return [key for key in donor if rules[region](key)]


def build_model(cfg_payload: dict, seed: int) -> LocalAgentLM:
    torch.manual_seed(seed)
    fields = {key: value for key, value in cfg_payload.items()
              if key in ModelConfig.__dataclass_fields__}
    return LocalAgentLM(ModelConfig(**fields))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    checkpoint = torch.load(args.donor, map_location="cpu", weights_only=False)
    donor_state = checkpoint["state_dict"]
    fresh = build_model(checkpoint["cfg"], args.seed)
    state = {key: tensor.clone() for key, tensor in fresh.state_dict().items()}

    copied, copied_params = [], 0
    for key in region_keys(args.region, donor_state, fresh.cfg.n_layers):
        if key in state and state[key].shape == donor_state[key].shape:
            state[key] = donor_state[key].clone().to(state[key].dtype)
            copied.append(key)
            copied_params += state[key].numel()

    total = sum(tensor.numel() for tensor in state.values())
    checkpoint["state_dict"] = state
    # The optimizer moments belong to the donor's trajectory; keeping them would leak the very
    # weights this arm is meant to withhold.
    checkpoint["optimizer"] = None
    checkpoint["region_ablation"] = {
        "region": args.region, "donor": args.donor, "tensors_copied": len(copied),
        "params_copied": copied_params, "params_total": total,
        "fraction_copied": copied_params / total,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    print(json.dumps(checkpoint["region_ablation"], indent=2))
    print("REGION_INIT_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
