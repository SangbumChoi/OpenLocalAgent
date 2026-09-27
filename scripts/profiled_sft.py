#!/usr/bin/env python
"""Train the student with per-module learning rates read off the open models' adaptation profile.

Eight open families, fine-tuned on the same corpus with the same recipe, do not spread their
update evenly: measured as ||dW||/||W||, the feed-forward down-projection moves about half as much
as everything else while the up and gate projections move most. That profile is a property of the
task rather than of any one checkpoint, so it can be handed to a differently-shaped student even
though the weights themselves cannot.

Each parameter group's learning rate is scaled by its role's measured share. The scaling is applied
inside `step()` and undone afterwards, so it survives the trainer's own scheduler writing a single
learning rate across all groups every iteration.

  python scripts/profiled_sft.py --profile runs/analysis/lora_profile.json \
      --config configs/train/sft-profiled-10m.yaml
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import torch

# Student parameter name -> the role the profile is keyed by.
ROLE_PATTERNS = (
    ("attn.in_proj", "attn_in"),
    ("attn.out_proj", "attn_out"),
    ("ffn.gate", "ffn_gate"),
    ("ffn.up", "ffn_up"),
    ("ffn.down", "ffn_down"),
)


def role_of(name: str) -> str | None:
    for needle, role in ROLE_PATTERNS:
        if needle in name:
            return role
    return None


def multipliers(profile_path: Path, floor: float, ceiling: float, invert: bool = False,
                power: float = 1.0) -> dict[str, float]:
    """Per-role learning-rate scale, normalised so the mean role is unchanged.

    `invert` flips the profile. Running both directions is what separates 'the profile carries no
    signal' from 'it carries signal I read backwards', which a single arm cannot distinguish.
    """
    summary = json.loads(profile_path.read_text())["summary"]["per_role"]
    mean = sum(summary.values()) / len(summary)
    ratios = {role: (value / mean) ** power for role, value in summary.items()}
    if invert:
        ratios = {role: 1.0 / value for role, value in ratios.items()}
    return {role: min(max(value, floor), ceiling) for role, value in ratios.items()}


def install(scales: dict[str, float]) -> None:
    from localagent.model import LocalAgentLM

    models: list[object] = []
    original_init = LocalAgentLM.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        models.append(self)

    LocalAgentLM.__init__ = recording_init
    base_adamw = torch.optim.AdamW

    class ProfiledAdamW(base_adamw):
        def __init__(self, params, lr=1e-3, **kwargs):
            params = list(params)
            names = {}
            for model in models:
                for name, parameter in model.named_parameters():
                    names[id(parameter)] = name
            buckets: dict[float, list] = {}
            for parameter in params:
                scale = scales.get(role_of(names.get(id(parameter), "")) or "", 1.0)
                buckets.setdefault(scale, []).append(parameter)
            groups = [{"params": items, "lr": lr, "lr_mult": scale}
                      for scale, items in sorted(buckets.items())]
            super().__init__(groups, lr=lr, **kwargs)
            counts = {f"{scale:.2f}x": len(items) for scale, items in sorted(buckets.items())}
            print(f"[profiled] parameter groups {counts}", flush=True)

        @torch.no_grad()
        def step(self, *args, **kwargs):
            # The trainer's scheduler writes one learning rate into every group, so the profile is
            # applied at the moment of the update and removed again immediately after.
            base = [group["lr"] for group in self.param_groups]
            for group, value in zip(self.param_groups, base):
                group["lr"] = value * group.get("lr_mult", 1.0)
            try:
                return super().step(*args, **kwargs)
            finally:
                for group, value in zip(self.param_groups, base):
                    group["lr"] = value

    torch.optim.AdamW = ProfiledAdamW


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="runs/analysis/lora_profile.json")
    ap.add_argument("--stage", default="sft")
    ap.add_argument("--config", required=True)
    ap.add_argument("--floor", type=float, default=0.25)
    ap.add_argument("--ceiling", type=float, default=2.0)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--power", type=float, default=1.0)
    args = ap.parse_args()

    scales = multipliers(Path(args.profile), args.floor, args.ceiling, args.invert, args.power)
    print(json.dumps({"learning_rate_scales": scales}, indent=2), flush=True)
    install(scales)

    sys.argv = ["localagent", "train", args.stage, args.config]
    runpy.run_module("localagent.cli", run_name="__main__")


if __name__ == "__main__":
    main()
