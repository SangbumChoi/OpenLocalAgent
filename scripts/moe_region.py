#!/usr/bin/env python
"""Region transfer inside a sparse mixture-of-experts model.

A routed FFN splits into parts a dense model does not have: the router that decides, the individual
experts that compute, and the dense trunk around them. This copies one part from a pretrained MoE
donor into a fresh student and continues pretraining on the same corpus, so the held-out loss says
what each part was carrying.

  python scripts/moe_region.py --region experts --steps 300 --out runs/moe-region/experts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from localagent.model import LocalAgentLM, ModelConfig
from localagent.data.pretrain_corpus import PackedShardDataset
from localagent.train.pretrain import pretrain

DONOR = Path("runs/moe-donor-seed2026/latest.pt")
SHARDS = "data/shards/h100-mix"

RULES = {
    "scratch": lambda key: False,
    "router": lambda key: ".ffn.router." in key,
    "expert0": lambda key: ".ffn.experts.0." in key,
    "experts": lambda key: ".ffn.experts." in key,
    "router_expert0": lambda key: ".ffn.router." in key or ".ffn.experts.0." in key,
    "attn": lambda key: ".attn." in key,
    "trunk": lambda key: ".ffn." not in key,
    "full": lambda key: True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=sorted(RULES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--donor", default=str(DONOR))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    donor_payload = torch.load(args.donor, map_location="cpu", weights_only=False)
    donor = donor_payload.get("model") or donor_payload.get("state_dict") or donor_payload
    cfg = ModelConfig.from_yaml("configs/model/webgpu-44m-moe.yaml")
    model = LocalAgentLM(cfg).to(args.device)

    state = model.state_dict()
    rule = RULES[args.region]
    copied = 0
    for key in list(state):
        if rule(key) and key in donor and donor[key].shape == state[key].shape:
            state[key] = donor[key].to(state[key].dtype).clone()
            copied += state[key].numel()
    model.load_state_dict(state)
    total = sum(t.numel() for t in state.values())
    print(f"[{args.region}] copied {copied:,}/{total:,} parameters "
          f"({copied/total*100:.1f}%)", flush=True)

    data = PackedShardDataset(SHARDS, "train")
    validation = PackedShardDataset(SHARDS, "val")
    started = time.time()
    losses = pretrain(model, data, None, steps=args.steps, batch_size=8, accum_steps=8,
                      lr=3e-4, device=args.device, seed=args.seed, val_data=validation,
                      eval_every=max(50, args.steps // 4), eval_batches=32,
                      log=lambda *a: None)
    with torch.no_grad():
        model.eval()
        total_loss, batches = 0.0, 32
        for _ in range(batches):
            x, y = validation.sample_batch(8, __import__("random").Random(12345), args.device)
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            total_loss += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100).item()
    report = {"region": args.region, "params_copied": copied, "params_total": total,
              "fraction_copied": copied / total, "steps": args.steps,
              "train_loss_last": losses[-1] if losses else None,
              "held_out_loss": total_loss / batches, "seconds": round(time.time() - started, 1)}
    print(json.dumps(report), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("MOE_REGION_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
