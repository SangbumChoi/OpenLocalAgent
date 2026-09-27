#!/usr/bin/env python
"""Ablation harness around scripts/flywheel.py: same loop, three knobs.

Isolates why a rerun of the committed flywheel does not land on the accuracy trajectory recorded
in docs/EXPERIMENTS.md: device/precision (--device), RNG seed (--seed), or training budget
(--sft-scale). Everything else — generators, eval sets, LR, batch — is copied from flywheel.py so
arms stay comparable.

  python scripts/ablate_flywheel.py --device cuda --seed 0 --rounds 5 --out runs/ablate/cuda-s0
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.data.render import build_pretrain_stream
from localagent.eval.harness import evaluate, evaluate_grounded, multi_turn_eval
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.pretrain import pretrain
from localagent.train.rl import grpo
from localagent.train.sft import sft


def fmt(d):
    return f"overall={d['overall']*100:.1f}%  " + " ".join(
        f"{k}={v*100:.0f}%" for k, v in sorted(d["groups"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sft-scale", type=float, default=1.0,
                    help="multiplier on pretrain/SFT step counts (training-budget arm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--freegen", action="store_true", help="also run the slow free-gen comparison")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    tok = load_tokenizer("byte")
    cfg = ModelConfig.from_yaml("configs/model/ultra-tiny-1m.yaml")
    model = LocalAgentLM(cfg).to(device)
    print(f"model {cfg.name}: {model.num_params()/1e6:.3f}M params on {device} "
          f"seed={args.seed} sft_scale={args.sft_scale}", flush=True)

    n_train, n_eval, n_ep = 2500, 30, 120
    scale = args.sft_scale
    pre_steps, sft1, sft_inc, grpo_steps = int(200 * scale), int(380 * scale), int(130 * scale), 4

    g0 = Generator(level=1, seed=0, split="train").generate(n_train)
    pre_loss = pretrain(model, build_pretrain_stream(g0, tok), tok, steps=pre_steps,
                        batch_size=64, device=device, seed=args.seed)

    metrics = {"config": vars(args), "rounds": [], "pretrain_loss": pre_loss}
    for r in range(1, args.rounds + 1):
        train = Generator(level=r, seed=r, split="train").generate(n_train)
        episodes = Generator(level=r, seed=5000 + r, split="train").episodes(n_ep)
        held = Generator(level=r, seed=1000 + r, split="eval").generate_balanced(n_eval)
        held_ep = Generator(level=r, seed=6000 + r, split="eval").episodes(n_ep // 4)
        steps = sft1 if r == 1 else sft_inc
        sft_loss, head, ptr = sft(model, train, tok, steps=steps, batch_size=32, lr=1.5e-3,
                                  device=device, log=lambda *a: None, joint_tool_head=True,
                                  conversations=episodes, seed=args.seed)
        grpo(model, train, tok, steps=grpo_steps, device=device, log=lambda *a: None)
        gr = evaluate_grounded(model, held, tok, TOOLS, device=device, tool_head=head)
        mt = multi_turn_eval(model, held_ep, tok, TOOLS, device=device, tool_head=head, ptr_head=ptr)
        print(f"=== Round {r}: {fmt(gr)}", flush=True)
        print(f"    multi-turn: step_acc={mt['step_acc']*100:.0f}% "
              f"episode_acc={mt['episode_acc']*100:.0f}% ({mt['steps']} steps)", flush=True)
        metrics["rounds"].append({"round": r, "grounded": gr, "multi_turn": mt,
                                  "sft_loss_last": sft_loss[-1]})
        json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=2)

    if args.freegen:
        final_held = Generator(level=args.rounds, seed=4242, split="eval").generate_balanced(20)
        metrics["final_freegen"] = evaluate(model, final_held, tok, device=device)
        metrics["final_grounded"] = evaluate_grounded(model, final_held, tok, TOOLS, device=device,
                                                      tool_head=head, ptr_head=ptr)
        print(f"FINAL free-gen : {fmt(metrics['final_freegen'])}", flush=True)
        print(f"FINAL grounded : {fmt(metrics['final_grounded'])}", flush=True)

    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
                "tool_head": head.state_dict() if head is not None else None,
                "ptr_head": ptr.state_dict() if ptr is not None else None},
               f"{args.out}/model.pt")
    json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=2)
    print("ABLATION_DONE " + args.out, flush=True)


if __name__ == "__main__":
    main()
