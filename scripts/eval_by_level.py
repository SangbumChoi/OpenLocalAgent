#!/usr/bin/env python
"""Grounded accuracy of one checkpoint against every enrichment level.

The flywheel's published ~100% grounded figure is reported without the enrichment level it was
measured at, and level r controls how hard the held-out set is. This scores a single trained
checkpoint on levels 1..5 so the difficulty axis can be separated from the training-budget axis.

  python scripts/eval_by_level.py --ckpt results/runs/ablate/cuda-s0-x10/model.pt --out results/runs/eval_by_level.json
"""

from __future__ import annotations

import argparse
import json

import torch

from openlocalagent.agent.toolset import STANDARD_TOOLS as TOOLS
from openlocalagent.data.synth.agent_synth import Generator
from openlocalagent.eval.harness import evaluate, evaluate_grounded, multi_turn_eval
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-eval", type=int, default=30)
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = load_tokenizer("byte")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["cfg"])
    model = LocalAgentLM(cfg).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    from openlocalagent.agent.pointer_head import PointerHead
    from openlocalagent.agent.tool_head import ToolHead

    head = ptr = None
    if state.get("tool_head") is not None:
        head = ToolHead(cfg.d_model).to(device)
        head.load_state_dict(state["tool_head"])
        head.eval()
    if state.get("ptr_head") is not None:
        ptr = PointerHead(cfg.d_model).to(device)
        ptr.load_state_dict(state["ptr_head"])
        ptr.eval()

    report = {"checkpoint": args.ckpt, "levels": []}
    for level in range(1, 6):
        held = Generator(level=level, seed=1000 + level, split="eval").generate_balanced(args.n_eval)
        grounded = evaluate_grounded(model, held, tok, TOOLS, device=device, tool_head=head,
                                     ptr_head=ptr)
        free = evaluate(model, held, tok, device=device)
        episodes = Generator(level=level, seed=6000 + level, split="eval").episodes(30)
        turns = multi_turn_eval(model, episodes, tok, TOOLS, device=device, tool_head=head,
                                ptr_head=ptr)
        print(f"level {level}: grounded={grounded['overall']*100:.1f}% "
              f"free-gen={free['overall']*100:.1f}% mt_step={turns['step_acc']*100:.1f}%", flush=True)
        report["levels"].append({"level": level, "grounded": grounded, "free_gen": free,
                                 "multi_turn": turns, "n": len(held)})
        json.dump(report, open(args.out, "w"), indent=2)
    print("EVAL_BY_LEVEL_DONE", flush=True)


if __name__ == "__main__":
    main()
