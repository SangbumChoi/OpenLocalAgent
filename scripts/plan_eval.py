#!/usr/bin/env python
"""Run the stage-3 PLANNER eval on a trained checkpoint.

Loads a checkpoint (`runs/.../model.pt` with `cfg`/`state_dict`/`tool_head`/`ptr_head` — same
bundle written by scripts/flywheel.py & analyze_loop.py), rebuilds the model + tool/pointer heads,
generates held-out plan episodes via `Generator(2, 5001, 'eval').plan_episodes(n)`, runs the
learned `plan_rollout` through `eval.harness.plan_eval`, and prints the report.

Usage:
  python scripts/plan_eval.py --checkpoint runs/analyze_ultra-tiny-1m/model.pt [--n 64] [--max-steps 4]
"""

from __future__ import annotations

import argparse

import torch

from localagent.agent.pointer_head import PointerHead
from localagent.agent.tool_head import ToolHead
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.eval.harness import format_plan_eval, plan_eval
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import resolve_device


def _load(checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"] if isinstance(ck["cfg"], dict) else ck["cfg"].__dict__
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    model = LocalAgentLM(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tool_head = ptr_head = None
    if ck.get("tool_head"):
        tool_head = ToolHead(cfg.d_model).to(device)
        tool_head.load_state_dict(ck["tool_head"])
        tool_head.eval()
    if ck.get("ptr_head"):
        ptr_head = PointerHead(cfg.d_model).to(device)
        ptr_head.load_state_dict(ck["ptr_head"])
        ptr_head.eval()
    return model, tool_head, ptr_head, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/analyze_ultra-tiny-1m/model.pt")
    ap.add_argument("--n", type=int, default=64, help="held-out plan episodes to eval")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    tok = load_tokenizer("byte")
    model, tool_head, ptr_head, cfg = _load(args.checkpoint, device)
    print(f"loaded {cfg.name}: {model.num_params()/1e6:.3f}M params on {device}"
          f" (tool_head={tool_head is not None}, ptr_head={ptr_head is not None})", flush=True)

    # held-out (disjoint-slot) eval split, fixed seed per the stage-3 contract
    episodes = Generator(level=2, seed=5001, split="eval").plan_episodes(args.n)
    res = plan_eval(model, tok, TOOLS, episodes, tool_head=tool_head, ptr_head=ptr_head,
                    max_steps=args.max_steps, device=device)
    print(format_plan_eval(res), flush=True)


if __name__ == "__main__":
    main()
