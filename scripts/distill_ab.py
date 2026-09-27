#!/usr/bin/env python
"""Controlled A/B: does 30M -> 1M Top-K knowledge distillation lift the 1M's tool-calling
above training the SAME 1M from scratch?

This isolates the DISTILLATION stage. Both arms share: the student init seed (one pretrained
1M init, deep-copied), the SFT samples/seed/steps, the episode set, and the eval sets. The ONLY
difference is whether a `distill(...)` stage runs between the pretrain-init and the head-SFT:

  CONTROL   : 1M -> pretrain-init -> joint head-SFT (fixed budget)                   -> eval
  TREATMENT : 1M -> pretrain-init -> distill(30M->1M, top-k, k=16) -> SAME head-SFT  -> eval

The 30M teacher is the frozen round-1 best (runs/tiny-30m-byte-best.pt, ~71% single-turn).
We report, per arm: single-turn overall, teacher-forced step/episode acc, grounded acc, and
free-rollout whole-plan acc, plus deltas and wall-clock. Baseline to beat (just-finished
from-scratch 1M, runs/analyze_ultra-tiny-1m round 4): single-turn 54.8% / next-tool 89%.

Usage:  python scripts/distill_ab.py [--round 1] [--quick]
        OMP_NUM_THREADS=4 python scripts/distill_ab.py     # throttle on CPU
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.data.render import build_pretrain_stream
from localagent.eval.harness import evaluate_grounded, plan_eval
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import resolve_device
from localagent.train.distill import distill
from localagent.train.pretrain import pretrain
from localagent.train.sft import sft

OUT = "runs/distill_ab"
STUDENT_CFG = "configs/model/ultra-tiny-1m.yaml"
TEACHER_CFG = "configs/model/tiny-30m-byte.yaml"
TEACHER_CKPT = "runs/tiny-30m-byte-best.pt"

# from-scratch 1M baseline (runs/analyze_ultra-tiny-1m, round 4) — what we must beat
BASELINE = {"single_turn": 0.5478, "tf_step_acc": 0.8889, "tf_episode_acc": 0.7442,
            "grounded": 0.7083, "whole_plan": 0.04}


def load_teacher(device):
    ck = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
    tcfg = ModelConfig.from_yaml(TEACHER_CFG)
    assert ck["cfg"] == tcfg.__dict__, "teacher ckpt cfg != tiny-30m-byte.yaml"
    teacher = LocalAgentLM(tcfg).to(device)
    teacher.load_state_dict(ck["state_dict"])
    teacher.eval()
    return teacher


def run_sft_and_eval(model, *, train, episodes, held, held_ep, tok, device, steps, batch,
                     mt_weight, log):
    """Identical joint head-SFT + eval for both arms. Returns (metrics, head, ptr)."""
    _, head, ptr = sft(model, train, tok, steps=steps, batch_size=batch, lr=1.5e-3,
                       device=device, log=log, joint_tool_head=True,
                       conversations=episodes, mt_weight=mt_weight)
    res = evaluate_grounded(model, held, tok, TOOLS, device=device, tool_head=head, ptr_head=ptr)
    pe = plan_eval(model, tok, TOOLS, held_ep, tool_head=head, ptr_head=ptr, device=device)
    metrics = {
        "single_turn": res["overall"],
        "tf_step_acc": pe["teacher_forced"]["step_acc"],
        "tf_episode_acc": pe["teacher_forced"]["episode_acc"],
        "grounded": pe["grounded_acc"],
        "whole_plan": pe["whole_plan_acc"],
        "plan_step_acc": pe["step_acc"],
        "categories": res["categories"],
    }
    return metrics, head, ptr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, help="eval/data round r (held seeds 1000+r/6000+r)")
    ap.add_argument("--seed", type=int, default=0, help="student init seed (shared by both arms)")
    ap.add_argument("--pre", type=int, default=200, help="pretrain-init steps (shared init)")
    ap.add_argument("--distill-steps", type=int, default=300)
    ap.add_argument("--sft-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mt-weight", type=float, default=1.5)
    ap.add_argument("--n-train", type=int, default=4000, help="single-turn samples (SFT + distill)")
    ap.add_argument("--n-ep", type=int, default=120, help="plan episodes for training")
    ap.add_argument("--n-eval", type=int, default=16, help="held-out per-category single-turn")
    ap.add_argument("--n-eval-ep", type=int, default=50, help="held-out plan episodes")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.pre, args.distill_steps, args.sft_steps = 40, 60, 80
        args.n_train, args.n_ep, args.n_eval, args.n_eval_ep = 800, 40, 8, 16

    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")
    tok = load_tokenizer("byte")
    r = args.round
    log = lambda *a: None  # noqa: E731

    print(f"device={device}  threads={torch.get_num_threads()}  round={r}", flush=True)

    # --- shared data (identical across arms) ---
    train = Generator(level=r, seed=r, split="train").generate(args.n_train)
    episodes = Generator(level=r, seed=5000 + r, split="train").plan_episodes(args.n_ep)
    held = Generator(level=r, seed=1000 + r, split="eval").generate_balanced(args.n_eval)
    held_ep = Generator(level=r, seed=6000 + r, split="eval").plan_episodes(args.n_eval_ep)
    print(f"data: {len(train)} single-turn, {len(episodes)} train-eps, "
          f"{len(held)} held single-turn, {len(held_ep)} held-eps", flush=True)

    # --- teacher (frozen round-1 best 30M) ---
    teacher = load_teacher(device)
    print(f"teacher tiny-30m-byte: {teacher.num_params()/1e6:.1f}M (frozen)", flush=True)

    # --- ONE shared 1M init: same seed, same pretrain stream, deep-copied for both arms ---
    scfg = ModelConfig.from_yaml(STUDENT_CFG)
    scfg.assert_within_budget()
    torch.manual_seed(args.seed)
    init = LocalAgentLM(scfg).to(device)
    print(f"student {scfg.name}: {init.num_params()/1e6:.3f}M — pretrain-init {args.pre} steps "
          f"(seed {args.seed})", flush=True)
    pretrain(init, build_pretrain_stream(train, tok), tok, steps=args.pre, batch_size=64,
             device=device, log=log)
    init_state = copy.deepcopy(init.state_dict())

    results = {}

    # ===== CONTROL: pretrain-init -> head-SFT (NO distill) =====
    print("\n=== CONTROL: from-scratch (pretrain-init -> head-SFT) ===", flush=True)
    t0 = time.time()
    ctrl = LocalAgentLM(scfg).to(device)
    ctrl.load_state_dict(copy.deepcopy(init_state))
    ctrl_metrics, ctrl_head, ctrl_ptr = run_sft_and_eval(
        ctrl, train=train, episodes=episodes, held=held, held_ep=held_ep, tok=tok, device=device,
        steps=args.sft_steps, batch=args.batch, mt_weight=args.mt_weight, log=log)
    ctrl_metrics["wall_s"] = time.time() - t0
    results["control"] = ctrl_metrics
    print(f"  single_turn={ctrl_metrics['single_turn']*100:.1f}%  "
          f"tf_step={ctrl_metrics['tf_step_acc']*100:.1f}%  grounded={ctrl_metrics['grounded']*100:.1f}%  "
          f"whole_plan={ctrl_metrics['whole_plan']*100:.1f}%  ({ctrl_metrics['wall_s']:.0f}s)", flush=True)

    # ===== TREATMENT: pretrain-init -> distill(30M->1M, top-k) -> SAME head-SFT =====
    print("\n=== TREATMENT: distilled (pretrain-init -> top-k distill -> head-SFT) ===", flush=True)
    t0 = time.time()
    treat = LocalAgentLM(scfg).to(device)
    treat.load_state_dict(copy.deepcopy(init_state))
    distill(treat, train, teacher, tok, kd_type="topk", kd_k=16, steps=args.distill_steps,
            temperature=2.0, kd_weight=1.0, ce_weight=0.2, lr=1.5e-3, device=device, log=print)
    distill_s = time.time() - t0
    treat_metrics, treat_head, treat_ptr = run_sft_and_eval(
        treat, train=train, episodes=episodes, held=held, held_ep=held_ep, tok=tok, device=device,
        steps=args.sft_steps, batch=args.batch, mt_weight=args.mt_weight, log=log)
    treat_metrics["wall_s"] = time.time() - t0
    treat_metrics["distill_s"] = distill_s
    results["treatment"] = treat_metrics
    print(f"  single_turn={treat_metrics['single_turn']*100:.1f}%  "
          f"tf_step={treat_metrics['tf_step_acc']*100:.1f}%  grounded={treat_metrics['grounded']*100:.1f}%  "
          f"whole_plan={treat_metrics['whole_plan']*100:.1f}%  "
          f"(distill {distill_s:.0f}s, total {treat_metrics['wall_s']:.0f}s)", flush=True)

    # --- report ---
    keys = ["single_turn", "tf_step_acc", "tf_episode_acc", "grounded", "whole_plan"]
    labels = {"single_turn": "single-turn overall", "tf_step_acc": "TF next-tool step_acc",
              "tf_episode_acc": "TF episode_acc", "grounded": "grounded_acc",
              "whole_plan": "free-rollout whole-plan"}
    print("\n" + "=" * 78)
    print(f"{'metric':<26}{'baseline':>10}{'control':>10}{'treatment':>11}{'T-C delta':>11}")
    print("-" * 78)
    for k in keys:
        b = BASELINE.get(k)
        c = ctrl_metrics[k] * 100
        t = treat_metrics[k] * 100
        bs = f"{b*100:.1f}%" if b is not None else "--"
        print(f"{labels[k]:<26}{bs:>10}{c:>9.1f}%{t:>10.1f}%{(t - c):>+10.1f}")
    print("=" * 78)
    print(f"wall-clock: control {ctrl_metrics['wall_s']:.0f}s  |  "
          f"treatment {treat_metrics['wall_s']:.0f}s (distill {treat_metrics['distill_s']:.0f}s)")

    # --- save the better student (by single-turn, tie-break whole-plan) ---
    def score(m):
        return m["single_turn"] + 0.25 * m["whole_plan"]
    if score(treat_metrics) >= score(ctrl_metrics):
        best, head, ptr, arm = treat, treat_head, treat_ptr, "treatment"
    else:
        best, head, ptr, arm = ctrl, ctrl_head, ctrl_ptr, "control"
    torch.save({"cfg": scfg.__dict__, "state_dict": best.state_dict(),
                "tool_head": head.state_dict() if head is not None else None,
                "ptr_head": ptr.state_dict() if ptr is not None else None,
                "arm": arm},
               f"{OUT}/student_best.pt")
    results["baseline"] = BASELINE
    results["config"] = vars(args)
    results["best_arm"] = arm
    json.dump(results, open(f"{OUT}/result.json", "w"), indent=2)
    print(f"\nsaved better arm ({arm}) -> {OUT}/student_best.pt ; metrics -> {OUT}/result.json")


if __name__ == "__main__":
    main()
