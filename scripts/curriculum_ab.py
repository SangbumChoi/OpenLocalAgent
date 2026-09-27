#!/usr/bin/env python
"""Controlled 2-arm A/B: does LFM2-style curriculum ordering (easy->hard) of the SFT data help
single-turn tool-calling accuracy at a FIXED training budget, or is it within noise?

LFM2 (and the ARCHITECTURE_DEBATE axis-7 note) order pretraining data by empirical success
probability — easy samples first, hard later. We port the *principle* to tool-calling SFT: instead
of i.i.d. shuffling every step, the curriculum arm walks the samples in difficulty order (a
transparent score = parallel-call count + extra-arg count + has-arg + abstention + prompt-length;
see ``agent_synth.curriculum_order`` / ``difficulty_score``).

Both arms share, byte-for-byte: the ONE pretrained 1M init (same seed, deep-copied), the SAME SFT
sample SET, the same steps / batch / lr / episodes, and the same eval sets. The ONLY difference is
sample ORDERING during head-SFT:

  C (control)    : sft(..., shuffle=True)  on the same set            -> i.i.d. random each step
  T (curriculum) : sft(..., shuffle=False) on curriculum_order(set)   -> easy->hard, no reshuffle

Because plain SFT reshuffles every step, the control is matched by running it i.i.d. on the SAME
set; the treatment runs ordered passes (shuffle=False) over the difficulty-sorted set so the
curriculum is actually realized. We report single-turn overall and teacher-forced next-tool
step_acc for both arms, plus the delta.

Usage:  OMP_NUM_THREADS=4 python scripts/curriculum_ab.py [--quick]

MEMORY/TIME: this sandbox SIGKILLs big batches. Defaults use batch_size=8 and a small pretrain-init
batch; do NOT raise the pretrain batch to 64. Keep steps bounded so the A/B finishes well under
~30 min.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator, curriculum_order, difficulty_score
from localagent.data.render import build_pretrain_stream
from localagent.eval.harness import evaluate_grounded, plan_eval
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import resolve_device
from localagent.train.pretrain import pretrain
from localagent.train.sft import sft

OUT = "runs/curriculum_ab"
STUDENT_CFG = "configs/model/ultra-tiny-1m.yaml"


def eval_arm(model, head, ptr, *, held, held_ep, tok, device):
    res = evaluate_grounded(model, held, tok, TOOLS, device=device, tool_head=head, ptr_head=ptr)
    pe = plan_eval(model, tok, TOOLS, held_ep, tool_head=head, ptr_head=ptr, device=device)
    return {
        "single_turn": res["overall"],
        "tf_step_acc": pe["teacher_forced"]["step_acc"],
        "tf_episode_acc": pe["teacher_forced"]["episode_acc"],
        "grounded": pe["grounded_acc"],
        "whole_plan": pe["whole_plan_acc"],
        "categories": res["categories"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="student init seed (shared by both arms)")
    ap.add_argument("--pre", type=int, default=200, help="pretrain-init steps (shared init)")
    ap.add_argument("--sft-steps", type=int, default=400, help="head-SFT steps (both arms)")
    ap.add_argument("--batch", type=int, default=8, help="SFT batch (keep small: OOM)")
    ap.add_argument("--pre-batch", type=int, default=8, help="pretrain-init batch (DO NOT raise)")
    ap.add_argument("--mt-weight", type=float, default=1.5)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-ep", type=int, default=120)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--n-eval-ep", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.pre, args.sft_steps = 30, 60
        args.n_train, args.n_ep, args.n_eval, args.n_eval_ep = 600, 30, 8, 12

    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")
    tok = load_tokenizer("byte")
    r = args.round
    log = lambda *a: None  # noqa: E731

    print(f"device={device}  threads={torch.get_num_threads()}  round={r}", flush=True)

    # --- shared data (identical SET across both arms; only the ORDER differs) ---
    train = Generator(level=r, seed=r, split="train").generate(args.n_train)
    train_curr = curriculum_order(train)            # easy->hard view of the SAME set
    episodes = Generator(level=r, seed=5000 + r, split="train").plan_episodes(args.n_ep)
    held = Generator(level=r, seed=1000 + r, split="eval").generate_balanced(args.n_eval)
    held_ep = Generator(level=r, seed=6000 + r, split="eval").plan_episodes(args.n_eval_ep)
    sc = [difficulty_score(s) for s in train_curr]
    print(f"data: {len(train)} single-turn (curriculum score {sc[0]:.2f}..{sc[-1]:.2f}), "
          f"{len(episodes)} train-eps, {len(held)} held single-turn, {len(held_ep)} held-eps",
          flush=True)

    # --- ONE shared 1M init: same seed + pretrain stream, deep-copied for both arms ---
    scfg = ModelConfig.from_yaml(STUDENT_CFG)
    scfg.assert_within_budget()
    torch.manual_seed(args.seed)
    init = LocalAgentLM(scfg).to(device)
    print(f"student {scfg.name}: {init.num_params()/1e6:.3f}M — pretrain-init {args.pre} steps "
          f"(seed {args.seed}, batch {args.pre_batch})", flush=True)
    pretrain(init, build_pretrain_stream(train, tok), tok, steps=args.pre,
             batch_size=args.pre_batch, device=device, log=log)
    init_state = copy.deepcopy(init.state_dict())

    def fresh():
        m = LocalAgentLM(scfg).to(device)
        m.load_state_dict(copy.deepcopy(init_state))
        return m

    results = {}

    def report_arm(name, m):
        print(f"  {name}: single_turn={m['single_turn']*100:.1f}%  "
              f"tf_step={m['tf_step_acc']*100:.1f}%  grounded={m['grounded']*100:.1f}%  "
              f"whole_plan={m['whole_plan']*100:.1f}%  ({m['wall_s']:.0f}s)", flush=True)

    # ===== C: control — i.i.d. shuffled head-SFT on the set =====
    print("\n=== C (control): head-SFT, i.i.d. shuffle (shuffle=True) ===", flush=True)
    t0 = time.time()
    cm = fresh()
    _, ch, cp = sft(cm, train, tok, steps=args.sft_steps, batch_size=args.batch, lr=1.5e-3,
                    device=device, log=log, joint_tool_head=True, conversations=episodes,
                    mt_weight=args.mt_weight, shuffle=True)
    mC = eval_arm(cm, ch, cp, held=held, held_ep=held_ep, tok=tok, device=device)
    mC["wall_s"] = time.time() - t0
    results["control_shuffle"] = mC
    report_arm("C", mC)

    # ===== T: curriculum — ordered easy->hard passes (shuffle=False) on the SAME set =====
    print("\n=== T (curriculum): head-SFT, easy->hard ordered passes (shuffle=False) ===",
          flush=True)
    t0 = time.time()
    tm = fresh()
    _, th, tp = sft(tm, train_curr, tok, steps=args.sft_steps, batch_size=args.batch, lr=1.5e-3,
                    device=device, log=log, joint_tool_head=True, conversations=episodes,
                    mt_weight=args.mt_weight, shuffle=False)
    mT = eval_arm(tm, th, tp, held=held, held_ep=held_ep, tok=tok, device=device)
    mT["wall_s"] = time.time() - t0
    results["curriculum_ordered"] = mT
    report_arm("T", mT)

    # --- report table ---
    keys = ["single_turn", "tf_step_acc", "tf_episode_acc", "grounded", "whole_plan"]
    labels = {"single_turn": "single-turn overall", "tf_step_acc": "TF next-tool step_acc",
              "tf_episode_acc": "TF episode_acc", "grounded": "grounded_acc",
              "whole_plan": "free-rollout whole-plan"}
    print("\n" + "=" * 70)
    print(f"{'metric':<26}{'C shuffle':>12}{'T curric':>12}{'T-C':>12}")
    print("-" * 70)
    for k in keys:
        c, t = mC[k] * 100, mT[k] * 100
        print(f"{labels[k]:<26}{c:>11.1f}%{t:>11.1f}%{t - c:>+12.1f}")
    print("=" * 70)
    print(f"wall-clock: C {mC['wall_s']:.0f}s | T {mT['wall_s']:.0f}s")

    # --- verdict: does curriculum ordering help single-turn at fixed budget? ---
    st_delta = (mT["single_turn"] - mC["single_turn"]) * 100
    step_delta = (mT["tf_step_acc"] - mC["tf_step_acc"]) * 100
    if st_delta > 2.0:
        verdict = f"curriculum HELPS single-turn (+{st_delta:.1f})"
    elif st_delta < -2.0:
        verdict = f"curriculum HURTS single-turn ({st_delta:.1f})"
    else:
        verdict = f"curriculum WITHIN NOISE on single-turn ({st_delta:+.1f})"
    print(f"\nVERDICT: {verdict}")
    print(f"  single_turn T-C: {st_delta:+.1f} | tf_step T-C: {step_delta:+.1f}")

    results["config"] = vars(args)
    results["verdict"] = verdict
    json.dump(results, open(f"{OUT}/result.json", "w"), indent=2)
    print(f"\nmetrics -> {OUT}/result.json")


if __name__ == "__main__":
    main()
