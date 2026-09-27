#!/usr/bin/env python
"""Controlled A/B: does the Warmup-Stable-Decay (WSD) LR schedule (MiniCPM 2404.06395) lower
final SFT loss / lift agent accuracy at a FIXED step budget vs the current cosine schedule, and
does injecting a cleaner curated sample pool in the decay window help further?

Three arms share, byte-for-byte: the one pretrained 1M init (same seed, deep-copied), the SFT
samples / seed / steps / batch, the episode set, and the eval sets. The ONLY difference is the LR
schedule (and, for T2, the decay-window data pool):

  C  (control) : init -> head-SFT, cosine LR (warmup + cosine decay to 0.1*peak)        -> eval
  T1 (wsd)     : init -> head-SFT, WSD LR (warmup -> stable plateau -> exp 0.5^((s-S)/T))-> eval
  T2 (wsd+inj) : init -> head-SFT, WSD LR AND decay_samples=<cleanest subset> in the decay window

The "cleanest subset" for T2 is the level-1 (easiest/most canonical) slice of a fresh generator
draw — a stand-in for curated/high-quality data. All arms train on the SAME `train` pool in the
stable phase; T2 only swaps the LM rows DURING the decay window.

We report, per arm: final train loss (mean of last 10 steps), single-turn overall, teacher-forced
next-tool step_acc and episode_acc, grounded_acc, and free-rollout whole-plan acc, plus deltas.

Usage:  OMP_NUM_THREADS=4 python scripts/wsd_ab.py [--quick]

MEMORY/TIME: this sandbox SIGKILLs big batches. Defaults use batch_size=8 and a small pretrain
init batch; do NOT raise pretrain batch to 64. Keep steps bounded; the A/B finishes well under
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
from localagent.data.agent_synth import Generator
from localagent.data.render import build_pretrain_stream
from localagent.eval.harness import evaluate_grounded, plan_eval
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import resolve_device
from localagent.train.pretrain import pretrain
from localagent.train.sft import sft

OUT = "runs/wsd_ab"
STUDENT_CFG = "configs/model/ultra-tiny-1m.yaml"


def eval_arm(model, head, ptr, *, held, held_ep, tok, device, heads):
    """Single-turn overall is always computed (works head-less, via greedy LM rollout). The
    teacher-forced / grounded / whole-plan plan metrics require the trained heads, so they are
    only computed (and reported) when `heads` is True; otherwise they're returned as None."""
    res = evaluate_grounded(model, held, tok, TOOLS, device=device, tool_head=head, ptr_head=ptr)
    out = {"single_turn": res["overall"], "categories": res["categories"]}
    if heads:
        pe = plan_eval(model, tok, TOOLS, held_ep, tool_head=head, ptr_head=ptr, device=device)
        out.update(tf_step_acc=pe["teacher_forced"]["step_acc"],
                   tf_episode_acc=pe["teacher_forced"]["episode_acc"],
                   grounded=pe["grounded_acc"], whole_plan=pe["whole_plan_acc"],
                   plan_step_acc=pe["step_acc"])
    else:
        out.update(tf_step_acc=None, tf_episode_acc=None, grounded=None, whole_plan=None,
                   plan_step_acc=None)
    return out


def final_loss(hist, k=10):
    """Mean of the last k step losses — a low-variance read of where training landed."""
    tail = hist[-k:] if len(hist) >= k else hist
    return sum(tail) / max(1, len(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="student init seed (shared by all arms)")
    ap.add_argument("--pre", type=int, default=200, help="pretrain-init steps (shared init)")
    ap.add_argument("--sft-steps", type=int, default=400, help="head-SFT steps (all arms)")
    ap.add_argument("--batch", type=int, default=8, help="SFT batch (keep small: OOM)")
    ap.add_argument("--pre-batch", type=int, default=8, help="pretrain-init batch (DO NOT raise)")
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--decay-frac", type=float, default=0.2, help="WSD decay-window fraction")
    ap.add_argument("--mt-weight", type=float, default=1.5)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-decay", type=int, default=1500, help="T2 curated decay-window pool size")
    ap.add_argument("--n-ep", type=int, default=120)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--n-eval-ep", type=int, default=50)
    ap.add_argument("--heads", action="store_true",
                    help="train joint tool/pointer heads + report teacher-forced/grounded/plan "
                         "metrics. ~10x slower per step on CPU; off => LM-only (loss + single-turn "
                         "overall), which isolates the WSD schedule's effect cheaply.")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.pre, args.sft_steps = 30, 80
        args.n_train, args.n_decay = 600, 250
        args.n_ep, args.n_eval, args.n_eval_ep = 30, 8, 12
        args.warmup = 8

    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")
    tok = load_tokenizer("byte")
    r = args.round
    log = lambda *a: None  # noqa: E731

    print(f"device={device}  threads={torch.get_num_threads()}  round={r}", flush=True)

    # --- shared data (identical across all arms) ---
    train = Generator(level=r, seed=r, split="train").generate(args.n_train)
    # T2 curated decay pool: a fresh LEVEL-1 (easiest/most canonical) draw — a stand-in for the
    # "cleanest data" you'd reserve for the decay window. Distinct seed so it's held-out from train.
    decay_pool = Generator(level=1, seed=9000 + r, split="train").generate_balanced(args.n_decay)
    episodes = Generator(level=r, seed=5000 + r, split="train").plan_episodes(args.n_ep)
    held = Generator(level=r, seed=1000 + r, split="eval").generate_balanced(args.n_eval)
    held_ep = Generator(level=r, seed=6000 + r, split="eval").plan_episodes(args.n_eval_ep)
    print(f"data: {len(train)} single-turn, {len(decay_pool)} decay-pool, {len(episodes)} "
          f"train-eps, {len(held)} held single-turn, {len(held_ep)} held-eps", flush=True)

    # --- ONE shared 1M init: same seed + pretrain stream, deep-copied for every arm ---
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
        extra = ""
        if args.heads:
            extra = (f"  tf_step={m['tf_step_acc']*100:.1f}%  grounded={m['grounded']*100:.1f}%  "
                     f"whole_plan={m['whole_plan']*100:.1f}%")
        print(f"  {name}: loss={m['final_loss']:.3f}  "
              f"single_turn={m['single_turn']*100:.1f}%{extra}  ({m['wall_s']:.0f}s)", flush=True)

    common = dict(steps=args.sft_steps, batch_size=args.batch, lr=args.lr, warmup=args.warmup,
                  device=device, log=log, joint_tool_head=args.heads,
                  conversations=(episodes if args.heads else None), mt_weight=args.mt_weight)
    mode = "head-SFT" if args.heads else "LM-only SFT"

    def run_arm(name, tag, key, **sft_kw):
        t0 = time.time()
        m = fresh()
        hist, head, ptr = sft(m, train, tok, **sft_kw, **common)
        met = eval_arm(m, head, ptr, held=held, held_ep=held_ep, tok=tok, device=device,
                       heads=args.heads)
        met["final_loss"] = final_loss(hist)
        met["wall_s"] = time.time() - t0
        results[key] = met
        report_arm(tag, met)
        return met

    # ===== C: control — cosine LR =====
    print(f"\n=== C (control): {mode}, cosine LR ===", flush=True)
    mC = run_arm("C", "C ", "control_cosine", lr_schedule="cosine")

    # ===== T1: WSD LR (same train pool throughout) =====
    print(f"\n=== T1 (wsd): {mode}, WSD LR (decay_frac={args.decay_frac}) ===", flush=True)
    mT1 = run_arm("T1", "T1", "wsd", lr_schedule="wsd", decay_frac=args.decay_frac)

    # ===== T2: WSD LR + cleanest subset injected in the decay window =====
    print(f"\n=== T2 (wsd+inject): WSD LR + curated decay pool ({len(decay_pool)}) "
          "in decay window ===", flush=True)
    mT2 = run_arm("T2", "T2", "wsd_inject", lr_schedule="wsd", decay_frac=args.decay_frac,
                  decay_samples=decay_pool)

    # --- report table (head metrics only shown when trained) ---
    keys = ["final_loss", "single_turn"]
    if args.heads:
        keys += ["tf_step_acc", "tf_episode_acc", "grounded", "whole_plan"]
    labels = {"final_loss": "final train loss (LOWER)", "single_turn": "single-turn overall",
              "tf_step_acc": "TF next-tool step_acc", "tf_episode_acc": "TF episode_acc",
              "grounded": "grounded_acc", "whole_plan": "free-rollout whole-plan"}
    is_loss = {"final_loss"}
    print("\n" + "=" * 96)
    print(f"{'metric':<26}{'C cos':>10}{'T1 wsd':>10}{'T2 wsd+inj':>12}"
          f"{'T1-C':>10}{'T2-C':>10}{'T2-T1':>10}")
    print("-" * 96)
    for k in keys:
        scale = 1.0 if k in is_loss else 100.0
        c, t1, t2 = mC[k] * scale, mT1[k] * scale, mT2[k] * scale
        suf = "" if k in is_loss else "%"
        print(f"{labels[k]:<26}{c:>9.2f}{suf:>1}{t1:>9.2f}{suf:>1}{t2:>11.2f}{suf:>1}"
              f"{t1 - c:>+10.2f}{t2 - c:>+10.2f}{t2 - t1:>+10.2f}")
    print("=" * 96)
    print(f"mode={mode}  wall-clock: C {mC['wall_s']:.0f}s | T1 {mT1['wall_s']:.0f}s | "
          f"T2 {mT2['wall_s']:.0f}s")

    # --- verdict: at fixed budget, did WSD lower final loss or lift single-turn accuracy? ---
    loss_win = mT1["final_loss"] < mC["final_loss"] - 1e-3
    acc_win = mT1["single_turn"] > mC["single_turn"] + 1e-6
    inj_win = mT2["single_turn"] > mT1["single_turn"] + 1e-6 \
        or mT2["final_loss"] < mT1["final_loss"] - 1e-3
    if loss_win or acc_win:
        verdict = "WSD HELPS at this budget (lower loss and/or higher accuracy)"
    else:
        verdict = "WSD within noise at this scale (expected: payoff is clearest over long runs)"
    print(f"\nVERDICT: {verdict}")
    grnd = f" | grounded {(mT1['grounded']-mC['grounded'])*100:+.1f}" if args.heads else ""
    print(f"  T1 vs C: loss {mT1['final_loss']-mC['final_loss']:+.3f} | "
          f"single_turn {(mT1['single_turn']-mC['single_turn'])*100:+.1f}{grnd}")
    print(f"  T2 (decay-window injection) vs T1: loss {mT2['final_loss']-mT1['final_loss']:+.3f} | "
          f"single_turn {(mT2['single_turn']-mT1['single_turn'])*100:+.1f} | "
          f"injection helps: {inj_win}")

    results["config"] = vars(args)
    results["verdict"] = verdict
    json.dump(results, open(f"{OUT}/result.json", "w"), indent=2)
    print(f"\nmetrics -> {OUT}/result.json")


if __name__ == "__main__":
    main()
