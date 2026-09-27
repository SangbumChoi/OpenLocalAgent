#!/usr/bin/env python
"""Controlled 3-arm A/B: does running the 30M->1M KD term *throughout* head-SFT recover the
grounding that distill-then-SFT regressed, while keeping its single-turn / free-rollout gains?

Background (scripts/distill_ab.py): distill-then-SFT warmed the 1M backbone with Top-K KD, THEN
head-SFT. It won single-turn (+5.4) and free-rollout whole-plan (+7.5) but REGRESSED grounded-call
acc (-29) because head-SFT re-specialized the backbone away from verbatim argument-copying.

This script isolates the *schedule* of the KD term. All three arms share, byte-for-byte: the one
pretrained 1M init (same seed, deep-copied), the SFT samples / seed / steps, the episode set, and
the eval sets. The ONLY difference is when/whether the 30M teacher's Top-K next-token term is
applied:

  C  (control)              : init -> head-SFT only                                    -> eval
  T1 (distill-then-SFT)     : init -> distill(30M->1M, top-k) warmup -> SAME head-SFT  -> eval
  T2 (distill-throughout)   : init -> head-SFT WITH teacher KD term concurrent         -> eval

T2 uses the new opt-in path: sft(..., teacher=teacher30m, kd_type='topk', kd_weight=...). The
KD term keeps the backbone matching the teacher's next-token distribution WHILE the heads train,
so the backbone is never pulled off arg-copying.

We report, per arm: single-turn overall, teacher-forced step/episode acc, grounded_acc (the
metric that regressed -- the headline), and free-rollout whole-plan acc, plus deltas vs control.

Usage:  OMP_NUM_THREADS=4 python scripts/distill_sft_ab.py [--quick]

MEMORY/TIME: this sandbox SIGKILLs big batches. Defaults use batch_size=8 everywhere and a small
pretrain-init batch; do NOT raise pretrain batch to 64. Keep steps bounded.
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

OUT = "runs/distill_sft_ab"
STUDENT_CFG = "configs/model/ultra-tiny-1m.yaml"
TEACHER_CFG = "configs/model/tiny-30m-byte.yaml"
TEACHER_CKPT = "runs/tiny-30m-byte-best.pt"
TEACHER_CKPT_FALLBACK = "runs/analyze_tiny-30m-byte/model.pt"


def load_teacher(device):
    ckpt = TEACHER_CKPT if os.path.exists(TEACHER_CKPT) else TEACHER_CKPT_FALLBACK
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    # Build the teacher from the cfg that PRODUCED its weights (ck["cfg"]). The committed
    # tiny-30m-byte.yaml has since gained optional fields (conv_kernel/qk_norm/layer_types) that
    # default-off, so we sanity-check only the fields the checkpoint actually stored rather than
    # requiring an exact dict match against the newer YAML.
    tcfg = ModelConfig(**ck["cfg"])
    ycfg = ModelConfig.from_yaml(TEACHER_CFG).__dict__
    for k, v in ck["cfg"].items():
        assert ycfg.get(k) == v, f"teacher ckpt cfg.{k}={v} != {TEACHER_CFG} {ycfg.get(k)}"
    teacher = LocalAgentLM(tcfg).to(device)
    teacher.load_state_dict(ck["state_dict"])
    teacher.eval()
    print(f"teacher loaded from {ckpt}: {teacher.num_params()/1e6:.1f}M (frozen)", flush=True)
    return teacher


def eval_arm(model, head, ptr, *, held, held_ep, tok, device):
    res = evaluate_grounded(model, held, tok, TOOLS, device=device, tool_head=head, ptr_head=ptr)
    pe = plan_eval(model, tok, TOOLS, held_ep, tool_head=head, ptr_head=ptr, device=device)
    return {
        "single_turn": res["overall"],
        "tf_step_acc": pe["teacher_forced"]["step_acc"],
        "tf_episode_acc": pe["teacher_forced"]["episode_acc"],
        "grounded": pe["grounded_acc"],
        "whole_plan": pe["whole_plan_acc"],
        "plan_step_acc": pe["step_acc"],
        "categories": res["categories"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="student init seed (shared by all arms)")
    ap.add_argument("--pre", type=int, default=200, help="pretrain-init steps (shared init)")
    ap.add_argument("--distill-steps", type=int, default=300, help="T1 warmup distill steps")
    ap.add_argument("--sft-steps", type=int, default=400, help="head-SFT steps (all arms)")
    ap.add_argument("--batch", type=int, default=8, help="SFT/distill batch (keep small: OOM)")
    ap.add_argument("--pre-batch", type=int, default=8, help="pretrain-init batch (DO NOT raise)")
    ap.add_argument("--mt-weight", type=float, default=1.5)
    ap.add_argument("--kd-weight", type=float, default=0.5, help="T2 concurrent KD weight")
    ap.add_argument("--kd-k", type=int, default=16)
    ap.add_argument("--kd-temperature", type=float, default=2.0)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-ep", type=int, default=120)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--n-eval-ep", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.pre, args.distill_steps, args.sft_steps = 30, 50, 60
        args.n_train, args.n_ep, args.n_eval, args.n_eval_ep = 600, 30, 8, 12

    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")
    tok = load_tokenizer("byte")
    r = args.round
    log = lambda *a: None  # noqa: E731

    print(f"device={device}  threads={torch.get_num_threads()}  round={r}", flush=True)

    # --- shared data (identical across all arms) ---
    train = Generator(level=r, seed=r, split="train").generate(args.n_train)
    episodes = Generator(level=r, seed=5000 + r, split="train").plan_episodes(args.n_ep)
    held = Generator(level=r, seed=1000 + r, split="eval").generate_balanced(args.n_eval)
    held_ep = Generator(level=r, seed=6000 + r, split="eval").plan_episodes(args.n_eval_ep)
    print(f"data: {len(train)} single-turn, {len(episodes)} train-eps, "
          f"{len(held)} held single-turn, {len(held_ep)} held-eps", flush=True)

    # --- teacher (frozen 30M) ---
    teacher = load_teacher(device)

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
        print(f"  {name}: single_turn={m['single_turn']*100:.1f}%  "
              f"tf_step={m['tf_step_acc']*100:.1f}%  grounded={m['grounded']*100:.1f}%  "
              f"whole_plan={m['whole_plan']*100:.1f}%  ({m['wall_s']:.0f}s)", flush=True)

    # ===== C: control — head-SFT only =====
    print("\n=== C (control): init -> head-SFT only ===", flush=True)
    t0 = time.time()
    cm = fresh()
    _, ch, cp = sft(cm, train, tok, steps=args.sft_steps, batch_size=args.batch, lr=1.5e-3,
                    device=device, log=log, joint_tool_head=True, conversations=episodes,
                    mt_weight=args.mt_weight)
    mC = eval_arm(cm, ch, cp, held=held, held_ep=held_ep, tok=tok, device=device)
    mC["wall_s"] = time.time() - t0
    results["control"] = mC
    report_arm("C ", mC)

    # ===== T1: distill-then-SFT (old recipe) =====
    print("\n=== T1 (distill-then-SFT): init -> distill warmup -> head-SFT ===", flush=True)
    t0 = time.time()
    t1m = fresh()
    distill(t1m, train, teacher, tok, kd_type="topk", kd_k=args.kd_k, steps=args.distill_steps,
            temperature=args.kd_temperature, kd_weight=1.0, ce_weight=0.2, lr=1.5e-3,
            batch_size=args.batch, device=device, log=log)
    distill_s = time.time() - t0
    _, t1h, t1p = sft(t1m, train, tok, steps=args.sft_steps, batch_size=args.batch, lr=1.5e-3,
                      device=device, log=log, joint_tool_head=True, conversations=episodes,
                      mt_weight=args.mt_weight)
    mT1 = eval_arm(t1m, t1h, t1p, held=held, held_ep=held_ep, tok=tok, device=device)
    mT1["wall_s"] = time.time() - t0
    mT1["distill_s"] = distill_s
    results["distill_then_sft"] = mT1
    report_arm("T1", mT1)

    # ===== T2: distill-throughout-SFT (new path) =====
    print("\n=== T2 (distill-throughout): init -> head-SFT WITH concurrent teacher KD ===",
          flush=True)
    t0 = time.time()
    t2m = fresh()
    _, t2h, t2p = sft(t2m, train, tok, steps=args.sft_steps, batch_size=args.batch, lr=1.5e-3,
                      device=device, log=log, joint_tool_head=True, conversations=episodes,
                      mt_weight=args.mt_weight, teacher=teacher, kd_type="topk", kd_k=args.kd_k,
                      kd_weight=args.kd_weight, kd_temperature=args.kd_temperature)
    mT2 = eval_arm(t2m, t2h, t2p, held=held, held_ep=held_ep, tok=tok, device=device)
    mT2["wall_s"] = time.time() - t0
    results["distill_throughout_sft"] = mT2
    report_arm("T2", mT2)

    # --- report table ---
    keys = ["single_turn", "tf_step_acc", "tf_episode_acc", "grounded", "whole_plan"]
    labels = {"single_turn": "single-turn overall", "tf_step_acc": "TF next-tool step_acc",
              "tf_episode_acc": "TF episode_acc", "grounded": "grounded_acc (HEADLINE)",
              "whole_plan": "free-rollout whole-plan"}
    print("\n" + "=" * 92)
    print(f"{'metric':<26}{'C ctrl':>10}{'T1 then':>10}{'T2 thru':>10}"
          f"{'T1-C':>10}{'T2-C':>10}{'T2-T1':>10}")
    print("-" * 92)
    for k in keys:
        c, t1, t2 = mC[k] * 100, mT1[k] * 100, mT2[k] * 100
        print(f"{labels[k]:<26}{c:>9.1f}%{t1:>9.1f}%{t2:>9.1f}%"
              f"{t1 - c:>+10.1f}{t2 - c:>+10.1f}{t2 - t1:>+10.1f}")
    print("=" * 92)
    print(f"wall-clock: C {mC['wall_s']:.0f}s | T1 {mT1['wall_s']:.0f}s "
          f"(distill {mT1['distill_s']:.0f}s) | T2 {mT2['wall_s']:.0f}s")

    # --- verdict: did T2 keep T1's single-turn/whole-plan gains WITHOUT the grounding regression?
    st_gain = mT2["single_turn"] >= mC["single_turn"] - 0.01
    wp_gain = mT2["whole_plan"] >= mC["whole_plan"] - 0.01
    grnd_ok = mT2["grounded"] >= mC["grounded"] - 0.03
    t1_regressed = mT1["grounded"] < mC["grounded"] - 0.03
    verdict = (
        "T2 RECOVERS grounding while keeping gains"
        if (st_gain and wp_gain and grnd_ok)
        else "T2 did NOT cleanly recover grounding + gains"
    )
    print(f"\nVERDICT: {verdict}")
    print(f"  T1 grounded regressed vs C: {t1_regressed} "
          f"({(mT1['grounded']-mC['grounded'])*100:+.1f})")
    print(f"  T2 grounded vs C: {(mT2['grounded']-mC['grounded'])*100:+.1f} | "
          f"single_turn vs C: {(mT2['single_turn']-mC['single_turn'])*100:+.1f} | "
          f"whole_plan vs C: {(mT2['whole_plan']-mC['whole_plan'])*100:+.1f}")

    results["config"] = vars(args)
    results["verdict"] = verdict
    json.dump(results, open(f"{OUT}/result.json", "w"), indent=2)
    print(f"\nmetrics -> {OUT}/result.json")


if __name__ == "__main__":
    main()
