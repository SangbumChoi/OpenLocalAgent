#!/usr/bin/env python
"""Distillation demo: a ~28M byte teacher -> the ~1M byte student (offline logit KD).

Trains a teacher, then trains the SAME 1M student two ways from one pretrained init:
  baseline : SFT (ground-truth CE) only
  distilled: KD against the teacher's soft next-byte targets (+ a little CE)
and compares held-out next-byte NLL and top-1 accuracy on the assistant/tool-call spans.
Lower NLL / higher accuracy for the distilled student = the teacher's knowledge transferred.

Usage:  python scripts/distill_demo.py [--kd forward_kl|reverse_kl]
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import torch

from localagent.data.agent_synth import Generator
from localagent.data.render import IGNORE, build_pretrain_stream, render_sft
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import resolve_device
from localagent.train.distill import distill
from localagent.train.pretrain import pretrain
from localagent.train.sft import sft

OUT = "runs/distill"


@torch.no_grad()
def eval_lm(model, samples, tok, device):
    """Mean NLL and top-1 accuracy on assistant/tool-call tokens over held-out samples."""
    model.eval()
    rows = [render_sft(s, tok) for s in samples]
    nll = correct = ntok = 0.0
    for ids, labels in rows:
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor(labels[1:], device=device)
        logits, _ = model(x)
        lp = logits[0].log_softmax(-1)
        m = y != IGNORE
        if m.sum() == 0:
            continue
        yi = y[m]
        nll += -lp[m][torch.arange(yi.shape[0]), yi].sum().item()
        correct += (logits[0][m].argmax(-1) == yi).sum().item()
        ntok += yi.shape[0]
    return {"nll": nll / ntok, "top1": correct / ntok, "tokens": int(ntok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kd", default="forward_kl", choices=["forward_kl", "reverse_kl"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")
    tok = load_tokenizer("byte")
    n_train = 400 if args.quick else 1500
    t_pre, t_sft, s_pre, s_steps = (40, 60, 40, 120) if args.quick else (150, 200, 150, 300)

    train = Generator(level=1, seed=0, split="train").generate(n_train)
    held = Generator(level=1, seed=999, split="eval").generate_balanced(20)

    # 1) teacher ~28M
    tcfg = ModelConfig.from_yaml("configs/model/tiny-30m-byte.yaml")
    teacher = LocalAgentLM(tcfg).to(device)
    print(f"teacher {tcfg.name}: {teacher.num_params()/1e6:.1f}M — pretrain+SFT", flush=True)
    pretrain(teacher, build_pretrain_stream(train, tok), tok, steps=t_pre, batch_size=64,
             device=device, log=lambda *a: None)
    sft(teacher, train, tok, steps=t_sft, batch_size=16, lr=1e-3, device=device, log=lambda *a: None)
    t_eval = eval_lm(teacher, held, tok, device)
    print(f"  teacher held-out: NLL={t_eval['nll']:.3f} top1={t_eval['top1']*100:.1f}%", flush=True)

    # 2) one pretrained 1M init, cloned for a fair baseline-vs-KD comparison
    scfg = ModelConfig.from_yaml("configs/model/ultra-tiny-1m.yaml")
    base_init = LocalAgentLM(scfg).to(device)
    pretrain(base_init, build_pretrain_stream(train, tok), tok, steps=s_pre, batch_size=64,
             device=device, log=lambda *a: None)
    init_state = copy.deepcopy(base_init.state_dict())

    student_base = LocalAgentLM(scfg).to(device); student_base.load_state_dict(init_state)
    sft(student_base, train, tok, steps=s_steps, batch_size=32, lr=1.5e-3, device=device,
        log=lambda *a: None)
    b_eval = eval_lm(student_base, held, tok, device)
    print(f"  student SFT-only : NLL={b_eval['nll']:.3f} top1={b_eval['top1']*100:.1f}%", flush=True)

    student_kd = LocalAgentLM(scfg).to(device); student_kd.load_state_dict(init_state)
    distill(student_kd, train, teacher, tok, steps=s_steps, kd_type=args.kd, temperature=2.0,
            kd_weight=1.0, ce_weight=0.2, lr=1.5e-3, device=device, log=print)
    k_eval = eval_lm(student_kd, held, tok, device)
    print(f"  student DISTILLED: NLL={k_eval['nll']:.3f} top1={k_eval['top1']*100:.1f}%", flush=True)

    res = {"kd_type": args.kd, "teacher": t_eval, "student_sft": b_eval, "student_distilled": k_eval}
    json.dump(res, open(f"{OUT}/result.json", "w"), indent=2)
    _plot(res)
    d = (k_eval["top1"] - b_eval["top1"]) * 100
    print(f"\nDistillation effect (1M student): top1 {b_eval['top1']*100:.1f}% -> "
          f"{k_eval['top1']*100:.1f}%  ({d:+.1f} pts), NLL {b_eval['nll']:.3f} -> {k_eval['nll']:.3f}")
    print(f"Artifacts in {OUT}/")


def _plot(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    names = ["teacher ~28M", "student 1M\nSFT-only", "student 1M\ndistilled"]
    top1 = [res["teacher"]["top1"] * 100, res["student_sft"]["top1"] * 100,
            res["student_distilled"]["top1"] * 100]
    nll = [res["teacher"]["nll"], res["student_sft"]["nll"], res["student_distilled"]["nll"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    c = ["#3070c0", "#bbb", "#e08020"]
    a1.bar(names, top1, color=c); a1.set_ylabel("held-out next-byte top-1 (%)"); a1.set_ylim(0, 105)
    a1.set_title("Assistant-token accuracy"); a1.grid(alpha=.3, axis="y")
    for i, v in enumerate(top1):
        a1.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    a2.bar(names, nll, color=c); a2.set_ylabel("held-out NLL (lower=better)")
    a2.set_title(f"Next-byte NLL ({res['kd_type']})"); a2.grid(alpha=.3, axis="y")
    fig.suptitle("Distillation: 28M teacher -> 1M student (offline logit KD)")
    fig.tight_layout(); fig.savefig(f"{OUT}/distill.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
