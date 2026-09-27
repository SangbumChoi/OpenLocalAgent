#!/usr/bin/env python
"""Data-flywheel driver for the ultra-tiny (~1M, byte-level) agent.

Per round: SFT (+ short GRPO) on the current enrichment level, evaluate on HELD-OUT slot values,
then enrich (level += 1) and repeat. The model persists across rounds (round 1 trains hardest;
later rounds adapt incrementally).

Two metrics, both honest:
  - free-gen  : the model autoregressively generates the call (the raw ability).
  - grounded  : prompt-grounded constrained decoding (the deployed decoder; ARCHITECTURE_IDEAS
                §2b) — the model ranks candidate calls whose args are grounded in the prompt.
A <100M byte model learns call *structure* fast but not generalizable slot *copying*, so free-gen
stays low on held-out while grounded decoding reaches ~100%. That gap is the whole point.

Outputs (runs/flywheel/): metrics.json, accuracy.png, freegen_vs_grounded.png, loss.png,
samples.json, ultra-tiny.pt

Usage:  python scripts/flywheel.py [--rounds 5] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from localagent.data.agent_synth import Generator
from localagent.data.render import build_pretrain_stream, prompt_text
from localagent.eval.harness import evaluate, evaluate_grounded, multi_turn_eval
from localagent.inference.generate import generate
from localagent.agent.constrained import grounded_decode
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import resolve_device
from localagent.train.pretrain import pretrain
from localagent.train.rl import grpo
from localagent.train.sft import sft

OUT = "runs/flywheel"


def fmt(d):
    return f"overall={d['overall']*100:.1f}%  " + " ".join(
        f"{k}={v*100:.0f}%" for k, v in sorted(d["groups"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")
    tok = load_tokenizer("byte")
    cfg = ModelConfig.from_yaml("configs/model/ultra-tiny-1m.yaml")
    model = LocalAgentLM(cfg).to(device)
    print(f"model {cfg.name}: {model.num_params()/1e6:.3f}M params on {device}", flush=True)

    n_train = 400 if args.quick else 2500
    n_eval = 12 if args.quick else 30        # per category (balanced held-out)
    n_ep = 40 if args.quick else 120         # multi-turn coding episodes per round
    pre_steps = 60 if args.quick else 200
    sft1 = 150 if args.quick else 380
    sft_inc = 80 if args.quick else 130
    grpo_steps = 4 if args.quick else 4

    g0 = Generator(level=1, seed=0, split="train").generate(n_train)
    pre_loss = pretrain(model, build_pretrain_stream(g0, tok), tok, steps=pre_steps,
                        batch_size=64, device=device)

    metrics = {"rounds": [], "pretrain_loss": pre_loss}
    for r in range(1, args.rounds + 1):
        train = Generator(level=r, seed=r, split="train").generate(n_train)
        episodes = Generator(level=r, seed=5000 + r, split="train").episodes(n_ep)
        held = Generator(level=r, seed=1000 + r, split="eval").generate_balanced(n_eval)
        held_ep = Generator(level=r, seed=6000 + r, split="eval").episodes(n_ep // 4)
        steps = sft1 if r == 1 else sft_inc
        print(f"\n=== Round {r} (level {r}, {len(train)} single-turn + {len(episodes)} episodes "
              f"/ {len(held)} held-out) ===", flush=True)
        sft_loss, head, ptr = sft(model, train, tok, steps=steps, batch_size=32, lr=1.5e-3,
                                  device=device, log=lambda *a: None, joint_tool_head=True,
                                  conversations=episodes)
        grpo(model, train, tok, steps=grpo_steps, device=device, log=lambda *a: None)  # RL stage
        # single-turn: heuristic grounding (best on clean templates); multi-turn: pointer head
        # (only it can ground a follow-up arg in an earlier tool response).
        gr = evaluate_grounded(model, held, tok, TOOLS, device=device, tool_head=head)
        mt = multi_turn_eval(model, held_ep, tok, TOOLS, device=device, tool_head=head, ptr_head=ptr)
        print(f"  grounded (held-out): {fmt(gr)}", flush=True)
        print(f"  multi-turn: step_acc={mt['step_acc']*100:.0f}% episode_acc={mt['episode_acc']*100:.0f}%"
              f" ({mt['steps']} steps)", flush=True)
        metrics["rounds"].append({"round": r, "level": r, "grounded": gr, "multi_turn": mt,
                                  "sft_loss_last": sft_loss[-1]})
        torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
                    "tool_head": head.state_dict() if head is not None else None,
                    "ptr_head": ptr.state_dict() if ptr is not None else None},
                   f"{OUT}/ultra-tiny.pt")
        json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
        _plot_rounds(metrics)

    # final comparison: raw free-generation vs grounded, on the last level's held-out
    final_held = Generator(level=args.rounds, seed=4242, split="eval").generate_balanced(20)
    fg = evaluate(model, final_held, tok, device=device)
    gr = evaluate_grounded(model, final_held, tok, TOOLS, device=device, tool_head=head, ptr_head=ptr)
    metrics["final_freegen"] = fg
    metrics["final_grounded"] = gr
    print(f"\nFINAL free-gen : {fmt(fg)}")
    print(f"FINAL grounded : {fmt(gr)}")

    samples = []
    for s in Generator(level=args.rounds, seed=999, split="eval").generate(8):
        out = grounded_decode(model, tok, s.prompt, TOOLS, device=device, tool_head=head, ptr_head=ptr)
        samples.append({"prompt": s.prompt, "expected": s.target, "grounded_out": out})
    json.dump(samples, open(f"{OUT}/samples.json", "w"), indent=2)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
    _plot_rounds(metrics)
    _plot_compare(metrics)
    _plot_loss(pre_loss)
    _plot_multiturn(metrics)
    print(f"\nArtifacts in {OUT}/ (accuracy.png, freegen_vs_grounded.png, loss.png, samples.json)")


def _plot_multiturn(metrics):
    """Tracked figure (figures/18): multi-turn trajectory accuracy across rounds, vs the untrained
    baseline (a frozen single-turn model replays the new episodes at ~18% step-acc)."""
    if not all("multi_turn" in m for m in metrics["rounds"]):
        return
    try:
        from localagent.figs import savefig
        plt = _mpl()
    except Exception:
        return
    rs = [m["round"] for m in metrics["rounds"]]
    step = [m["multi_turn"]["step_acc"] * 100 for m in metrics["rounds"]]
    ep = [m["multi_turn"]["episode_acc"] * 100 for m in metrics["rounds"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(18, ls="--", color="gray", lw=1.5, label="untrained baseline (~18% step)")
    ax.plot(rs, step, marker="o", lw=2.5, color="tab:purple", label="step accuracy")
    ax.plot(rs, ep, marker="s", lw=2, color="tab:orange", label="whole-episode accuracy")
    ax.set_xlabel("flywheel round (enrichment level)")
    ax.set_ylabel("multi-turn accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(rs)
    ax.grid(alpha=.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Learning plan->act: multi-turn trajectory accuracy (1M, coding+computer-use+planner)")
    for x, y in zip(rs, step):
        ax.text(x, y + 2, f"{y:.0f}", ha="center", fontsize=8, color="tab:purple")
    fig.tight_layout()
    savefig(fig, "18_multiturn_trajectory_learning")
    plt.close(fig)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_rounds(metrics):
    try:
        plt = _mpl()
    except Exception:
        return
    rs = [m["round"] for m in metrics["rounds"]]
    groups = sorted({g for m in metrics["rounds"] for g in m["grounded"]["groups"]})
    fig, ax = plt.subplots(figsize=(7, 4))
    for g in groups:
        ax.plot(rs, [m["grounded"]["groups"].get(g, 0) * 100 for m in metrics["rounds"]],
                marker="o", label=g)
    ax.plot(rs, [m["grounded"]["overall"] * 100 for m in metrics["rounds"]],
            marker="s", lw=2.5, color="black", label="overall")
    ax.set_xlabel("flywheel round (enrichment level)"); ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(0, 105); ax.set_xticks(rs); ax.grid(alpha=.3); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ultra-tiny ~1M: grounded held-out accuracy across flywheel rounds")
    fig.tight_layout(); fig.savefig(f"{OUT}/accuracy.png", dpi=120); plt.close(fig)


def _plot_compare(metrics):
    try:
        plt = _mpl(); import numpy as np
    except Exception:
        return
    fg, gr = metrics["final_freegen"], metrics["final_grounded"]
    groups = sorted(gr["groups"])
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, [fg["groups"].get(g, 0) * 100 for g in groups], w, label="free-gen (raw)")
    ax.bar(x + w / 2, [gr["groups"].get(g, 0) * 100 for g in groups], w, label="grounded decode")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(0, 105); ax.grid(alpha=.3, axis="y"); ax.legend()
    ax.set_title("Same 1M model: raw byte generation vs prompt-grounded decoding")
    fig.tight_layout(); fig.savefig(f"{OUT}/freegen_vs_grounded.png", dpi=120); plt.close(fig)


def _plot_loss(pre_loss):
    try:
        plt = _mpl()
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pre_loss); ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("Pretrain loss (next-byte CE)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/loss.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
