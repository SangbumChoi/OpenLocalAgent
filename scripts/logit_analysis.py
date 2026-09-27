#!/usr/bin/env python
"""Logit-level autopsy: WHY grounded constrained decoding beats free byte-generation.

WHY: experiment 01 shows the same tiny byte model scores ~1% by free argmax generation but 100%
with grounded constrained decoding. The doc needs the *mechanistic* reason, in terms of the
model's own next-byte logit distribution.

HYPOTHESIS (tested here): a from-scratch byte model learns the tool-call *structure* — the JSON
scaffolding `{"arguments":{...},"name":"..."}` and the tool name, which are in-distribution — with
HIGH confidence (low next-byte entropy, ~100% top-1). But at the *argument-value bytes* (the copied
slot value, e.g. a held-out city "Boston") its next-byte distribution is HIGH-ENTROPY and its argmax
is usually WRONG, because eval slot pools are disjoint from train: the value is genuinely not in the
weights, it lives in the prompt. That gap is exactly why free-generation ≈1% (it must argmax through
the uncertain slot region) while grounded constrained decoding = 100% (it COPIES the slot bytes from
the prompt, sidestepping the model's uncertain logits at precisely those positions).

What it tests: teacher-force the assistant body of held-out single-call tool samples through the
trained 28M byte model, collect per-position next-byte logits, classify each body byte as STRUCTURAL
(scaffolding/keys/tool-name/punctuation/EOS) vs SLOT-VALUE (inside an argument value string), and
compare mean entropy, top-1 accuracy, mean gold-byte prob, and prob mass on the prompt's byte-set
("copy mass") between the two groups.

  python scripts/logit_analysis.py
"""

from __future__ import annotations

import json
import os

import torch

from localagent.data.agent_synth import Generator
from localagent.data.render import IGNORE, assistant_body, prompt_text, render_sft
from localagent.figs import savefig
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

CKPT = "runs/analyze_tiny-30m-byte/model.pt"
OUT = "runs/logit_analysis"


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in ck["cfg"].items() if k != "name"}
    model = LocalAgentLM(ModelConfig(**cfg))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def slot_byte_indices(s) -> set[int]:
    """Body-text byte offsets (0-based, within assistant_body bytes) that lie inside an argument
    *value* string. Everything else in the body is structural scaffolding.

    We only count occurrences that are quoted JSON string *values* — the bytes must be immediately
    preceded by `:"` and followed by `"`. This avoids over-marking short values (e.g. unit "c")
    that also appear incidentally in keys/punctuation."""
    body = assistant_body(s)
    bbytes = body.encode("utf-8")
    args = json.loads(s.ref_args) if s.ref_args else {}
    slot: set[int] = set()
    for v in args.values():
        vb = str(v).encode("utf-8")
        if not vb:
            continue
        start = 0
        while True:
            i = bbytes.find(vb, start)
            if i < 0:
                break
            before = bbytes[i - 2:i]
            after = bbytes[i + len(vb):i + len(vb) + 1]
            if before == b':"' and after == b'"':   # quoted JSON string value
                slot.update(range(i, i + len(vb)))
            start = i + 1
    return slot


def analyze(model, tok, samples):
    # accumulators: [sum_entropy, sum_top1, sum_goldprob, sum_copymass, n]
    agg = {"structural": [0.0, 0, 0.0, 0.0, 0], "slot": [0.0, 0, 0.0, 0.0, 0]}
    for s in samples:
        ids, labels = render_sft(s, tok)
        p_len = len(tok.encode(prompt_text(s)))  # body starts at index p_len
        slot_local = slot_byte_indices(s)        # offsets within the body bytes
        prompt_bytes = set(prompt_text(s).encode("utf-8"))
        copy_mask = torch.zeros(256, dtype=torch.bool)
        copy_mask[list(prompt_bytes)] = True

        with torch.no_grad():
            logits, _ = model(torch.tensor([ids]))
        logits = logits[0]  # [T, V]

        for j in range(1, len(ids)):
            if labels[j] == IGNORE:
                continue  # prompt position, not a learned body target
            row = logits[j - 1]  # position j-1 predicts token j
            probs = torch.softmax(row.float(), dim=-1)
            gold = ids[j]
            ent = float(-(probs * (probs.clamp_min(1e-12)).log2()).sum())
            top1 = int(torch.argmax(row).item() == gold)
            goldp = float(probs[gold])
            copyp = float(probs[copy_mask].sum())

            body_off = j - p_len  # 0-based offset of this byte within the body
            # the EOS target sits at body_off == len(body_bytes); treat as structural
            kind = "slot" if body_off in slot_local else "structural"
            a = agg[kind]
            a[0] += ent
            a[1] += top1
            a[2] += goldp
            a[3] += copyp
            a[4] += 1
    return agg


def summarize(agg):
    out = {}
    for k, (se, st, sg, sc, n) in agg.items():
        n = max(n, 1)
        out[k] = {
            "n_bytes": agg[k][4],
            "mean_entropy_bits": se / n,
            "top1_acc": st / n,
            "mean_gold_prob": sg / n,
            "mean_copy_mass": sc / n,
        }
    return out


def _entropy_top5(row):
    probs = torch.softmax(row.float(), dim=-1)
    ent = float(-(probs * probs.clamp_min(1e-12).log2()).sum())
    topv, topi = torch.topk(probs, 5)
    top = [(chr(i) if 32 <= i < 127 else f"\\x{i:02x}", round(float(p), 3))
           for i, p in zip(topi.tolist(), topv.tolist())]
    return ent, top


def unconditional_stats(model, tok):
    """Weights-only views (no user prompt), to check whether the slot effect is *in the weights*:

      raw          — unconditional next byte from a single EOS (the model's byte prior).
      value_start  — feed just the body scaffolding up to the opening quote of the city value
                     (`...{"arguments":{"city":"`) with NO user prompt. The model now has to *emit*
                     a value it cannot copy; if the slot uncertainty is genuinely a grounding gap,
                     this position is high-entropy and its argmax is an arbitrary common letter."""
    with torch.no_grad():
        raw, _ = model(torch.tensor([[0]]))
    raw_ent, raw_top = _entropy_top5(raw[0, 0])

    prefix = '<|assistant|><tool_call>{"arguments":{"city":"'
    pids = tok.encode(prefix)
    with torch.no_grad():
        vlog, _ = model(torch.tensor([pids]))
    v_ent, v_top = _entropy_top5(vlog[0, -1])
    return {"raw_entropy_bits": raw_ent, "raw_top5": raw_top,
            "value_start_entropy_bits": v_ent, "value_start_top5": v_top}


def plot(summ):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None
    groups = ["structural", "slot"]
    labels = ["structural\n(scaffolding/name)", "slot-value\n(copied arg)"]
    ent = [summ[g]["mean_entropy_bits"] for g in groups]
    acc = [summ[g]["top1_acc"] * 100 for g in groups]
    x = np.arange(len(groups))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    c = ["#3070c0", "#e08020"]
    b1 = ax1.bar(x, ent, color=c, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("mean next-byte entropy (bits)")
    ax1.set_title("(a) Where the model is uncertain")
    ax1.grid(alpha=.3, axis="y")
    for r, v in zip(b1, ent):
        ax1.text(r.get_x() + r.get_width() / 2, v + max(ent) * 0.01, f"{v:.2f}",
                 ha="center", fontsize=10)
    b2 = ax2.bar(x, acc, color=c, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("top-1 next-byte accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("(b) Where free argmax fails")
    ax2.grid(alpha=.3, axis="y")
    for r, v in zip(b2, acc):
        ax2.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
    fig.suptitle("Where the tiny model is uncertain: structure vs slot-value bytes",
                 fontsize=13, y=1.02)
    path = savefig(fig, "17_logit_entropy_structural_vs_slot")
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = load_tokenizer("byte")
    model = load_model()

    gen = Generator(level=3, seed=6000, split="eval")
    samples = gen.generate_balanced(per_category=12)
    samples = [s for s in samples if s.kind == "tool" and s.calls is None and s.ref_args
               and json.loads(s.ref_args)]  # single-call tools with at least one arg value
    print(f"{len(samples)} held-out single-call tool samples (disjoint eval slot pools)\n")

    agg = analyze(model, tok, samples)
    summ = summarize(agg)
    uncond = unconditional_stats(model, tok)

    print(f"{'group':<12}{'n_bytes':>9}{'entropy':>10}{'top1':>9}{'gold_p':>9}{'copy_mass':>11}")
    for g in ("structural", "slot"):
        d = summ[g]
        print(f"{g:<12}{d['n_bytes']:>9}{d['mean_entropy_bits']:>10.3f}"
              f"{d['top1_acc']*100:>8.1f}%{d['mean_gold_prob']:>9.3f}{d['mean_copy_mass']*100:>10.1f}%")
    print(f"\nweights-only (no prompt): raw next-byte entropy {uncond['raw_entropy_bits']:.3f} bits; "
          f"at city-value position {uncond['value_start_entropy_bits']:.3f} bits "
          f"(argmax top5={uncond['value_start_top5']})")

    report = {"n_samples": len(samples), "groups": summ, "unconditional": uncond}
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
    fig_path = plot(summ)
    print(f"\nreport -> {OUT}/report.json   figure -> {fig_path}")


if __name__ == "__main__":
    main()
