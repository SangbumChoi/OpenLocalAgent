#!/usr/bin/env python
"""Which selection method scales to 100s–1000s of tools?

A fixed N-way classifier head can't (fixed N, O(N) head, no unseen tools). This measures
**retrieval** top-k selection as the catalog grows: build catalogs of increasing size, then for
held-out queries (disjoint slot values) report recall@k and end-to-end tool-call accuracy, plus a
2-step multi-turn version. Also writes train/test use-case datasets.

Outputs (runs/catalog/): scale.png, report.json, train.jsonl, test.jsonl, episodes_test.jsonl
Usage:  python scripts/tool_scale_analysis.py
"""

from __future__ import annotations

import json
import os
import time

from localagent.agent.constrained import _quoted
from localagent.agent.retriever import ToolRetriever
from localagent.data.tool_catalog import build_catalog, dump_jsonl, gen_episodes, gen_usages

OUT = "runs/catalog"
SIZES = [50, 100, 250, 500, 1000, 1350]
KS = [1, 5, 10, 20]


def _metrics(retr, queries):
    """recall@k, MRR, and end-to-end (top-1 tool right AND quoted arg extracted right)."""
    rec = {k: 0 for k in KS}
    mrr = e2e = 0
    for q in queries:
        top = retr.retrieve(q["prompt"], k=max(KS))
        if q["tool"] in top:
            r = top.index(q["tool"]) + 1
            mrr += 1.0 / r
            for k in KS:
                rec[k] += r <= k
            if r == 1:
                got = _quoted(q["prompt"])
                e2e += bool(got) and got[0] == q["value"]
    n = len(queries)
    return {**{f"recall@{k}": rec[k] / n for k in KS}, "mrr": mrr / n, "e2e_acc": e2e / n, "n": n}


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"sizes": [], "ks": KS}
    for N in SIZES:
        tools = build_catalog(N, seed=0)
        # method A: index by description only.  method B: also index by example usages (paraphrased)
        examples = {}
        for u in gen_usages(tools, split="train", per_tool=4, seed=3, paraphrase=True):
            examples.setdefault(u["tool"], []).append(u["prompt"])
        t0 = time.time(); desc = ToolRetriever(tools); idx_s = time.time() - t0
        exr = ToolRetriever(tools, examples=examples)
        test = gen_usages(tools, split="eval", per_tool=1, seed=7)        # paraphrased eval
        mA, mB = _metrics(desc, test), _metrics(exr, test)
        eps = gen_episodes(tools, n=min(400, 4 * N), split="eval", seed=9)
        steps = [{"prompt": s["history"] + s["prompt"], "tool": s["tool"], "value": s["value"]}
                 for e in eps for s in e]
        mt = _metrics(exr, steps)
        report["sizes"].append({"N": N, "desc_only": mA, "example_aug": mB, "multi_turn": mt,
                                "index_s": round(idx_s, 2)})
        print(f"N={N:5d}  desc-only recall@1={mA['recall@1']*100:4.1f}% @10={mA['recall@10']*100:4.1f}%  "
              f"| example-aug recall@1={mB['recall@1']*100:4.1f}% @5={mB['recall@5']*100:4.1f}% "
              f"@10={mB['recall@10']*100:4.1f}%  | mt@1={mt['recall@1']*100:4.1f}%  (idx {idx_s:.2f}s)",
              flush=True)

    # write train/test use-case datasets + episodes for the full catalog
    full = build_catalog(max(SIZES), seed=0)
    dump_jsonl(gen_usages(full, "train", per_tool=3, seed=1), f"{OUT}/train.jsonl")
    dump_jsonl(gen_usages(full, "eval", per_tool=2, seed=2), f"{OUT}/test.jsonl")
    dump_jsonl([s for e in gen_episodes(full, 600, "eval", 3) for s in e], f"{OUT}/episodes_test.jsonl")
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
    _plot(report)
    print(f"\nDatasets + report + scale.png in {OUT}/  ({max(SIZES)} tools)")


def _plot(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [s["N"] for s in report["sizes"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, [s["desc_only"]["recall@1"] * 100 for s in report["sizes"]], marker="s",
            color="#bbb", label="desc-only recall@1")
    for k, c in zip([1, 5, 10], ["#c03070", "#3070c0", "#30a060"]):
        ax.plot(xs, [s["example_aug"][f"recall@{k}"] * 100 for s in report["sizes"]], marker="o",
                color=c, label=f"example-aug recall@{k}")
    ax.plot(xs, [s["multi_turn"]["recall@1"] * 100 for s in report["sizes"]], marker="x",
            ls="--", color="gray", label="multi-turn recall@1 (ex-aug)")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(xs)
    ax.set_xlabel("tool catalog size (log)"); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 102)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Tool selection at scale (paraphrased queries): index tools by example usages")
    fig.tight_layout(); fig.savefig(f"{OUT}/scale.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
