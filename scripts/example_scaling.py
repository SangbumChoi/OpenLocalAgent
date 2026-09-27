#!/usr/bin/env python
"""How much per-tool data does retrieval need? (example-usage scaling)

The retriever can index tools by their description OR by example usages. We proved example-usages
bridge the paraphrase gap — but *how many* do you need? Here we vary the number of (paraphrased)
example phrasings per tool and measure selection/full-call on held-out paraphrased queries, with
distractor tools mixed in. The point is to find the saturation knee — the practical answer to
"add more data until it stops helping".

  python scripts/example_scaling.py
"""

from __future__ import annotations

import json
import os
import random

from localagent.agent.caller import ToolCaller
from localagent.eval import codebench
from localagent.figs import savefig

OUT = "runs/example_scaling"
COUNTS = [0, 1, 2, 4, 8, 16, 32]
DISTRACTORS = 300


def gen_examples(defs, n: int, seed: int = 0) -> dict:
    """n paraphrased example phrasings per tool (random verb synonym + random train value)."""
    rng = random.Random(seed)
    ex = {}
    for name, desc, args, templates, verb, syn in defs:
        verbs = [verb, *syn]
        ex[name] = []
        for _ in range(n):
            t = rng.choice(templates)
            vals = [rng.choice(a[2]) for a in args]
            ex[name].append(t.format(verb=rng.choice(verbs),
                                     **{f"a{i}": v for i, v in enumerate(vals)}))
    return ex


def main():
    os.makedirs(OUT, exist_ok=True)
    base = codebench.build_tools()
    gold = codebench.gold_set("eval", seed=7)
    from localagent.data.tool_catalog import build_catalog, gen_usages
    extra = build_catalog(DISTRACTORS, seed=2)
    distractor_ex = {}
    for u in gen_usages(extra, "train", per_tool=6, seed=3, paraphrase=True):
        distractor_ex.setdefault(u["tool"], []).append(u["prompt"])
    tools = base + extra

    print(f"{len(base)} coding tools + {DISTRACTORS} distractors = {len(tools)} tools, "
          f"{len(gold)} held-out paraphrased queries\n")
    rows = []
    for n in COUNTS:
        ex = {**gen_examples(codebench.CODE_TOOLS, n, seed=1), **distractor_ex}
        caller = ToolCaller(tools, examples=ex)
        tool_ok = full = 0
        for q, g in gold:
            r = caller.call(q)
            t = r is not None and r.name == g["name"]
            tool_ok += t; full += t and r.arguments == g["arguments"]
        rows.append({"examples_per_tool": n, "tool@1": tool_ok / len(gold), "full": full / len(gold)})
        print(f"  {n:2d} examples/tool -> tool@1={tool_ok/len(gold)*100:4.1f}%  "
              f"full-call={full/len(gold)*100:4.1f}%")

    json.dump(rows, open(f"{OUT}/report.json", "w"), indent=2)
    _plot(rows)

    # find the knee: first n within 3 points of the best
    best = max(r["tool@1"] for r in rows)
    knee = next(r["examples_per_tool"] for r in rows if r["tool@1"] >= best - 0.03)
    print(f"\nINSIGHT: desc-only (0) = {rows[0]['tool@1']*100:.0f}% tool@1; "
          f"saturates near best ({best*100:.0f}%) by ~{knee} examples/tool "
          f"-> a handful of example phrasings per tool is enough; more barely helps.")


def _plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [r["examples_per_tool"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(xs, [r["tool@1"] * 100 for r in rows], marker="o", lw=2, label="tool@1")
    ax.plot(xs, [r["full"] * 100 for r in rows], marker="s", lw=2, label="full-call")
    ax.set_xlabel("example usages indexed per tool"); ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(0, 100); ax.set_xticks(xs); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"How much per-tool data does retrieval need? ({len(xs) and ''}coding + 300 distractors)")
    savefig(fig, "15_example_usage_scaling")
    plt.close(fig)


if __name__ == "__main__":
    main()
