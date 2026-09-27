#!/usr/bin/env python
"""Analyze ToolCaller on the software/coding tool benchmark (eval/codebench.py).

Reports full-call / tool@1 / args-exact / abstention, a breakdown **by argument count** (how hard
is multi-arg grounding?), and a sweep over distractor-catalog size. Saves a chart + report.

  python scripts/codebench_eval.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from localagent.agent.caller import ToolCaller
from localagent.eval import codebench

OUT = "runs/codebench"
SCALES = [0, 100, 500, 1000]


def _eval(caller, gold):
    tool_ok = args_ok = full = 0
    by_arity = defaultdict(lambda: [0, 0])      # n_args -> [full_correct, total]
    misses = []
    for q, g in gold:
        ar = len(g["arguments"])
        by_arity[ar][1] += 1
        r = caller.call(q)
        t = r is not None and r.name == g["name"]
        a = t and r.arguments == g["arguments"]
        tool_ok += t; args_ok += a; full += a; by_arity[ar][0] += a
        if not a and len(misses) < 8:
            misses.append((q, g, None if r is None else f"{r.name}{r.arguments}"))
    n = len(gold)
    return {"tool@1": tool_ok / n, "args_exact": args_ok / max(1, tool_ok), "full": full / n,
            "by_arity": {k: v[0] / v[1] for k, v in sorted(by_arity.items())},
            "arity_counts": {k: v[1] for k, v in sorted(by_arity.items())}, "misses": misses}


def main():
    os.makedirs(OUT, exist_ok=True)
    base = codebench.build_tools()
    ex = codebench.examples()
    gold = codebench.gold_set("eval", seed=7)
    report = {"n_tools": len(base), "sweep": []}

    print(f"coding tool benchmark: {len(base)} tools, {len(gold)} held-out queries\n")

    for N in SCALES:
        tools, exx = base, dict(ex)
        if N:
            from localagent.data.tool_catalog import build_catalog, gen_usages
            extra = build_catalog(N, seed=2)
            tools = base + extra
            for u in gen_usages(extra, "train", per_tool=4, seed=3, paraphrase=True):
                exx.setdefault(u["tool"], []).append(u["prompt"])
        m = _eval(ToolCaller(tools, examples=exx), gold)
        report["sweep"].append({"N": len(tools), "distractors": N, **{k: v for k, v in m.items() if k != "misses"}})
        ar = "  ".join(f"{k}-arg={v*100:.0f}%" for k, v in m["by_arity"].items())
        print(f"+{N:4d} distractors ({len(tools):4d} tools): full-call={m['full']*100:4.1f}%  "
              f"tool@1={m['tool@1']*100:4.1f}%  args-exact={m['args_exact']*100:4.1f}%  | {ar}")

    # abstention at a tuned threshold
    caller_ms = ToolCaller(base, examples=ex, min_score=0.14)
    abst = sum(caller_ms.call(q) is None for q in codebench.IRRELEVANT) / len(codebench.IRRELEVANT)
    report["abstention@0.14"] = abst
    print(f"\nabstention (min_score 0.14): {abst*100:.0f}%  ({len(codebench.IRRELEVANT)} irrelevant)")

    base_m = _eval(ToolCaller(base, examples=ex), gold)
    print("\nmisses (no distractors):")
    for q, g, got in base_m["misses"]:
        print(f"  {q!r}\n    gold {g['name']}{g['arguments']}\n    got  {got}")
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
    _plot(report, base_m["by_arity"], base_m["arity_counts"])
    print(f"\nreport + chart in {OUT}/")


def _plot(report, by_arity, counts):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    xs = [s["N"] for s in report["sweep"]]
    a1.plot(xs, [s["full"] * 100 for s in report["sweep"]], marker="o", label="full-call")
    a1.plot(xs, [s["tool@1"] * 100 for s in report["sweep"]], marker="s", label="tool@1")
    a1.set_xscale("log"); a1.set_xticks(xs); a1.set_xticklabels(xs)
    a1.set_xlabel("catalog size (coding tools + distractors)"); a1.set_ylabel("accuracy (%)")
    a1.set_ylim(0, 102); a1.grid(alpha=.3); a1.legend(); a1.set_title("Coding tools: accuracy vs scale")
    ks = list(by_arity)
    a2.bar([f"{k}-arg\n(n={counts[k]})" for k in ks], [by_arity[k] * 100 for k in ks],
           color=["#3070c0", "#e08020", "#c03070"][:len(ks)])
    a2.set_ylabel("full-call accuracy (%)"); a2.set_ylim(0, 102); a2.grid(alpha=.3, axis="y")
    a2.set_title("Grounding difficulty by #arguments")
    for i, k in enumerate(ks):
        a2.text(i, by_arity[k] * 100 + 1, f"{by_arity[k]*100:.0f}%", ha="center", fontsize=9)
    fig.suptitle(f"ToolCaller on {report['n_tools']} software/coding tools (no training)")
    fig.tight_layout(); fig.savefig(f"{OUT}/codebench.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
