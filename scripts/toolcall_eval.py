#!/usr/bin/env python
"""Benchmark the schema-guided ToolCaller on realistic multi-argument tools.

Measures, on paraphrased held-out queries (disjoint slot values):
  tool@1        — correct tool selected
  args-exact    — all arguments grounded exactly (given the right tool)
  full-call     — correct tool AND all args exact
  abstention    — % of irrelevant queries correctly declined (with --min-score)

  python scripts/toolcall_eval.py [--scale 1000] [--min-score 0.15]
"""

from __future__ import annotations

import argparse

from localagent.agent.caller import ToolCaller
from localagent.eval.toolcall_bench import IRRELEVANT, build_tools, examples, gold_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=0, help="pad the catalog with N synthetic tools")
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()

    tools = build_tools()
    ex = examples()
    if args.scale:                                   # distractor tools to test selection at scale
        from localagent.data.tool_catalog import build_catalog, gen_usages
        extra = build_catalog(args.scale, seed=1)
        tools = tools + extra
        for u in gen_usages(extra, "train", per_tool=4, seed=3, paraphrase=True):
            ex.setdefault(u["tool"], []).append(u["prompt"])
    caller = ToolCaller(tools, examples=ex, min_score=args.min_score)
    print(f"benchmark over {len(tools)} tools "
          f"({'+%d synthetic distractors' % args.scale if args.scale else 'realistic only'})\n")

    gold = gold_set("eval", seed=7)
    tool_ok = args_ok = full_ok = 0
    misses = []
    for q, g in gold:
        r = caller.call(q)
        t = r is not None and r.name == g["name"]
        a = t and r.arguments == g["arguments"]
        tool_ok += t; args_ok += a; full_ok += a
        if not a:
            misses.append((q, g, None if r is None else f"{r.name}{r.arguments}"))
    n = len(gold)
    abst = sum(caller.call(q) is None for q in IRRELEVANT)

    print(f"tool@1     : {tool_ok/n*100:5.1f}%")
    print(f"args-exact : {args_ok/max(1,tool_ok)*100:5.1f}%  (of correctly-selected)")
    print(f"full-call  : {full_ok/n*100:5.1f}%  ({full_ok}/{n})")
    print(f"abstention : {abst/len(IRRELEVANT)*100:5.1f}%  ({abst}/{len(IRRELEVANT)} irrelevant declined)")
    if misses:
        print("\nmisses:")
        for q, g, got in misses[:8]:
            print(f"  {q!r}\n    gold {g['name']}{g['arguments']}\n    got  {got}")


if __name__ == "__main__":
    main()
