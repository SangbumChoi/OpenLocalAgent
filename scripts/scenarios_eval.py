#!/usr/bin/env python
"""Analyze ToolCaller across real-world tool surface forms (MCP / REST / CLI / SDK).

WHY: a deployed agent doesn't see one clean tool format — it sees MCP server schemas, REST
endpoints, CLI commands with flags, and SDK method calls. Hypothesis: named-JSON args (MCP/REST)
ground more easily than CLI flags / SDK identifiers (ports, hosts, instance ids, keys). This breaks
down full-call accuracy **by modality** to test that, with a distractor sweep and abstention.

  python scripts/scenarios_eval.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from localagent.agent.caller import ToolCaller
from localagent.eval import scenarios_bench as sb
from localagent.figs import savefig

OUT = "runs/scenarios"
SCALES = [0, 200, 1000]


def _by_modality(caller, gold):
    m = defaultdict(lambda: [0, 0, 0])           # modality -> [tool_ok, full_ok, total]
    for q, g in gold:
        mod = sb.MODALITY[g["name"]]
        r = caller.call(q)
        t = r is not None and r.name == g["name"]
        a = t and r.arguments == g["arguments"]
        m[mod][0] += t; m[mod][1] += a; m[mod][2] += 1
    return {k: {"tool@1": v[0] / v[2], "full": v[1] / v[2], "n": v[2]} for k, v in m.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    base = sb.build_tools()
    ex = sb.examples()
    gold = sb.gold_set("eval", seed=7)
    mods = ["MCP", "REST", "CLI", "SDK"]
    print(f"{len(base)} tools across {mods}, {len(gold)} held-out paraphrased queries\n")

    report = {"sweep": []}
    for Nd in SCALES:
        tools, exx = base, dict(ex)
        if Nd:
            from localagent.data.tool_catalog import build_catalog, gen_usages
            extra = build_catalog(Nd, seed=2)
            tools = base + extra
            for u in gen_usages(extra, "train", per_tool=4, seed=3, paraphrase=True):
                exx.setdefault(u["tool"], []).append(u["prompt"])
        bym = _by_modality(ToolCaller(tools, examples=exx), gold)
        overall_full = sum(bym[m]["full"] * bym[m]["n"] for m in bym) / len(gold)
        report["sweep"].append({"distractors": Nd, "n_tools": len(tools),
                                "overall_full": overall_full, "by_modality": bym})
        line = "  ".join(f"{m}={bym[m]['full']*100:.0f}%" for m in mods if m in bym)
        print(f"+{Nd:4d} distractors: full-call by modality -> {line}   (overall {overall_full*100:.0f}%)")

    abst = sum(ToolCaller(base, examples=ex, min_score=0.13).call(q) is None for q in sb.IRRELEVANT)
    report["abstention"] = abst / len(sb.IRRELEVANT)
    print(f"\nabstention (min_score 0.13): {abst/len(sb.IRRELEVANT)*100:.0f}%")
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
    _plot(report["sweep"][0]["by_modality"], mods)
    print(f"report + figure in {OUT}/ and figures/16_*")


def _plot(bym, mods):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    mods = [m for m in mods if m in bym]
    x = np.arange(len(mods)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - w / 2, [bym[m]["tool@1"] * 100 for m in mods], w, label="tool@1", color="#3070c0")
    ax.bar(x + w / 2, [bym[m]["full"] * 100 for m in mods], w, label="full-call", color="#e08020")
    ax.set_xticks(x); ax.set_xticklabels([f"{m}\n(n={bym[m]['n']})" for m in mods])
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 102); ax.grid(alpha=.3, axis="y"); ax.legend()
    ax.set_title("Tool calling by surface form: MCP / REST / CLI / SDK (no training)")
    for i, m in enumerate(mods):
        ax.text(i + w / 2, bym[m]["full"] * 100 + 1, f"{bym[m]['full']*100:.0f}%", ha="center", fontsize=9)
    savefig(fig, "16_scenarios_mcp_rest_cli_sdk")
    plt.close(fig)


if __name__ == "__main__":
    main()
