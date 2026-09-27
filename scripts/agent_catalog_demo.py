#!/usr/bin/env python
"""End-to-end agent over a large tool catalog: retrieve top-k -> ground args -> dispatch.

No trained LM needed — selection is by the retriever (indexed on example usages) and arguments are
schema-grounded. Shows the live agent picking the right tool out of 1000 and calling it.

  python scripts/agent_catalog_demo.py [--n 1000]
"""

from __future__ import annotations

import argparse

from localagent.agent.retriever import ToolRetriever
from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.data.tool_catalog import build_catalog, gen_usages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    tools = build_catalog(args.n, seed=0)
    examples = {}
    for u in gen_usages(tools, "train", per_tool=4, seed=3, paraphrase=True):
        examples.setdefault(u["tool"], []).append(u["prompt"])
    retr = ToolRetriever(tools, examples=examples)

    reg = ToolRegistry()
    for t in tools:
        reg.register(t, lambda **kw: {"status": "ok", **kw})   # stub executor
    agent = Agent(reg, catalog=tools, retriever=retr, retrieve_k=args.k)
    print(f"agent over {len(tools)} tools (retrieve top-{args.k} -> ground -> dispatch)\n")

    # a few live turns (paraphrased — the API verb is NOT in the query)
    for q in ["Please reserve the flight 'city tour'.",
              "Can you remove the subscription 'beta launch'?",
              "Recap the report 'audit prep'.",
              "Pull up the dashboard 'monthly close'."]:
        print(f"> {q}\n  {agent.chat(q)}")

    # end-to-end accuracy on held-out paraphrased queries
    test = gen_usages(tools, "eval", per_tool=1, seed=7)
    ok = tool_ok = 0
    for u in test:
        tool_ok += retr.retrieve(u["prompt"], 1)[0] == u["tool"]
        out = agent.chat(u["prompt"])
        ok += out.startswith(f"[{u['tool']}(") and u["value"] in out
    n = len(test)
    print(f"\nend-to-end over {len(tools)} tools: tool@1={tool_ok/n*100:.1f}%  "
          f"full-call(correct tool+arg)={ok/n*100:.1f}%  (n={n} held-out paraphrased)")


if __name__ == "__main__":
    main()
