#!/usr/bin/env python
"""LocalAgent demo: a retrieval-driven tool-calling agent (Gradio UI + CLI fallback).

Type a request -> the agent retrieves the most relevant tool from the catalog (which includes the
newly added write_file / move_file / google_search / click_link / summarize, plus optional 1000s of
synthetic tools), grounds the argument from your text, and "executes" it. New tools need **zero
retraining** because selection is by retrieval.

  python demos/web/app.py            # launches the Gradio UI (if gradio is installed)
  python demos/web/app.py --cli      # prints example turns (no browser needed)
  python demos/web/app.py --scale 1000   # add 1000 synthetic tools to the catalog
"""

from __future__ import annotations

import argparse

from openlocalagent.agent.demo_tools import curated_examples, curated_specs
from openlocalagent.agent.parser import extract_tool_calls
from openlocalagent.agent.retriever import ToolRetriever
from openlocalagent.agent.tools import ToolRegistry
from openlocalagent.agent.constrained import _tool_bodies


def _stub_result(name: str, args: dict) -> str:
    v = next(iter(args.values()), "")
    return {
        "write_file": f"wrote {v}", "move_file": f"moved {v}", "read_file": f"contents of {v} (…)",
        "google_search": f"top result for ‘{v}’: example.com/{v.replace(' ', '-')}",
        "click_link": f"opened {v}", "summarize": f"summary: ‘{v}’ in one line.",
        "send_email": f"email drafted to {v}", "calendar_event": f"event ‘{v}’ scheduled",
        "slack_send": f"posted to Slack: ‘{v}’", "jira_issue": f"created JIRA-123: ‘{v}’",
        "get_weather": f"{v}: 19°C, cloudy", "run_command": f"$ {v}\n(exit 0)",
        "git_commit": f"[main abc123] {v}", "grep_search": f"3 matches for ‘{v}’",
        "play_music": f"now playing {v}", "set_timer": f"timer set for {v}",
    }.get(name, f"ok({v})")


class DemoAgent:
    def __init__(self, scale: int = 0, k: int = 5):
        specs = curated_specs()
        examples = curated_examples()
        if scale:                                   # add synthetic tools to show catalog scale
            from openlocalagent.data.tool_catalog import build_catalog, gen_usages
            extra = build_catalog(scale, seed=0)
            specs = specs + extra
            for u in gen_usages(extra, "train", per_tool=4, seed=3, paraphrase=True):
                examples.setdefault(u["tool"], []).append(u["prompt"])
        self.specs = {t.name: t for t in specs}
        self.retr = ToolRetriever(specs, examples=examples)
        self.reg = ToolRegistry()
        for t in specs:
            self.reg.register(t, lambda _n=t.name, **kw: _stub_result(_n, kw))
        self.k = k

    def step(self, msg: str) -> str:
        cands = self.retr.retrieve(msg, self.k)
        spec = self.specs[cands[0]]
        body = (_tool_bodies(msg, spec) or [""])[0]
        calls = extract_tool_calls(body)
        if not calls:
            return f"(retrieved {cands}) — no groundable call"
        c = calls[0]
        result = self.reg.dispatch(c.name, c.arguments)
        return (f"🔧 **{c.name}**({c.arguments})\n"
                f"→ {result}\n\n*candidates:* {', '.join(cands)}")


EXAMPLES = [
    "Move the file src/old.py to the archive.",
    "Google the best pizza in town.",
    "Click through to figma.com.",
    "Summarize 'the quarterly earnings report'.",
    "Write to config/prod.yaml.",
    "Send an email to Greta.",
    "What's the weather in Cusco?",
    "Commit with message 'fix the login bug'.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--scale", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    agent = DemoAgent(scale=args.scale, k=args.k)
    n = len(agent.specs)

    if not args.cli:
        try:
            import gradio as gr

            def respond(message, history):
                return agent.step(message)

            gr.ChatInterface(
                respond, title="LocalAgent — retrieval-driven tool calling",
                description=f"A from-scratch byte-level agent over {n} tools "
                            f"(retrieve → ground → execute). New tools need no retraining.",
                examples=EXAMPLES,
            ).launch()
            return
        except Exception as e:
            print(f"(gradio unavailable: {e} — falling back to CLI)\n")

    print(f"LocalAgent demo over {n} tools (retrieval-driven)\n")
    for q in EXAMPLES:
        print(f"> {q}\n  {agent.step(q).splitlines()[0]}")


if __name__ == "__main__":
    main()
