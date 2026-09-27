"""A small benchmark for schema-guided tool calling on realistic multi-argument tools.

~24 tools across domains (files, dev, comms, calendar, smart-home, finance, media, travel), each
with a real JSON schema (1-3 args, mixed types). Queries are generated from templates with
**disjoint train/eval slot values** and **paraphrased eval verbs** (so retrieval is non-trivial).
Also a set of irrelevant queries for abstention. Used by scripts/toolcall_eval.py.
"""

from __future__ import annotations

import random

from localagent.data.schema import ToolSpec

# name, description, [(arg, schema, train_vals, eval_vals)], [templates with {verb}/{a0}/{a1}/...]
_TOOLS = [
    ("move_file", "move or rename a file",
     [("source", {"type": "string", "format": "path"}, ["src/a.py", "lib/x.js"], ["app/m.py", "db/q.sql"]),
      ("dest", {"type": "string", "format": "path"}, ["backup/a.py", "old/x.js"], ["arch/m.py", "tmp/q.sql"])],
     ["{verb} {a0} to {a1}.", "Please {verb} {a0} to {a1}."], "move", ["relocate", "transfer"]),
    ("read_file", "read a file",
     [("path", {"type": "string", "format": "path"}, ["src/main.py", "README.md"], ["api/routes.go", "web/app.tsx"])],
     ["{verb} {a0}.", "{verb} the file {a0}."], "open", ["read", "show"]),
    ("grep_search", "search the codebase for a pattern",
     [("pattern", {"type": "string", "format": "quoted"}, ["TODO", "def main"], ["API_KEY", "raise ValueError"])],
     ["{verb} for '{a0}'.", "{verb} the code for '{a0}'."], "grep", ["search", "find"]),
    ("run_command", "run a shell command",
     [("command", {"type": "string", "format": "quoted"}, ["ls -la", "npm test"], ["git pull", "make build"])],
     ["{verb} '{a0}'.", "{verb} the command '{a0}'."], "run", ["execute", "exec"]),
    ("git_commit", "make a git commit",
     [("message", {"type": "string", "format": "quoted"}, ["fix bug", "add tests"], ["tidy imports", "bump deps"])],
     ["{verb} with message '{a0}'.", "{verb} '{a0}'."], "commit", ["check in", "save"]),
    ("send_email", "send an email to someone",
     [("recipient", {"type": "string"}, ["Alice", "Bob"], ["Greta", "Mateo"]),
      ("subject", {"type": "string", "format": "quoted"}, ["status", "lunch"], ["Q3 plan", "demo"])],
     ["{verb} {a0} with subject '{a1}'.", "{verb} an email to {a0} about '{a1}'."], "email", ["message", "write"]),
    ("schedule_meeting", "schedule a calendar meeting",
     [("title", {"type": "string", "format": "quoted"}, ["standup", "sync"], ["retro", "kickoff"]),
      ("time", {"type": "string"}, ["9am", "noon"], ["3pm", "10am"])],
     ["{verb} '{a0}' at {a1}.", "{verb} a meeting '{a0}' for {a1}."], "schedule", ["set up", "book"]),
    ("set_reminder", "set a reminder",
     [("task", {"type": "string"}, ["call mom", "pay rent"], ["water plants", "book flight"])],
     ["{verb} to {a0}.", "{verb} me to {a0}."], "remind", ["remind", "remind"]),
    ("set_thermostat", "set the thermostat temperature",
     [("temperature", {"type": "integer"}, ["70", "68"], ["72", "65"]),
      ("unit", {"type": "string", "enum": ["c", "f"]}, ["f", "c"], ["f", "c"])],
     ["{verb} the thermostat to {a0} {a1}.", "{verb} it to {a0} {a1}."], "set", ["set", "set"]),
    ("toggle_light", "turn a light on or off",
     [("room", {"type": "string"}, ["kitchen", "office"], ["bedroom", "garage"]),
      ("state", {"type": "boolean"}, ["on", "off"], ["on", "off"])],
     ["Turn {a1} the {a0} light.", "Switch {a1} the {a0} light."], "turn", ["turn", "switch"]),
    ("convert_currency", "convert an amount of money",
     [("amount", {"type": "number"}, ["100", "50"], ["250", "75"]),
      ("to", {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"]}, ["EUR", "USD"], ["GBP", "JPY"])],
     ["{verb} {a0} to {a1}.", "{verb} {a0} into {a1}."], "convert", ["change", "exchange"]),
    ("get_weather", "get the weather for a city",
     [("city", {"type": "string"}, ["Paris", "Tokyo"], ["Cusco", "Oslo"])],
     ["{verb} the weather in {a0}?", "What's the weather in {a0}?"], "get", ["check", "show"]),
    ("play_music", "play a song",
     [("song", {"type": "string"}, ["Yesterday", "Africa"], ["Imagine", "Clocks"])],
     ["{verb} {a0}.", "{verb} the song {a0}."], "play", ["play", "put on"]),
    ("open_url", "open a website in the browser",
     [("url", {"type": "string", "format": "url"}, ["github.com", "python.org"], ["figma.com", "openai.com"])],
     ["{verb} {a0}.", "{verb} the website {a0}."], "open", ["visit", "go to"]),
    ("translate_text", "translate text into a language",
     [("text", {"type": "string", "format": "quoted"}, ["hello", "thanks"], ["goodbye", "welcome"]),
      ("language", {"type": "string"}, ["French", "German"], ["Spanish", "Italian"])],
     ["{verb} '{a0}' to {a1}.", "{verb} '{a0}' into {a1}."], "translate", ["translate", "convert"]),
    ("create_ticket", "create a support/Jira ticket",
     [("summary", {"type": "string", "format": "quoted"}, ["login bug", "slow page"], ["broken link", "data loss"])],
     ["{verb} a ticket '{a0}'.", "{verb} a Jira issue for '{a0}'."], "create", ["file", "open"]),
    ("book_flight", "book a flight to a city",
     [("destination", {"type": "string"}, ["Rome", "Cairo"], ["Lima", "Seoul"])],
     ["{verb} a flight to {a0}.", "{verb} me a flight to {a0}."], "book", ["book", "reserve"]),
    ("delete_file", "delete a file",
     [("path", {"type": "string", "format": "path"}, ["tmp/x.log", "cache/y.tmp"], ["build/z.o", "out/w.bin"])],
     ["{verb} {a0}.", "{verb} the file {a0}."], "delete", ["remove", "delete"]),
]


def build_tools(defs=None) -> list[ToolSpec]:
    out = []
    for name, desc, args, templates, _v, _s in (defs or _TOOLS):
        props = {a: sch for a, sch, *_ in args}
        out.append(ToolSpec(name, desc, {"type": "object", "properties": props,
                                          "required": list(props)}))
    return out


def examples(defs=None) -> dict:
    """Example phrasings per tool for retrieval — including the *verb synonyms*, so the index
    bridges the paraphrase gap (a query that says 'execute' matches the tool whose API verb is
    'run'). This is the example-augmented-retrieval trick from scripts/tool_scale_analysis.py."""
    ex = {}
    for name, desc, args, templates, verb, syn in (defs or _TOOLS):
        vals = [a[2][0] for a in args]                # first train value per arg
        fills = {f"a{i}": v for i, v in enumerate(vals)}
        for vb in [verb, *syn]:                        # train verb + every synonym
            for t in templates:
                ex.setdefault(name, []).append(t.format(verb=vb, **fills))
    return ex


def gold_set(split: str = "eval", seed: int = 0, defs=None):
    """(query, gold_call) pairs. Eval paraphrases the verb (synonyms) and uses disjoint values."""
    rng = random.Random(seed)
    rows = []
    for name, desc, args, templates, verb, syn in (defs or _TOOLS):
        for t in templates:
            vals, gold = [], {}
            for a, sch, tr, ev in args:
                v = rng.choice(ev if split == "eval" else tr)
                vals.append(v)
                if sch.get("type") == "integer":
                    gold[a] = int(v)
                elif sch.get("type") == "number":
                    gold[a] = float(v)
                elif sch.get("type") == "boolean":
                    gold[a] = v == "on"
                else:
                    gold[a] = v
            vb = rng.choice(syn) if split == "eval" else verb
            rows.append((t.format(verb=vb, **{f"a{i}": x for i, x in enumerate(vals)}),
                         {"name": name, "arguments": gold}))
    return rows


IRRELEVANT = ["Tell me a joke.", "How are you today?", "I love you.", "What's your name?",
              "Sing me a song about nothing.", "asdf qwer zxcv", "Thanks for your help!",
              "Are you a robot?"]
