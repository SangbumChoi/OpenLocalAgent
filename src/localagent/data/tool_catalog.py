"""A large synthetic tool catalog (100s–1000s of tools) for studying tool selection at scale.

Tools are generated combinatorially from verb×noun (e.g. `book_flight`, `cancel_subscription`,
`summarize_report`), each with a one-string-argument schema (a quoted value, so grounding is
trivial and the study isolates *selection*). Usage examples and 2-step multi-turn episodes are
generated for train and test with **disjoint slot values**. This is the "all the use cases" the
agent must learn to route over.
"""

from __future__ import annotations

import json
import random

from localagent.data.schema import ToolSpec

VERBS = ["get", "search", "create", "update", "delete", "list", "book", "cancel", "send",
         "schedule", "analyze", "summarize", "translate", "export", "import", "fetch", "open",
         "share", "archive", "restore", "rename", "duplicate", "sync", "publish", "draft",
         "review", "approve", "assign", "track", "monitor"]
NOUNS = ["flight", "hotel", "invoice", "ticket", "playlist", "recipe", "workout", "contact",
         "meeting", "document", "dataset", "repository", "server", "dashboard", "budget",
         "portfolio", "order", "shipment", "refund", "subscription", "campaign", "lead",
         "candidate", "shift", "expense", "reminder", "note", "bookmark", "alert", "report",
         "metric", "pipeline", "deployment", "incident", "article", "podcast", "video", "photo",
         "album", "event", "task", "project", "message", "channel", "survey"]

VALUES_TRAIN = ["Q3 report", "summer trip", "team offsite", "launch plan", "client demo",
                "weekly sync", "annual budget", "north star", "data migration", "alpha build",
                "user study", "press release", "sales deck", "road trip", "house move",
                "tax filing", "book club", "garden party", "ski week", "code review",
                "design sprint", "hiring loop", "board update", "growth plan", "bug bash"]
VALUES_EVAL = ["spring sale", "winter retreat", "city tour", "beta launch", "vendor call",
               "monthly close", "kickoff deck", "field trip", "audit prep", "fall festival"]

_PATTERNS = [
    "Please {v} the {n} '{val}'.",
    "Can you {v} a {n} called '{val}'?",
    "I need to {v} the {n} '{val}'.",
    "{V} the {n} named '{val}'.",
    "Go ahead and {v} the {n} '{val}'.",
]

# Realistic queries name the object but paraphrase the action verb (people don't say the API verb).
VERB_SYNONYMS = {
    "get": ["show", "fetch", "pull up"], "search": ["find", "look up", "look for"],
    "create": ["make", "set up", "start"], "update": ["edit", "change", "modify"],
    "delete": ["remove", "drop", "trash"], "list": ["show all", "enumerate"],
    "book": ["reserve", "arrange"], "cancel": ["call off", "scrap"], "send": ["dispatch", "forward"],
    "schedule": ["set up", "line up"], "analyze": ["examine", "break down"],
    "summarize": ["recap", "condense"], "translate": ["convert"], "export": ["download", "save out"],
    "import": ["load", "bring in"], "fetch": ["grab", "pull"], "open": ["launch", "bring up"],
    "share": ["pass along", "circulate"], "archive": ["file away", "store"],
    "restore": ["recover", "bring back"], "rename": ["relabel"], "duplicate": ["copy", "clone"],
    "sync": ["synchronize", "refresh"], "publish": ["post", "release"], "draft": ["write up", "compose"],
    "review": ["check", "go over"], "approve": ["sign off on", "greenlight"],
    "assign": ["delegate", "hand off"], "track": ["follow"], "monitor": ["watch", "keep an eye on"],
}


def build_catalog(n: int, seed: int = 0) -> list[ToolSpec]:
    combos = [(v, nn) for v in VERBS for nn in NOUNS]
    random.Random(seed).shuffle(combos)
    tools = []
    for v, nn in combos[:n]:
        tools.append(ToolSpec(
            name=f"{v}_{nn}", description=f"{v} a {nn}",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string", "format": "quoted"}},
                        "required": ["query"]}))
    return tools


def usage(tool: ToolSpec, value: str, rng: random.Random, paraphrase: bool = False) -> tuple[str, dict]:
    v, nn = tool.name.split("_", 1)
    verb = rng.choice(VERB_SYNONYMS.get(v, [v])) if paraphrase else v
    p = rng.choice(_PATTERNS).format(v=verb, V=verb.capitalize(), n=nn, val=value)
    return p, {"name": tool.name, "arguments": {"query": value}}


def gen_usages(tools, split: str = "train", per_tool: int = 2, seed: int = 0, paraphrase=None):
    """One-per-tool (×per_tool) usage examples. Returns list of {prompt, tool, value}.
    Eval defaults to paraphrased (verb synonyms) so retrieval is non-trivial."""
    vals = VALUES_TRAIN if split == "train" else VALUES_EVAL
    para = (split == "eval") if paraphrase is None else paraphrase
    rng = random.Random(seed)
    out = []
    for t in tools:
        for _ in range(per_tool):
            val = rng.choice(vals)
            p, call = usage(t, val, rng, paraphrase=para)
            out.append({"prompt": p, "tool": t.name, "value": val})
    return out


def gen_episodes(tools, n: int, split: str = "train", seed: int = 0):
    """2-step multi-turn episodes: tool A -> response -> follow-up tool B. Returns list of
    [{prompt, tool, value, history}] steps (history = text before that step)."""
    vals = VALUES_TRAIN if split == "train" else VALUES_EVAL
    para = split == "eval"
    rng = random.Random(seed)
    eps = []
    for _ in range(n):
        ta, tb = rng.choice(tools), rng.choice(tools)
        va, vb = rng.choice(vals), rng.choice(vals)
        pa, ca = usage(ta, va, rng, paraphrase=para)
        vb_verb, nb = tb.name.split("_", 1)
        vbb = rng.choice(VERB_SYNONYMS.get(vb_verb, [vb_verb])) if para else vb_verb
        pb = rng.choice([f"Then {vbb} the {nb} '{vb}'.", f"After that, {vbb} the {nb} '{vb}'."])
        hist = f"{pa} [done] "
        eps.append([{"prompt": pa, "tool": ta.name, "value": va, "history": ""},
                    {"prompt": pb, "tool": tb.name, "value": vb, "history": hist}])
    return eps


def dump_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
